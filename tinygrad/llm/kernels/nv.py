"""
Hand-written Ampere fp16 GEMM (nv_gemm.cu) plugged into tinygrad with Tensor.custom_kernel, for NV and CUDA devices.

  from tinygrad.llm.kernels.nv import ampere_matmul
  c = ampere_matmul(a, b)          # a: (M,K) half, b: (K,N) half -> (M,N) half, fp32 accumulate

nv_linear(x, w) is the LLM's Linear layer: x @ w.T with w stored (out, in), read transposed in place, fp32 output.

Falls back to a.matmul(b) when the device/dtype/shape isn't supported. Differentiable: the backward is two more custom GEMMs
that read B and A transposed in place (TRANS_B / TRANS_A in the kernel).
With Cfg.SK (default), tiles that would leave the last wave partly empty are split along K across every SM (stream-K),
which needs a small fp32 workspace and a zeroed flag buffer per call.
Run as a script for a correctness check: DEV=NV python -m tinygrad.llm.kernels.nv
"""
import ctypes, functools, hashlib, itertools, pathlib
from typing import Any, Callable, Sequence, cast
from dataclasses import dataclass
from tinygrad import Tensor, dtypes, Device, Context
from tinygrad.dtype import DType
from tinygrad.helpers import getenv, prod
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates

_DIR = pathlib.Path(__file__).parent
COMMON = (_DIR / "nv_common.cuh").read_text()
SRC = COMMON + (_DIR / "nv_gemm.cu").read_text()
FLASH_SRC = COMMON + (_DIR / "nv_flash.cu").read_text()
DECODE_MERGE = (_DIR / "nv_decode_merge.cuh").read_text()
DECODE_SRC = COMMON + DECODE_MERGE + (_DIR / "nv_decode.cu").read_text()
QUANT_SRC = COMMON + DECODE_MERGE + (_DIR / "nv_quant.cu").read_text()
MMQ_SRC = COMMON + (_DIR / "nv_mmq.cu").read_text()

@dataclass(frozen=True)
class Cfg:
  # defaults: the best config on a 3080 (256x128 block tile, 8 warps of 64x64, 4 stages = 96 KB of shared memory)
  BM: int = 256
  BN: int = 128
  BK: int = 32
  WM: int = 64
  WN: int = 64
  STAGES: int = 4
  GROUP_M: int = 8
  SK: bool = True  # stream-K the tail wave
  @property
  def threads(self) -> int: return (self.BM // self.WM) * (self.BN // self.WN) * 32
  @property
  def smem(self) -> int: return self.STAGES * (self.BM * self.BK + self.BK * self.BN) * 2
  def __str__(self):
    tile = f"{self.BM}x{self.BN}x{self.BK}_w{self.WM}x{self.WN}_s{self.STAGES}"
    return tile + (f"_g{self.GROUP_M}" if self.GROUP_M else "") + ("_sk" if self.SK else "")
  @property
  def blocks_per_sm(self) -> int:
    # resident blocks per SM on sm_86: 64K registers (assume the kernel uses up to 256 per thread, it's register heavy),
    # 48 warps, and 100 KB shared memory with 1 KB reserved per block
    return max(1, min(16, 48 * 32 // self.threads, 65536 // (self.threads * 256), (100 * 1024) // (self.smem + 1024)))

Glu = tuple[float, float|None, float]  # model.gated_activation's SwiGLU: (alpha, limit or None, up_bias)
def _glu_defines(glu:Glu) -> tuple[str, dict[str, Any]]:
  # kernel name suffix and #defines for a SwiGLU applied in a kernel (nv_gemm.cu's epilogue, nv_quant.cu's QUANTIZE)
  alpha, limit, up_bias = glu
  defines:dict[str, Any] = {"GLU": 1, "GLU_ALPHA": f"{alpha!r}f", "GLU_UP_BIAS": f"{up_bias!r}f"}
  if limit is not None: defines["GLU_LIMIT"] = f"{limit!r}f"
  return "_glu" + ("" if glu == (1.0, None, 0.0) else hashlib.md5(repr(glu).encode()).hexdigest()[:6]), defines

DEFAULT = Cfg()
MAX_SMEM = 99 * 1024  # per-block shared memory limit on sm_86/sm_89 (sm_80 allows 163 KB)

# Sustained (power capped) benchmarks on a 3080: the big tile with stream-K was the best or tied on every K >= 4096 shape
# (bench/logs/soak_sweep.log); big tiles move less data per FLOP, so the card holds a higher clock under its power limit.
# With a short K (fewer k-steps per tile) the per-tile pipeline fill and epilogue show, and two 48 KB blocks per SM that
# overlap each other's win: 128x128 (4 warps, 3 stages) beat the big tile by 4-15% for K <= 2048 (bench/logs/harness_k_crossover.log,
# harness_tn_shortk.log). That's the weight gradient in training, where K is the token count.
BIG, MID, SMALL = Cfg(), Cfg(BM=128, STAGES=3), Cfg(BM=128, BN=64, WN=32)
# few rows (LLM prefill chunks): 32 or 64 row tiles, and stream-K spreads the few column tiles across every SM
ROWS32, ROWS64 = Cfg(BM=32, BN=128, WM=32, WN=32, STAGES=4), Cfg(BM=64, BN=128, WM=32, WN=64, STAGES=4)

def auto_cfg(M:int, N:int, K:int) -> Cfg:
  cands = (ROWS32,) if M <= 32 else (ROWS64,) if M <= 64 else (BIG, MID, SMALL) if K >= 4096 else (MID, SMALL)
  return next((c for c in cands if M % c.BM == 0 and N % c.BN == 0 and K % c.BK == 0), DEFAULT)

@functools.cache
def num_sms(device:str) -> int:
  with Context(ALLOW_DEVICE_USAGE=1): dev:Any = Device[device]  # CUDADevice or NVDevice
  if device.split(":")[0] == "CUDA":
    from tinygrad.runtime.autogen import cuda
    dev.api.cuDeviceGetAttribute(ctypes.byref(v:=ctypes.c_int()), cuda.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev.cu_device)
    return v.value
  if device.split(":")[0] != "NV": raise RuntimeError(f"num_sms: unsupported device {device}")
  # NV: count the enabled TPCs (num_gpcs * num_tpc_per_gpc is the full die, the 3080's GA102 has 84 SMs but only 68 enabled)
  from tinygrad.runtime.ops_nv import nv_gpu
  masks = [dev.iface.rm_control(dev.subdevice, nv_gpu.NV2080_CTRL_CMD_GR_GET_TPC_MASK, nv_gpu.NV2080_CTRL_GR_GET_TPC_MASK_PARAMS(gpcId=i)).tpcMask
           for i in range(dev.num_gpcs)]
  return sum(bin(m).count("1") for m in masks) * dev.num_sm_per_tpc

@dataclass(frozen=True)
class Split:
  sk_p: int        # concurrent block slots (0: no stream-K)
  sk_tiles: int    # tiles shared out by K iteration
  sk_workers: int  # stream-K blocks, launched first
  dp_tiles: int    # whole tiles, one block each
  @property
  def grid(self) -> int: return self.sk_workers + self.dp_tiles

SK_OVERHEAD = 0.1  # stream-K costs ~10% extra on the work it splits (restarts, fp32 partials through L2, and the power they draw)

def split(M:int, N:int, K:int, cfg:"Cfg|MmqCfg", sms:int) -> Split:
  # mirrors the constexprs in ampere_gemm.cu
  p, tiles, kt = (sms * cfg.blocks_per_sm if cfg.SK else 0), (M // cfg.BM) * (N // cfg.BN), K // cfg.BK
  sk_tiles = 0 if p == 0 or tiles % p == 0 else (tiles % p + p if tiles >= 2 * p else tiles)
  # only when it beats whole tiles in waves: on a 3080 that's 4096^3 (7.5 waves, +2%) and 512x4096x6144 (1.4 waves, +27%),
  # but not 2048^3 (1.9 waves, -4%) or 512x14336x4096 (0.94 waves, -1.4%)
  if sk_tiles and tiles / p + SK_OVERHEAD * sk_tiles / p >= -(-tiles // p): return Split(0, 0, 0, tiles)
  return Split(p, sk_tiles, min(sk_tiles * kt, p), tiles - sk_tiles)

def _dims(a:Tensor, b:Tensor, ta:bool, tb:bool) -> tuple:
  # (M, K, K', N) of op(a) @ op(b), where op transposes when the flag is set
  (M, K), (K2, N) = a.shape[::-1] if ta else a.shape, b.shape[::-1] if tb else b.shape
  return M, K, K2, N

@functools.cache
def nv_gemm_supported(device:str|tuple[str, ...]|None) -> bool:
  # the kernel is CUDA C, so the device has to compile CUDA C (not the PTX renderer), on sm_80+
  if getenv("DISABLE_NV_GEMM") or not isinstance(device, str) or device.split(":")[0] not in {"NV", "CUDA"}: return False
  from tinygrad.renderer.cstyle import CUDARenderer
  with Context(ALLOW_DEVICE_USAGE=1):
    r = Device[device].renderer
    return isinstance(r, CUDARenderer) and (arch:=r.target.arch).startswith("sm_") and int(arch[3:5]) >= 80

def supported(a:Tensor, b:Tensor, cfg:Cfg=DEFAULT, ta:bool=False, tb:bool=False) -> bool:
  if not nv_gemm_supported(a.device) or a.device != b.device: return False
  if a.dtype != b.dtype or a.dtype not in (dtypes.half, dtypes.bfloat16) or a.ndim != 2 or b.ndim != 2: return False
  if cfg.smem > MAX_SMEM: return False
  M, K, K2, N = _dims(a, b, ta, tb)
  return all(isinstance(x, int) for x in (M, K, N)) and K == K2 and M % cfg.BM == 0 and N % cfg.BN == 0 and K % cfg.BK == 0

@functools.cache
def _source(M:int, N:int, K:int, cfg:Cfg, sk_p:int, ta:bool, tb:bool, f32:bool, bf16:bool, glu:Glu|None=None) -> tuple[str, str]:
  name = f"ampere_gemm_{'t' if ta else 'n'}{'t' if tb else 'n'}{'_bf16' if bf16 else ''}{'_f32' if f32 else ''}_{M}_{N}_{K}_{cfg}" + \
         (f"{sk_p}" if sk_p else "")
  defines:dict[str, Any] = dict(KNAME=name, M_DIM=M, N_DIM=N, K_DIM=K, BM=cfg.BM, BN=cfg.BN, BK=cfg.BK, WM=cfg.WM, WN=cfg.WN,
                                STAGES=cfg.STAGES, GROUP_M=cfg.GROUP_M, SK_P=sk_p, TRANS_A=int(ta), TRANS_B=int(tb), OUT_F32=int(f32),
                                BF16=int(bf16))
  if glu is not None:
    suffix, gdefs = _glu_defines(glu)
    name = defines["KNAME"] = name + suffix
    defines.update(gdefs)
  return name, "".join(f"#define {k} {v}\n" for k, v in defines.items()) + SRC

def _program(C:UOp, A:UOp, B:UOp, WS:UOp, FLAGS:UOp, cfg:Cfg, sp:Split, ta:bool, tb:bool, glu:Glu|None=None) -> UOp:
  (M, NC), K = C.shape, (A.shape[0] if ta else A.shape[1])
  N = 2 * NC if glu is not None else NC  # with glu, C holds the activation of each (gate, up) column pair
  name, src = _source(M, N, K, cfg, sp.sk_p, ta, tb, C.dtype == dtypes.float32, A.dtype == dtypes.bfloat16, glu)
  threads, blocks = UOp.special(cfg.threads, "lidx0"), UOp.special(sp.grid, "gidx0")
  io = 2 * (M * K + K * N) + C.dtype.itemsize * M * NC
  est = Estimates(ops=2 * M * N * K, lds=io, mem=io)
  sink = UOp.sink(C.base, A.base, B.base, WS.base, FLAGS.base, threads, blocks, arg=KernelInfo(name, estimates=est, dyn_shared_mem=cfg.smem))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

@functools.cache
def _flags(device:str) -> Tensor:
  # Stream-K handoff flags. They must be 0 when a launch starts; every launch resets each flag it sets before it finishes, so
  # one zeroed buffer per device is reused (a fresh zeros() per call costs an extra kernel launch, ~5% on a 2048^3 GEMM).
  # This relies on kernels on a device running in order, one after another, as tinygrad's NV (dependent QMDs) and
  # CUDA (one stream) runtimes do. A launch that faults can leave a flag set; the next one would then read a stale partial.
  # created eagerly even while a @function is being traced (the LLM's blocks): it doesn't depend on the traced graph, it's
  # captured like a weight
  with Context(ALLOW_DEVICE_USAGE=1): return Tensor.zeros(MAX_SK_WORKERS, dtype=dtypes.int32, device=device).contiguous().realize()
MAX_SK_WORKERS = 4096  # >= SMs x blocks per SM on any GPU this runs on

def _grad(dc:UOp, call:UOp, ta:bool, tb:bool) -> tuple:
  # C = op(A) @ op(B):  d op(A) = dC @ op(B)^T  and  d op(B) = op(A)^T @ dC, transposed back when the operand was.
  # Every case is another op(x) @ op(y) GEMM on the stored tensors, so nothing is transposed in memory.
  _, a, b, _, _ = (Tensor(u) for u in call.src[1:])
  g = Tensor(dc).cast(a.dtype)
  da = _gemm(b, g, tb, True) if ta else _gemm(g, b, False, not tb)
  db = _gemm(g, a, True, ta) if tb else _gemm(a, g, not ta, False)
  return (None, da.uop, db.uop, None, None)

def _gemm(a:Tensor, b:Tensor, ta:bool=False, tb:bool=False, cfg:Cfg|None=None, out_dtype:DType|None=None, glu:Glu|None=None) -> Tensor:
  """op(a) @ op(b) with the custom kernel (op transposes when ta/tb is set), or tinygrad's matmul if it's unsupported.
  a and b are both fp16 or both bf16. out_dtype is their dtype (default) or float32; the accumulator is always fp32.
  With glu, the product's columns are interleaved (gate, up) pairs and the result (M, N/2) is their SwiGLU (no gradient)."""
  out_dtype = out_dtype or a.dtype
  if cfg is None and a.ndim == 2 and b.ndim == 2 and all(isinstance(x, int) for x in _dims(a, b, ta, tb)):
    M, K, _, N = _dims(a, b, ta, tb)
    cfg = auto_cfg(M, N, K)
  if cfg is None or not supported(a, b, cfg, ta, tb):
    out = (a.T if ta else a).matmul(b.T if tb else b)
    if glu is None: return out.cast(out_dtype)
    alpha, limit, up_bias = glu
    g, u = out.float()[:, 0::2], out.float()[:, 1::2]
    if limit is not None: g, u = g.minimum(limit), u.clip(-limit, limit)
    return (g * (g * alpha).sigmoid() * (u + up_bias)).cast(out_dtype)
  a, b = a.contiguous(), b.contiguous()
  M, K, _, N = _dims(a, b, ta, tb)
  sp = split(M, N, K, cfg, num_sms(a.device) if cfg.SK else 0)
  assert sp.sk_workers <= MAX_SK_WORKERS
  assert out_dtype in (a.dtype, dtypes.float32), out_dtype
  out = Tensor.invalids(M, N // 2 if glu is not None else N, dtype=out_dtype, device=a.device)
  # stream-K scratch: fp32 partial tiles (one per worker), and the per-device flag buffer (one-element dummy when stream-K is off)
  ws = Tensor.empty(sp.sk_workers * cfg.BM * cfg.BN if sp.sk_workers else 1, dtype=dtypes.float32, device=a.device)
  flags = _flags(a.device) if sp.sk_workers else Tensor.empty(1, dtype=dtypes.int32, device=a.device)
  return Tensor.custom_kernel(out, a, b, ws, flags, fxn=functools.partial(_program, cfg=cfg, sp=sp, ta=ta, tb=tb, glu=glu),
                              grad_fxn=functools.partial(_grad, ta=ta, tb=tb) if glu is None else None)[0]

# ******** implicit-GEMM convolution (nv_gemm.cu with CONV) ********

@dataclass(frozen=True)
class ConvShape:
  nb:int; cin:int; h:int; w:int; cout:int; kh:int; kw:int; stride:int; pad:int  # noqa: E702
  @property
  def oh(self) -> int: return (self.h + 2 * self.pad - self.kh) // self.stride + 1
  @property
  def ow(self) -> int: return (self.w + 2 * self.pad - self.kw) // self.stride + 1
  @property
  def m(self) -> int: return self.nb * self.oh * self.ow

def _conv_program(Y:UOp, X:UOp, Wt:UOp, WS:UOp, FLAGS:UOp, cfg:Cfg, cs:ConvShape) -> UOp:
  Mp, K = -(-cs.m // cfg.BM) * cfg.BM, cs.kh * cs.kw * cs.cin
  name = f"ampere_conv_{cs.nb}x{cs.cin}x{cs.h}x{cs.w}_{cs.cout}x{cs.kh}x{cs.kw}_s{cs.stride}p{cs.pad}_{cfg}"
  defines = dict(KNAME=name, M_DIM=Mp, N_DIM=cs.cout, K_DIM=K, BM=cfg.BM, BN=cfg.BN, BK=cfg.BK, WM=cfg.WM, WN=cfg.WN, STAGES=cfg.STAGES,
                 GROUP_M=cfg.GROUP_M, SK_P=0, TRANS_A=0, TRANS_B=1, OUT_F32=0, BF16=0, CONV=1, CONV_M=cs.m, CONV_H=cs.h, CONV_W=cs.w,
                 CONV_CIN=cs.cin, CONV_KW=cs.kw, CONV_OH=cs.oh, CONV_OW=cs.ow, CONV_S=cs.stride, CONV_P=cs.pad)
  src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + SRC
  threads, blocks = UOp.special(cfg.threads, "lidx0"), UOp.special((Mp // cfg.BM) * (cs.cout // cfg.BN), "gidx0")
  io = 2 * (cs.nb * cs.cin * cs.h * cs.w + cs.cout * K + cs.m * cs.cout)
  est = Estimates(ops=2 * cs.m * cs.cout * K, lds=io, mem=io)
  sink = UOp.sink(Y.base, X.base, Wt.base, WS.base, FLAGS.base, threads, blocks, arg=KernelInfo(name, estimates=est, dyn_shared_mem=cfg.smem))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

CONV_BIG, CONV_N64 = Cfg(SK=False), Cfg(BM=128, BN=64, WN=32, STAGES=4, SK=False)

def nv_conv2d(x:Tensor, w:Tensor, stride:int=1, padding:int=0) -> Tensor|None:
  """conv2d of fp16 NCHW x with w (Cout, Cin, KH, KW) as an implicit GEMM on the tensor cores (fp32 accumulate, fp16 out), or None
  if unsupported (groups/dilation, Cin not a multiple of 32, Cout not of 64). The input and weights are transposed to channels-last
  first (each im2col chunk is 8 channels of one pixel); the output is written NCHW."""
  if not nv_gemm_supported(x.device) or x.dtype != dtypes.half or w.dtype != dtypes.half or x.ndim != 4 or w.ndim != 4: return None
  if x.device != w.device or x.uop.axis is not None or w.uop.axis is not None or not all(isinstance(v, int) for v in (*x.shape, *w.shape)):
    return None
  (nb, cin, h, wd), (cout, cin2, kh, kw) = x.shape, w.shape
  cs = ConvShape(int(nb), int(cin), int(h), int(wd), int(cout), int(kh), int(kw), stride, padding)
  if cin2 != cin or cs.cin % 32 or cs.cout % 64 or cs.oh <= 0 or cs.ow <= 0: return None
  if (o:=getenv("NV_CONV_CFG", "")):  # BM,BN,BK,WM,WN,STAGES (tuning)
    cfg = Cfg(**{k: int(v) for k, v in zip(("BM", "BN", "BK", "WM", "WN", "STAGES"), o.split(","))}, SK=False)
  else: cfg = CONV_BIG if cs.cout % 128 == 0 and cs.m >= 4 * 1024 else CONV_N64
  if cs.cout % cfg.BN or cfg.smem > MAX_SMEM: return None
  xh = x.permute(0, 2, 3, 1).contiguous()
  wk = w.permute(0, 2, 3, 1).reshape(cs.cout, cs.kh * cs.kw * cs.cin).contiguous()
  dummy = Tensor.empty(1, dtype=dtypes.float32, device=x.device), Tensor.empty(1, dtype=dtypes.int32, device=x.device)
  out = Tensor.empty(cs.nb, cs.cout, cs.oh, cs.ow, dtype=dtypes.half, device=x.device)
  return Tensor.custom_kernel(out, xh, wk, *dummy, fxn=functools.partial(_conv_program, cfg=cfg, cs=cs))[0]

NV_LINEAR_MIN_ROWS = 16  # below this (decode), tinygrad's own GEMV is the right kernel

def nv_linear(x:Tensor, weight:Tensor, dequant:Callable[[Tensor], Tensor|None]|None=None, glu:Glu|None=None,
              out_dtype:DType|None=None) -> Tensor|None:
  """x @ weight.T for a Linear layer (weight: (out, in) fp16 or bf16), or None if the custom GEMM doesn't apply.
  x may be fp32 (it's cast to the weight dtype for the tensor cores); the result is in x's dtype (or out_dtype), accumulated in fp32.
  A symbolic token count is padded to its max so the kernel sees static shapes, and the padding is sliced off.
  dequant(x2) makes the fp16 weight from quantized blocks instead (weight then only gives the shape), given the GEMM's input.
  With glu, weight's rows are interleaved (gate, up) pairs and the result (..., out/2) is their SwiGLU in the weight's dtype."""
  if weight.dtype not in (dtypes.half, dtypes.bfloat16) or weight.ndim != 2 or weight.uop.axis is not None: return None
  if x.dtype not in (weight.dtype, dtypes.float32): return None
  if out_dtype is None: out_dtype = weight.dtype if glu is not None else x.dtype
  if out_dtype not in (weight.dtype, dtypes.float32): return None
  if not nv_gemm_supported(weight.device) or x.device != weight.device: return None
  N, K = weight.shape
  if not isinstance(N, int) or not isinstance(K, int) or x.shape[-1] != K: return None
  xs = x if isinstance(x.numel(), int) else x.pad_to(x.max_shape)
  M = prod(xs.shape[:-1])
  if not isinstance(M, int) or M < NV_LINEAR_MIN_ROWS: return None
  # rows are padded up to the tile: take the first config (in auto_cfg's order) that pads M by at most 1/8, else the least padding
  cands = [c for c in ((ROWS32,) if M <= 32 else (ROWS64,) if M <= 64 else (BIG, MID, SMALL, ROWS64) if K >= 4096 else (MID, SMALL, ROWS64))
           if N % c.BN == 0 and K % c.BK == 0]
  if not cands: return None
  cfg = next((c for c in cands if (-M % c.BM) * 8 <= M), min(cands, key=lambda c: -M % c.BM))
  Mp = -(-M // cfg.BM) * cfg.BM
  x2 = xs.reshape(M, K).cast(weight.dtype)
  if Mp != M: x2 = x2.pad(((0, Mp - M), None))
  if dequant is not None:
    if (w:=dequant(x2.contiguous())) is None: return None
    weight = w
  out = _gemm(x2, weight, tb=True, cfg=cfg, out_dtype=out_dtype, glu=glu)
  if Mp != M: out = out[:M]
  NO = N // 2 if glu is not None else N
  out = out.reshape(*xs.shape[:-1], NO)
  return out if xs is x else out.shrink(tuple((0, s) for s in (*x.shape[:-1], NO)))

# ******** causal flash attention on the KV cache (prefill) ********

@dataclass(frozen=True)
class FlashCfg:
  WARPS: int = 4   # 16 query rows per warp
  BN: int = 64     # keys per pipeline stage
  STAGES: int = 2
  def smem(self, hd:int) -> int: return max(2 * self.STAGES * self.BN, 16 * self.WARPS) * hd * 2

def _flash_program(O:UOp, Q:UOp, KV:UOp, P:UOp, cfg:FlashCfg) -> UOp:
  NB, TQ, NH, HD = (int(x) for x in O.shape)
  NBK, NKV, NCTX = int(KV.shape[1]), int(KV.shape[2]), int(KV.shape[3])
  name = f"nv_flash_attn_{NB}{f'of{NBK}' if NBK != NB else ''}_{NH}_{NKV}_{TQ}_{NCTX}_{HD}_w{cfg.WARPS}_n{cfg.BN}_s{cfg.STAGES}"
  defines = dict(KNAME=name, NB=NB, NBK=NBK, NH=NH, NKV=NKV, TQ=TQ, NCTX=NCTX, HD=HD, WARPS=cfg.WARPS, BN=cfg.BN, STAGES=cfg.STAGES)
  src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + FLASH_SRC
  threads, qblocks, heads = UOp.special(32 * cfg.WARPS, "lidx0"), UOp.special(TQ // (16 * cfg.WARPS), "gidx0"), UOp.special(NB * NH, "gidx1")
  # causal: about half of the TQ x (prefix + TQ) scores; the KV length is only known at runtime, so this counts the chunk itself
  io = (4 + 2) * NB * NH * TQ * HD  # fp32 out, fp16 queries (the cache reads depend on the runtime length)
  est = Estimates(ops=2 * 2 * NB * NH * TQ * TQ * HD // 2, lds=io, mem=io)
  sink = UOp.sink(O.base, Q.base, KV.base, P.base, threads, qblocks, heads, arg=KernelInfo(name, estimates=est, dyn_shared_mem=cfg.smem(HD)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

def _int_tensor(x:int|UOp) -> Tensor|None:
  # a symbolic int (sums/products of bound Variables and consts, e.g. start_pos + toks) as a 1-element int32 Tensor
  # (a scalar const, not Tensor([x]): a list is a host copy in every call)
  if isinstance(x, int): return Tensor(x, dtype=dtypes.int32).reshape(1)
  if x.op is Ops.CONST: return Tensor(int(x.arg), dtype=dtypes.int32).reshape(1)
  if x.is_bound_var: return Tensor(x).cast(dtypes.int32).reshape(1)
  if x.op in (Ops.ADD, Ops.MUL) and len(x.src) == 2 and (a:=_int_tensor(x.src[0])) is not None and (b:=_int_tensor(x.src[1])) is not None:
    return a + b if x.op is Ops.ADD else a * b
  return None

@dataclass(frozen=True)
class DecodeCfg:
  """Flash decoding's launch shape: NSPLIT blocks per (batch, KV head) of NWARPS warps, each warp issuing UNR steps of loads at once."""
  nb:int
  nbk:int  # the cache's rows (nb, or a batched cache's slots)
  nt:int  # query tokens per sequence
  nh:int
  nkv:int
  nctx:int
  hd:int
  nsplit:int
  nwarps:int = 4
  unr:int = 2
  @property
  def minkeys(self) -> int: return self.nwarps * (32 // (self.hd // 8)) * self.unr  # keys per block step: the fewest a split gets
  @property
  def name(self) -> str:
    return f"{self.nb}{f'of{self.nbk}' if self.nbk != self.nb else ''}{f'x{self.nt}' if self.nt > 1 else ''}_{self.nh}_{self.nkv}_" \
           f"{self.nctx}_{self.hd}_s{self.nsplit}w{self.nwarps}u{self.unr}"

def _decode_program(OUT:UOp, *srcs:UOp, stage:str, cfg:DecodeCfg) -> UOp:
  name = f"nv_flash_decode_{stage.lower()}_{cfg.name}"
  defines = {stage: 1, "KNAME": name, "NB": cfg.nb, "NBK": cfg.nbk, "NT": cfg.nt, "NH": cfg.nh, "NKV": cfg.nkv, "NCTX": cfg.nctx, "HD": cfg.hd,
             "NSPLIT": cfg.nsplit, "NWARPS": cfg.nwarps, "UNR": cfg.unr, "MINKEYS": cfg.minkeys}
  src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + DECODE_SRC
  dims:tuple[UOp, ...] = (UOp.special(32 * cfg.nwarps, "lidx0"), UOp.special(cfg.nsplit, "gidx0"), UOp.special(cfg.nb * cfg.nt * cfg.nkv, "gidx1")) \
    if stage == "PARTIAL" else (UOp.special(cfg.hd, "lidx0"), UOp.special(cfg.nb * cfg.nt * cfg.nh, "gidx0"))
  # the KV bytes read depend on the runtime length: count the query side only
  n = cfg.nb * cfg.nt * cfg.nh * cfg.hd
  est = Estimates(ops=4 * n, lds=4 * n, mem=4 * n)
  sink = UOp.sink(OUT.base, *[x.base for x in srcs], *dims, arg=KernelInfo(name, estimates=est))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

NV_DECODE_MAX_TOKENS = 8  # flash decoding for up to this many query tokens (speculative decoding's check); more go to the flash kernel

def nv_flash_decode(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp|tuple[int|UOp, ...], nsms:int,
                    rows:tuple[int|UOp, ...]|None=None) -> Tensor|None:
  """A few query tokens per sequence: q (B, H, T, D), token t of sequence b at position valid_end[b] - T + t (one valid_end: all of
  them), over the fp16 cache's keys up to it, sequence b in cache row rows[b] (default b). (B, T, H * D) fp32, or None."""
  B, H, T, D = (int(x) for x in q.shape)
  NBK, KVH, NCTX = int(assigned_kv.shape[1]), int(assigned_kv.shape[2]), int(assigned_kv.shape[3])
  ends = valid_end if isinstance(valid_end, tuple) else (valid_end,) * B
  end_ts, row_ts = [_int_tensor(e) for e in ends], [_int_tensor(r) for r in (rows or tuple(range(B)))]
  if len(ends) != B or len(row_ts) != B or any(t is None for t in end_ts + row_ts): return None
  # enough blocks to fill the GPU a few times over: each streams its share of the keys
  cfg = DecodeCfg(B, NBK, T, H, KVH, NCTX, D, nsplit=max(1, min(64, -(-getenv("NV_DECODE_BPS", 4) * nsms // (B * T * KVH)))),
                  nwarps=getenv("NV_DECODE_WARPS", 4), unr=getenv("NV_DECODE_UNR", 2))
  params = Tensor.cat(*cast(list[Tensor], end_ts + row_ts)).contiguous()
  part = Tensor.empty(B * T * H * cfg.nsplit * (D + 2), dtype=dtypes.float32, device=q.device)
  part = Tensor.custom_kernel(part, q.transpose(1, 2).cast(dtypes.half).contiguous(), assigned_kv, params,
                              fxn=functools.partial(_decode_program, stage="PARTIAL", cfg=cfg))[0]
  out = Tensor.invalids(B, T, H, D, dtype=dtypes.float32, device=q.device)
  out = Tensor.custom_kernel(out, part, params, fxn=functools.partial(_decode_program, stage="COMBINE", cfg=cfg))[0].reshape(B, T, H * D)
  _note_decode(out, part, params, cfg)
  return out

def nv_flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp|tuple[int|UOp, ...], rows:tuple[int|UOp, ...]|None=None) -> Tensor|None:
  """Causal attention of q (B, H, T, D) over the fp16 KV cache assigned_kv (2, Bk, KvH, max_context, D), sequence b in cache row
  rows[b] (default b; Bk > B: B sequences in a batched cache's slots), whose keys [0, valid_end) are valid; query t sits at position
  valid_end - T + t. valid_end can be one per sequence (batched decode, flash decoding only). Returns (B, T, H * D) fp32 (the heads
  side by side, as the output projection takes them), or None if the kernel doesn't apply (unsupported device or shapes). One
  token (decode) uses flash decoding. A symbolic T is padded to its max, and the padding is sliced off."""
  if not nv_gemm_supported(q.device) or assigned_kv.dtype != dtypes.half or q.ndim != 4 or assigned_kv.ndim != 5: return None
  if q.uop.axis is not None or assigned_kv.uop.axis is not None: return None
  B, H, T, D = q.shape
  _, Bk, KVH, NCTX, Dk = assigned_kv.shape
  if not all(isinstance(v, int) for v in (B, H, D, Bk, KVH, NCTX, Dk)): return None
  B, H, D, KVH, NCTX = int(B), int(H), int(D), int(KVH), int(NCTX)
  if D != Dk or H % KVH or D not in (64, 128) or (rows is None and B != Bk): return None
  if rows is not None and (len(rows) != B or any((r if isinstance(r, int) else int(r.vmax)) >= int(Bk) for r in rows)): return None
  if isinstance(T, int) and T <= NV_DECODE_MAX_TOKENS: return nv_flash_decode(q, assigned_kv, valid_end, num_sms(q.device), rows)
  # the prefill kernel: one valid_end, sequences in consecutive cache rows from row0
  if isinstance(valid_end, tuple) or (rows is not None and B != 1): return None
  row0 = 0 if rows is None else rows[0]
  T_max = q.max_shape[2]
  if not isinstance(T_max, int) or T_max < 16: return None  # a few tokens (e.g. speculative decode) stay on tinygrad's kernels
  cfg = FlashCfg(WARPS=4 if T_max >= 64 else 2)
  bm = 16 * cfg.WARPS
  if NCTX % cfg.BN or cfg.smem(D) > MAX_SMEM: return None
  TQ = -(-T_max // bm) * bm
  qp = q if isinstance(T, int) and T == TQ else q.pad_to((B, H, T_max, D)).pad((None, None, (0, TQ - T_max), None))
  if (end_t:=_int_tensor(valid_end)) is None or (start_t:=_int_tensor(valid_end - T)) is None or (row_t:=_int_tensor(row0)) is None: return None
  params = end_t.cat(start_t, row_t).contiguous()
  out = Tensor.invalids(B, TQ, H, D, dtype=dtypes.float32, device=q.device)
  out = Tensor.custom_kernel(out, qp.cast(dtypes.half).contiguous(), assigned_kv, params,
                             fxn=functools.partial(_flash_program, cfg=cfg))[0]
  return (out if isinstance(T, int) and T == TQ else out.shrink((None, (0, T), None, None))).reshape(B, T, H * D)

# ******** GEMV on GGUF quantized weights (decode) ********

Q8_0, Q4_K, Q5_K, Q6_K = 8, 12, 13, 14  # ggml types
# K must be a multiple of this: the quant block, and for Q6_K/Q8_0 enough of them for 8-byte aligned rows (the kernel's ld16)
NV_QUANT_UNIT = {Q8_0: 256, Q4_K: 256, Q5_K: 256, Q6_K: 1024}
NV_QUANT_BYTES = {Q8_0: 34 / 32, Q4_K: 144 / 256, Q5_K: 176 / 256, Q6_K: 210 / 256}  # bytes per weight
NV_QUANT_STEP = {Q8_0: 512, Q4_K: 1024, Q5_K: 1024, Q6_K: 1024}  # weights per warp per loop iteration
NV_QUANT_ALIGN = {Q8_0: 8, Q4_K: 16, Q5_K: 16, Q6_K: 8}  # byte alignment of the weights the kernel's loads need (Q4_K: 128-bit loads)
NV_QUANT_MAX_TOKENS = 8  # each warp keeps one accumulator per token; prefill dequantizes to fp16 for the GEMM instead

QuantSeg = tuple[Tensor, int, int]  # one weight of a GEMV or dequant launch: (packed blocks, ggml type, rows)

def _byte_offset(t:Tensor) -> int|None:
  # offset of t's data in its buffer, in bytes (a GGUF tensor is a view into the whole file's buffer)
  return None if (view:=t.uop.contiguous_view()) is None else view[1] * view[0].dtype.itemsize

def _seg_ok(seg:QuantSeg, k:int, device:str|tuple[str, ...]|None) -> bool:
  packed, qtype, n = seg
  return qtype in NV_QUANT_UNIT and k % NV_QUANT_UNIT[qtype] == 0 and packed.device == device and packed.uop.axis is None and \
    packed.nbytes() == int(n * k * NV_QUANT_BYTES[qtype])

def _seg_source(segs:tuple[tuple[int, int], ...], stage:str, ksplit:int, interleave:bool=False, rows_per_warp:int=1,
                resid:bool=False) -> dict[str, str]:
  # SEG_PARAMS and SEG_DISPATCH for nv_quant.cu: the weight pointers, and which weight a GEMV block / DEQUANT thread works on.
  # interleave (DEQUANT, two weights of the same rows): output row 2r is row r of the first, 2r + 1 row r of the second
  params = ", ".join(f"const unsigned char *__restrict__ W{i}" for i in range(len(segs)))
  fn, args = ("requant32", "blk, q, sc") if stage == "REQUANT" else ("dequant8", "col, v")
  if interleave:
    (q0, _), (q1, _) = segs
    return {"SEG_PARAMS": params, "SEG_DISPATCH": f"if (!(row & 1)) {fn}<{q0}>(W0 + (size_t)(row >> 1) * QRow<{q0}>::BYTES, {args}); "
                                                  f"else {fn}<{q1}>(W1 + (size_t)(row >> 1) * QRow<{q1}>::BYTES, {args});"}
  calls, start, col = [], 0, 0
  for i, (qtype, n) in enumerate(segs):
    if stage == "GEMV":
      end = start + -(-n // (4 // ksplit * rows_per_warp))
      calls.append(f"if (blk < {end}) gemv_rows<{qtype}, {n}>(OUT + {col}, W{i}, XQ, blk - {start}, part{', RES' if resid else ''});")
    else:
      end = start + n
      call = f"{fn}<{qtype}>(W{i} + (size_t)(row - {start}) * QRow<{qtype}>::BYTES, {args});"
      calls.append(call if i == len(segs) - 1 else f"if (row < {end}) {call}")
    start, col = end, col + n
  return {"SEG_PARAMS": params, "SEG_DISPATCH": " else ".join(calls)}

_CTYPE = {dtypes.float32: "float", dtypes.half: "half"}
# an RMSNorm folded into the activation quantization: (eps, input dtype, weight dtype as stored, weight dtype the model uses)
Norm = tuple[float, DType, DType, DType]

def _quant_program(OUT:UOp, *srcs:UOp, stage:str, k:int, ntok:int, segs:tuple[tuple[int, int], ...]=(), ksplit:int=1,
                   glu:Glu|None=None, norm:Norm|None=None, attn:DecodeCfg|None=None, interleave:bool=False, xtype:DType|None=None,
                   rows_per_warp:int=1, resid:bool=False) -> UOp:
  qts, ns, ntot = "x".join(str(q) for q, _ in segs), "x".join(str(n) for _, n in segs), sum(n for _, n in segs)
  wbytes = int(sum(n * k * NV_QUANT_BYTES[q] for q, n in segs))
  if stage in ("DEQUANT", "REQUANT"):
    name = f"nv_q{qts}_{stage.lower()}_{ns}_{k}" + ("_il" if interleave else "")
    defines = {stage: 1, "N_OUT": ntot, **_seg_source(segs, stage, ksplit, interleave)}
    dims = (UOp.special(128, "lidx0"), UOp.special(ntot * k // (1024 if stage == "DEQUANT" else 4096), "gidx0"))
    io = wbytes + ntot * k * (2 if stage == "DEQUANT" else 1.25)
    est = Estimates(ops=3 * ntot * k, lds=int(io), mem=int(io))
  elif stage == "QUANTIZE":
    name, defines = f"nv_q8_quantize_{ntok}_{k}", {"QUANTIZE": 1}
    if glu is not None:
      suffix, gdefs = _glu_defines(glu)
      name, defines = name + suffix, {**defines, **gdefs}
    if norm is not None:
      eps, xdt, wdt, wuse = norm
      name += "_norm" + hashlib.md5(repr(norm).encode()).hexdigest()[:6]
      defines.update({"NORM": 1, "NORM_EPS": f"{eps!r}f", "XTYPE": _CTYPE[xdt], "WTYPE": _CTYPE[wdt]})
      if wuse != wdt and wuse == dtypes.half: defines["WLOAD(w)"] = "__half2float(__float2half_rn((float)(w)))"
    if xtype is not None: name, defines = name + "_" + _CTYPE[xtype], {**defines, "XTYPE": _CTYPE[xtype]}
    if attn is not None:
      name += f"_attn{attn.hd}s{attn.nsplit}m{attn.minkeys}" + (f"t{attn.nt}" if attn.nt > 1 else "")
      defines.update({"ATTN": 1, "HD": attn.hd, "NSPLIT": attn.nsplit, "MINKEYS": attn.minkeys, "NT": attn.nt})
    dims = (UOp.special(128, "lidx0"), UOp.special(ntok * k // 128, "gidx0"))
    io = ntok * k * (4 * (2 if glu else attn.nsplit if attn else 1) + 1) + (k * 4 if norm else 0)
    est = Estimates(ops=(10 if glu else 6 if norm else 3) * ntok * k, lds=io, mem=io)
  else:
    name = f"nv_q{qts}_gemv_{ntok}_{ns}_{k}_ks{ksplit}" + (f"r{rows_per_warp}" if rows_per_warp > 1 else "") + ("_res" if resid else "")
    defines = {"GEMV": 1, "N_OUT": ntot, "NTOK": ntok, "KSPLIT": ksplit, "GEMV_R": rows_per_warp,
               **_seg_source(segs, stage, ksplit, rows_per_warp=rows_per_warp, resid=resid), **({"RESID": 1} if resid else {})}
    dims = (UOp.special(128, "lidx0"), UOp.special(sum(-(-n // (4 // ksplit * rows_per_warp)) for _, n in segs), "gidx0"))
    io = wbytes + ntok * (k + 4 * ntot)
    est = Estimates(ops=2 * ntok * ntot * k, lds=io, mem=io)
  src = "".join(f"#define {kk} {v}\n" for kk, v in {**defines, "KNAME": name, "K_DIM": k}.items()) + QUANT_SRC
  sink = UOp.sink(OUT.base, *[x.base for x in srcs], *dims, arg=KernelInfo(name, estimates=est))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

def nv_quant_gemv(x:Tensor, segs:Sequence[QuantSeg], in_features:int, glu:Glu|None=None) -> Tensor|None:
  """x @ concat(W_0, W_1, ...).T in fp32 for a few tokens, each W_i (rows_i, in_features) stored as packed GGUF blocks (Q4_K, Q6_K or
  Q8_0, as find_packed returns them): one launch for weights that share their input. With glu, x is (..., 2 * in_features), the
  output of a fused gate/up nv_quant_gemv, and the input is its gated activation (model.gated_activation's SwiGLU with glu's
  (alpha, limit, up_bias)) instead. None if the kernel doesn't apply."""
  K = in_features
  if not segs or not nv_gemm_supported(x.device) or x.uop.axis is not None or x.shape[-1] != (2 * K if glu else K): return None
  if glu is not None and x.dtype != dtypes.float32: return None
  tokens = prod(x.shape[:-1])
  if not isinstance(tokens, int) or not 1 <= tokens <= NV_QUANT_MAX_TOKENS or K % 128: return None
  for seg in segs:
    if not _seg_ok(seg, K, x.device) or (off:=_byte_offset(seg[0])) is None or off % NV_QUANT_ALIGN[seg[1]]: return None
  # split each row's K over up to 4 warps when there are too few rows to keep every SM streaming (~64 warps per SM: a long row
  # per warp leaves the SMs idle at the end, bench/quant_bench.py)
  # with several tokens a warp handles R rows, loading each token's activations once for all of them
  R = 1 if tokens == 1 else getenv("NV_GEMV_R", 2)
  N, step, ksplit = sum(n for _, _, n in segs), max(NV_QUANT_STEP[q] for _, q, _ in segs), 1
  while ksplit < 4 and N * ksplit < getenv("NV_QUANT_WARPS_PER_SM", 64) * num_sms(x.device) * R and K // step >= 2 * ksplit: ksplit *= 2
  fxn = functools.partial(_quant_program, stage="GEMV", k=K, ntok=tokens, segs=tuple((q, n) for _, q, n in segs), ksplit=ksplit,
                          rows_per_warp=R)
  srcs = (*(p for p, _, _ in segs), _quantize_x(x, tokens, K, glu))
  out = Tensor.custom_kernel(Tensor.empty(tokens, N, dtype=dtypes.float32, device=x.device), *srcs, fxn=fxn)[0].reshape(*x.shape[:-1], N)
  if len(segs) == 1:  # residual_add can rebuild it with the residual added in the kernel
    if len(_GEMV_SRC) >= 8: _GEMV_SRC.pop(next(iter(_GEMV_SRC)))
    _GEMV_SRC[out.uop] = (srcs, functools.partial(fxn, resid=True), tokens, N)
  return out

_GEMV_SRC:dict[UOp, tuple[tuple[Tensor, ...], Callable, int, int]] = {}
def nv_residual_add(x:Tensor, y:Tensor) -> Tensor|None:
  """x + y in one kernel when y is a quant GEMV's output (one weight, decode): the GEMV adds x as it stores, so the add's own kernel
  doesn't run (y isn't used). None otherwise."""
  if getenv("DISABLE_NV_RESID_FUSE") or (src:=_GEMV_SRC.get(y.uop)) is None: return None
  srcs, fxn, tokens, N = src
  if x.shape != y.shape or x.dtype != dtypes.float32 or x.device != y.device or x.uop.axis is not None: return None
  # an invalids output already in x's shape: as a @function's (the LLM block's) return, a Tensor.empty buffer was copied out
  return Tensor.custom_kernel(Tensor.invalids(*x.shape, dtype=dtypes.float32, device=x.device), *srcs, x.contiguous(), fxn=fxn)[0]

def nv_quant_linear(x:Tensor, packed:Tensor, qtype:int, out_features:int, in_features:int) -> Tensor|None:
  """x @ W.T for a few tokens, W (out_features, in_features) stored as packed GGUF blocks of type qtype (Q4_K, Q6_K or Q8_0,
  as find_packed returns them). The result is in x's dtype (accumulated in fp32), or None if the kernel doesn't apply."""
  return None if (out:=nv_quant_gemv(x, [(packed, qtype, out_features)], in_features)) is None else out.cast(x.dtype)

def nv_dequant(segs:Sequence[QuantSeg], in_features:int, dep:Tensor, interleave:bool=False) -> Tensor|None:
  """The (sum of rows, in_features) fp16 weight of the packed GGUF blocks (Q4_K, Q6_K or Q8_0) of segs, one after another, or None
  if unsupported. The kernel waits for dep (the layer's input, otherwise unused) so each layer's copy is made right before its GEMM.
  interleave (two segs of the same rows, e.g. gate and up): their rows alternate instead, for nv_linear's glu."""
  N, K = sum(n for _, _, n in segs), in_features
  if not segs or (N * K) % 1024 or not all(_seg_ok(seg, K, dep.device) for seg in segs): return None
  if interleave and (len(segs) != 2 or segs[0][2] != segs[1][2]): return None
  fxn = functools.partial(_quant_program, stage="DEQUANT", k=K, ntok=0, segs=tuple((q, n) for _, q, n in segs), interleave=interleave)
  return Tensor.custom_kernel(Tensor.empty(N, K, dtype=dtypes.half, device=dep.device), *(p for p, _, _ in segs), dep, fxn=fxn)[0]

# ******** int8 tensor core GEMM for prefill on GGUF quant weights (nv_mmq.cu) ********

# a B segment of the int8 GEMM: (format, first column, last column, number of pointer arguments); formats as nv_mmq.cu's F_*
MmqSeg = tuple[int, int, int, int]
F_I8, F_I8Q6, F_Q4K, F_Q8, F_IL = 0, 1, 2, 3, 4

@dataclass(frozen=True)
class MmqCfg:
  BM: int = 128
  BN: int = 128
  WM: int = 64
  WN: int = 32
  STAGES: int = 2
  NBLK: int = 4    # quant blocks (32 of K) per stage
  GROUP_M: int = 8
  SK: bool = True
  @property
  def BK(self) -> int: return 32 * self.NBLK  # in int8 (= K elements)
  @property
  def threads(self) -> int: return (self.BM // self.WM) * (self.BN // self.WN) * 32
  def brow(self, fmts:Sequence[int]) -> int:  # B stage row stride (nv_mmq.cu BROW): the largest the launch's formats need
    return max({F_Q4K: 112, F_Q8: 144}.get(f & 3, self.BK + self.NBLK * 8) for f in fmts)
  def smem(self, fmts:Sequence[int]=(F_I8,)) -> int:  # nv_mmq.cu: A tiles + scales, B rows, decoded B scales (packed formats)
    raw = any((f & 3) >= F_Q4K for f in fmts)
    return self.STAGES * (self.BM * (self.BK + self.NBLK * 8) + self.BN * self.brow(fmts)) + raw * self.BN * self.NBLK * 8
  @property
  def blocks_per_sm(self) -> int: return max(1, min(48 * 32 // self.threads, (100 * 1024) // (self.smem() + 1024)))
  def __str__(self): return f"{self.BM}x{self.BN}_w{self.WM}x{self.WN}_s{self.STAGES}b{self.NBLK}" + ("_sk" if self.SK else "")
MMQ = MmqCfg()
# smaller row tiles for short prompts and generate()'s 32-token prefill chunks (the weights are read once either way)
MMQ_SMALL = {16: MmqCfg(BM=16, WM=16, WN=32), 32: MmqCfg(BM=32, WM=32, WN=32), 64: MmqCfg(BM=64, WM=32, WN=32)}
def mmq_cfg(M:int) -> MmqCfg:
  # the 128-row tile when it pads M by at most 1/8, else the smaller tile that pads least (the larger one on ties)
  if M >= MMQ.BM and (-M % MMQ.BM) * 8 <= M: return MMQ
  return min((MMQ_SMALL[b] for b in (64, 32, 16)), key=lambda c: -M % c.BM)
Q6_K_TYPE = 14


def _mmq_program(C:UOp, X:UOp, *srcs:UOp, cfg:MmqCfg, sp:Split, k:int, n:int, msegs:tuple[MmqSeg, ...], glu:Glu|None) -> UOp:
  M, (SKWS, FLAGS), wsrcs = int(C.shape[0]), srcs[-2:], srcs[:-2]
  code = "x".join({F_I8: "i", F_I8Q6: "i6", F_Q4K: "r12", F_Q8: "r8"}[f & 3] + ("il" if f & F_IL else "") for f, _, _, _ in msegs)
  name = f"nv_mmq_{M}_{n}_{k}_{cfg}{sp.sk_p or ''}_{code}"
  # pick the tile's segment pointers first, then one mainloop per distinct format (each instantiation is a full copy of the loop)
  params, sel, pi = [], [], 0
  for i, (f, a, b, nptr) in enumerate(msegs):
    names = [f"W{pi + j}" for j in range(nptr)]
    params += [f"const {'float2' if (f & 3) <= F_I8Q6 and j == 1 else 'unsigned char'} *__restrict__ {nm}" for j, nm in enumerate(names)]
    asg = f"{{ wa = {names[0]}; wb = {names[1] if nptr > 1 else 'nullptr'}; nb = bn - {a}; f = {f}; }}"
    sel.append(asg if i == len(msegs) - 1 else f"if (bn < {b}) {asg}")
    pi += nptr
  fmts = sorted(set(f for f, _, _, _ in msegs))
  raw = any((f & 3) >= F_Q4K for f in fmts)
  run = " else ".join(f"if (f == {f}) mainloop<{f}>(t, wa, wb, nb, k0, k1);" for f in fmts[:-1]) + \
        (" else " if len(fmts) > 1 else "") + f"mainloop<{fmts[-1]}>(t, wa, wb, nb, k0, k1);"
  seg_run = "{ const unsigned char *wa; const void *wb; int nb, f; " + " else ".join(sel) + " " + run + " }"
  defines:dict[str, Any] = {"M_DIM": M, "N_DIM": n, "K_DIM": k, "BM": cfg.BM, "BN": cfg.BN, "WM": cfg.WM, "WN": cfg.WN,
                            "STAGES": cfg.STAGES, "NBLK": cfg.NBLK, "GROUP_M": cfg.GROUP_M, "SK_P": sp.sk_p, "SEG_PARAMS": ", ".join(params),
                            "SEG_RUN(k0, k1)": seg_run, "BROW": cfg.brow(fmts), "HAS_RAW": int(raw)}
  if glu is not None:
    suffix, gdefs = _glu_defines(glu)
    name, defines = name + suffix, {**defines, **gdefs}
  src = "".join(f"#define {kk} {v}\n" for kk, v in {**defines, "KNAME": name}.items()) + MMQ_SRC
  threads, blocks = UOp.special(cfg.threads, "lidx0"), UOp.special(sp.grid, "gidx0")
  io = M * k * 5 // 4 + n * k * 5 // 8 + int(C.shape[1]) * M * C.dtype.itemsize
  est = Estimates(ops=2 * M * n * k, lds=io, mem=io)
  sink = UOp.sink(C.base, X.base, *[w.base for w in wsrcs], SKWS.base, FLAGS.base, threads, blocks,
                  arg=KernelInfo(name, estimates=est, dyn_shared_mem=cfg.smem(fmts)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

def nv_requant(segs:Sequence[QuantSeg], in_features:int, dep:Tensor, interleave:bool=False) -> tuple[Tensor, Tensor]|None:
  """(int8 values (rows, in_features), float2 scales (rows, in_features/32, 2)) of the packed GGUF blocks of segs for nv_mmq, the rows
  one after another (or alternating with interleave). Waits for dep like nv_dequant."""
  N, K = sum(n for _, _, n in segs), in_features
  if not segs or (N * K) % 4096 or not all(_seg_ok(seg, K, dep.device) for seg in segs): return None
  if interleave and (len(segs) != 2 or segs[0][2] != segs[1][2]): return None
  if any((off:=_byte_offset(p)) is None or off % NV_QUANT_ALIGN[q] for p, q, _ in segs): return None  # (Q4_K: 128-bit loads)
  fxn = functools.partial(_quant_program, stage="REQUANT", k=K, ntok=0, segs=tuple((q, n) for _, q, n in segs), interleave=interleave)
  w8, ws = Tensor.empty(N, K, dtype=dtypes.int8, device=dep.device), Tensor.empty(N, K // 32, 2, dtype=dtypes.float32, device=dep.device)
  w8, ws = Tensor.custom_kernel(w8, ws, *(p for p, _, _ in segs), dep, fxn=fxn)[:2]
  return w8, ws

def nv_mmq(x:Tensor, segs:Sequence[QuantSeg], in_features:int, glu:Glu|None=None, interleave:bool=False) -> Tensor|None:
  """x @ concat(W_i).T for prefill on GGUF Q4_K/Q6_K/Q8_0 weights with the int8 tensor core GEMM: x is quantized to int8 per 32 values
  (like decode), the weights are requantized to int8 + per-block scales (exact), fp32 result (..., sum of rows). With glu (and
  interleave: two segs, gate and up) the result (..., rows/2) is fp16 SwiGLU of their outputs. None if it doesn't apply."""
  K, N = in_features, sum(n for _, _, n in segs)
  if getenv("DISABLE_NV_MMQ") or not segs or not nv_gemm_supported(x.device) or x.uop.axis is not None or x.shape[-1] != K: return None
  if x.dtype not in _CTYPE or K % MMQ.BK or N % MMQ.BN: return None
  xs = x if isinstance(x.numel(), int) else x.pad_to(x.max_shape)
  M = prod(xs.shape[:-1])
  if not isinstance(M, int) or M <= NV_QUANT_MAX_TOKENS: return None  # (a few tokens: the GEMV)
  cfg = mmq_cfg(M)
  Mp = -(-M // cfg.BM) * cfg.BM
  # each 128-column tile reads one segment, in one format: the packed Q4_K / Q8_0 blocks as they are (16 / 8-byte aligned), else
  # (Q6_K, misaligned) the requantized int8 copy; so segment boundaries have to fall on tiles
  starts = list(itertools.accumulate((n for _, _, n in segs), initial=0))
  if any(a % cfg.BN for a in starts) or not all(_seg_ok(seg, K, x.device) for seg in segs): return None
  def raw_fmt(seg:QuantSeg) -> int|None:
    if getenv("NV_MMQ_REQUANT") or (off:=_byte_offset(seg[0])) is None: return None
    return F_Q4K if seg[1] == 12 and off % 16 == 0 else F_Q8 if seg[1] == 8 and off % 8 == 0 else None
  x2 = xs.reshape(M, K)
  if Mp != M: x2 = x2.pad(((0, Mp - M), None))
  xq = _quantize_x(x2, Mp, K, None)
  msegs:list[MmqSeg] = []
  wts:list[Tensor] = []
  if interleave:
    if len(segs) != 2 or segs[0][1] != segs[1][1]: return None
    if (f:=raw_fmt(segs[0])) is not None and raw_fmt(segs[1]) == f: msegs, wts = [(f | F_IL, 0, N, 2)], [segs[0][0], segs[1][0]]
    else:
      if (wq:=nv_requant(segs, K, xq, interleave=True)) is None: return None
      msegs, wts = [(F_I8Q6 if segs[0][1] == Q6_K_TYPE else F_I8, 0, N, 2)], list(wq)
  else:
    # consecutive requantized segments of one kind share a requant launch and a segment
    groups:list[tuple[int, int, list[QuantSeg]]] = []
    for seg, a in zip(segs, starts):
      f = raw_fmt(seg)
      if f is None: f = F_I8Q6 if seg[1] == Q6_K_TYPE else F_I8
      if groups and f <= F_I8Q6 and groups[-1][0] == f: groups[-1][2].append(seg)
      else: groups.append((f, a, [seg]))
    for f, a, gsegs in groups:
      if f >= F_Q4K: wl = [gsegs[0][0]]
      elif (wq:=nv_requant(gsegs, K, xq)) is None: return None
      else: wl = list(wq)
      msegs.append((f, a, a + sum(g[2] for g in gsegs), len(wl)))
      wts += wl
  sp = split(Mp, N, K, cfg, num_sms(x.device))
  skws = Tensor.empty(sp.sk_workers * cfg.BM * cfg.BN if sp.sk_workers else 1, dtype=dtypes.float32, device=x.device)
  flags = _flags(x.device) if sp.sk_workers else Tensor.empty(1, dtype=dtypes.int32, device=x.device)
  NO = N // 2 if glu is not None else N
  out = Tensor.invalids(Mp, NO, dtype=dtypes.half if glu is not None else dtypes.float32, device=x.device)
  fxn = functools.partial(_mmq_program, cfg=cfg, sp=sp, k=K, n=N, msegs=tuple(msegs), glu=glu)
  out = Tensor.custom_kernel(out, xq, *wts, skws, flags, fxn=fxn)[0]
  if Mp != M: out = out[:M]
  out = out.reshape(*xs.shape[:-1], NO)
  return out if xs is x else out.shrink(tuple((0, s) for s in (*x.shape[:-1], NO)))

_NORM_SRC:dict[UOp, tuple[Tensor, Tensor, float]] = {}
def nv_note_rmsnorm(out:Tensor, x:Tensor, weight:Tensor, eps:float):
  """Record that out is RMSNorm(x) (x * rsqrt(mean(x^2) + eps) * weight over the last axis): a quant GEMV of out then quantizes
  from x and applies the norm in the same kernel, so the norm's own kernels never run (out isn't used). Keeps a few entries."""
  if len(_NORM_SRC) >= 8: _NORM_SRC.pop(next(iter(_NORM_SRC)))
  _NORM_SRC[out.uop] = (x, weight, eps)

def _norm_src(x:Tensor, K:int) -> tuple[Tensor, Tensor, Tensor, float]|None:
  # (norm input, norm weight, the weight as stored, eps) when x is a recorded RMSNorm the quantize kernel can apply
  if (src:=_NORM_SRC.get(x.uop)) is None: return None
  nx, w, eps = src
  if nx.shape != x.shape or w.shape != (K,) or nx.dtype not in _CTYPE or w.dtype not in _CTYPE: return None
  if nx.uop.axis is not None or w.uop.axis is not None or not nx.device == w.device == x.device: return None
  # a GGUF norm weight is fp32 cast to the model's dtype: read the fp32 buffer and round in the kernel (no cast kernel per call)
  stored = Tensor(w.uop.src[0]) if w.uop.op is Ops.CAST and w.uop.src[0].dtype in _CTYPE else w
  return nx, w, stored, eps

_ATTN_SRC:dict[UOp, tuple[Tensor, Tensor, DecodeCfg]] = {}
def _note_decode(out:Tensor, part:Tensor, params:Tensor, cfg:DecodeCfg):
  # out is the merge of flash decoding's partials part: a quant GEMV of out (the output projection) merges them in its quantize
  # kernel instead, so the merge kernel doesn't run
  if len(_ATTN_SRC) >= 8: _ATTN_SRC.pop(next(iter(_ATTN_SRC)))
  _ATTN_SRC[out.uop] = (part, params, cfg)

_XQ_CACHE:dict[tuple[UOp, Glu|None], Tensor] = {}
def _quantize_x(x:Tensor, tokens:int, K:int, glu:Glu|None) -> Tensor:
  # int8 activations for the GEMV. Linears that take the same x one after another (when not fused) quantize it once.
  # (A fresh Tensor.empty output per call would make each quantization a distinct kernel.) Keyed on the UOp, a few entries.
  if (xq:=_XQ_CACHE.get(key:=(x.uop, glu))) is not None: return xq
  xq = Tensor.empty(tokens * (K // 4 + K // 16), dtype=dtypes.int32, device=x.device)
  fxn = functools.partial(_quant_program, stage="QUANTIZE", k=K, ntok=tokens, glu=glu)
  if glu is None and not getenv("DISABLE_NV_NORM_FUSE") and (src:=_norm_src(x, K)) is not None:
    nx, w, stored, eps = src
    xq = Tensor.custom_kernel(xq, nx.reshape(tokens, K).contiguous(), stored.contiguous(),
                              fxn=functools.partial(fxn, norm=(eps, nx.dtype, stored.dtype, w.dtype)))[0]
  elif glu is None and not getenv("DISABLE_NV_ATTN_FUSE") and (asrc:=_ATTN_SRC.get(x.uop)) is not None and \
       asrc[2].nh * asrc[2].hd == K and asrc[2].nb * asrc[2].nt == tokens:
    part, params, cfg = asrc
    xq = Tensor.custom_kernel(xq, part, params, fxn=functools.partial(fxn, attn=cfg))[0]
  elif x.dtype == dtypes.half and glu is None:  # read as is (e.g. the prefill SwiGLU's fp16 output): no cast kernel
    xq = Tensor.custom_kernel(xq, x.reshape(tokens, K).contiguous(), fxn=functools.partial(fxn, xtype=dtypes.half))[0]
  else: xq = Tensor.custom_kernel(xq, x.reshape(tokens, x.shape[-1]).float().contiguous(), fxn=fxn)[0]
  if len(_XQ_CACHE) >= 4: _XQ_CACHE.pop(next(iter(_XQ_CACHE)))
  _XQ_CACHE[key] = xq
  return xq

def ampere_matmul(a:Tensor, b:Tensor, cfg:Cfg|None=None) -> Tensor:
  """a @ b. Differentiable: the backward runs as two more custom GEMMs (dA = dC @ B^T, dB = A^T @ dC), each with its own
  auto config. Falls back to a.matmul(b) for unsupported inputs."""
  return _gemm(a, b, cfg=cfg)

if __name__ == "__main__":
  import numpy as np
  N = getenv("N", 4096)
  a, b = Tensor.randn(N, N, dtype=dtypes.half).realize(), Tensor.randn(N, N, dtype=dtypes.half).realize()
  c = ampere_matmul(a, b).realize()
  ref = a.float().numpy() @ b.float().numpy()
  err = np.abs(c.float().numpy() - ref).max()
  print(f"{Device.DEFAULT} N={N} cfg={auto_cfg(N, N, N)}  max abs err vs fp32 numpy: {err:.4f}")
  np.testing.assert_allclose(c.float().numpy(), ref, atol=0.5, rtol=1e-2)
  print("correct")
