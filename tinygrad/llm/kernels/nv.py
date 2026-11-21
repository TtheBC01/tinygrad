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
import ctypes, functools, pathlib
from typing import Any, Callable
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
DECODE_SRC = COMMON + (_DIR / "nv_decode.cu").read_text()
QUANT_SRC = COMMON + (_DIR / "nv_quant.cu").read_text()

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

def split(M:int, N:int, K:int, cfg:Cfg, sms:int) -> Split:
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
def _source(M:int, N:int, K:int, cfg:Cfg, sk_p:int, ta:bool, tb:bool, f32:bool, bf16:bool) -> tuple[str, str]:
  name = f"ampere_gemm_{'t' if ta else 'n'}{'t' if tb else 'n'}{'_bf16' if bf16 else ''}{'_f32' if f32 else ''}_{M}_{N}_{K}_{cfg}" + \
         (f"{sk_p}" if sk_p else "")
  defines = dict(KNAME=name, M_DIM=M, N_DIM=N, K_DIM=K, BM=cfg.BM, BN=cfg.BN, BK=cfg.BK, WM=cfg.WM, WN=cfg.WN, STAGES=cfg.STAGES,
                 GROUP_M=cfg.GROUP_M, SK_P=sk_p, TRANS_A=int(ta), TRANS_B=int(tb), OUT_F32=int(f32), BF16=int(bf16))
  return name, "".join(f"#define {k} {v}\n" for k, v in defines.items()) + SRC

def _program(C:UOp, A:UOp, B:UOp, WS:UOp, FLAGS:UOp, cfg:Cfg, sp:Split, ta:bool, tb:bool) -> UOp:
  (M, N), K = C.shape, (A.shape[0] if ta else A.shape[1])
  name, src = _source(M, N, K, cfg, sp.sk_p, ta, tb, C.dtype == dtypes.float32, A.dtype == dtypes.bfloat16)
  threads, blocks = UOp.special(cfg.threads, "lidx0"), UOp.special(sp.grid, "gidx0")
  io = 2 * (M * K + K * N) + C.dtype.itemsize * M * N
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

def _gemm(a:Tensor, b:Tensor, ta:bool=False, tb:bool=False, cfg:Cfg|None=None, out_dtype:DType|None=None) -> Tensor:
  """op(a) @ op(b) with the custom kernel (op transposes when ta/tb is set), or tinygrad's matmul if it's unsupported.
  a and b are both fp16 or both bf16. out_dtype is their dtype (default) or float32; the accumulator is always fp32."""
  out_dtype = out_dtype or a.dtype
  if cfg is None and a.ndim == 2 and b.ndim == 2 and all(isinstance(x, int) for x in _dims(a, b, ta, tb)):
    M, K, _, N = _dims(a, b, ta, tb)
    cfg = auto_cfg(M, N, K)
  if cfg is None or not supported(a, b, cfg, ta, tb): return (a.T if ta else a).matmul(b.T if tb else b).cast(out_dtype)
  a, b = a.contiguous(), b.contiguous()
  M, K, _, N = _dims(a, b, ta, tb)
  sp = split(M, N, K, cfg, num_sms(a.device) if cfg.SK else 0)
  assert sp.sk_workers <= MAX_SK_WORKERS
  assert out_dtype in (a.dtype, dtypes.float32), out_dtype
  out = Tensor.invalids(M, N, dtype=out_dtype, device=a.device)
  # stream-K scratch: fp32 partial tiles (one per worker), and the per-device flag buffer (one-element dummy when stream-K is off)
  ws = Tensor.empty(sp.sk_workers * cfg.BM * cfg.BN if sp.sk_workers else 1, dtype=dtypes.float32, device=a.device)
  flags = _flags(a.device) if sp.sk_workers else Tensor.empty(1, dtype=dtypes.int32, device=a.device)
  return Tensor.custom_kernel(out, a, b, ws, flags, fxn=functools.partial(_program, cfg=cfg, sp=sp, ta=ta, tb=tb),
                              grad_fxn=functools.partial(_grad, ta=ta, tb=tb))[0]

NV_LINEAR_MIN_ROWS = 16  # below this (decode), tinygrad's own GEMV is the right kernel

def nv_linear(x:Tensor, weight:Tensor, dequant:Callable[[Tensor], Tensor|None]|None=None) -> Tensor|None:
  """x @ weight.T for a Linear layer (weight: (out, in) fp16 or bf16), or None if the custom GEMM doesn't apply.
  x may be fp32 (it's cast to the weight dtype for the tensor cores); the result is fp32 then, accumulated in fp32 either way.
  A symbolic token count is padded to its max so the kernel sees static shapes, and the padding is sliced off.
  dequant(x2) makes the fp16 weight from quantized blocks instead (weight then only gives the shape), given the GEMM's input."""
  if weight.dtype not in (dtypes.half, dtypes.bfloat16) or weight.ndim != 2 or weight.uop.axis is not None: return None
  if x.dtype not in (weight.dtype, dtypes.float32): return None
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
  out = _gemm(x2, weight, tb=True, cfg=cfg, out_dtype=x.dtype)
  if Mp != M: out = out[:M]
  out = out.reshape(*xs.shape[:-1], N)
  return out if xs is x else out.shrink(tuple((0, s) for s in (*x.shape[:-1], N)))

# ******** causal flash attention on the KV cache (prefill) ********

@dataclass(frozen=True)
class FlashCfg:
  WARPS: int = 4   # 16 query rows per warp
  BN: int = 64     # keys per pipeline stage
  STAGES: int = 2
  def smem(self, hd:int) -> int: return max(2 * self.STAGES * self.BN, 16 * self.WARPS) * hd * 2

def _flash_program(O:UOp, Q:UOp, KV:UOp, P:UOp, cfg:FlashCfg) -> UOp:
  NB, TQ, NH, HD = (int(x) for x in O.shape)
  NKV, NCTX = int(KV.shape[2]), int(KV.shape[3])
  name = f"nv_flash_attn_{NB}_{NH}_{NKV}_{TQ}_{NCTX}_{HD}_w{cfg.WARPS}_n{cfg.BN}_s{cfg.STAGES}"
  defines = dict(KNAME=name, NB=NB, NH=NH, NKV=NKV, TQ=TQ, NCTX=NCTX, HD=HD, WARPS=cfg.WARPS, BN=cfg.BN, STAGES=cfg.STAGES)
  src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + FLASH_SRC
  threads, qblocks, heads = UOp.special(32 * cfg.WARPS, "lidx0"), UOp.special(TQ // (16 * cfg.WARPS), "gidx0"), UOp.special(NB * NH, "gidx1")
  # causal: about half of the TQ x (prefix + TQ) scores; the KV length is only known at runtime, so this counts the chunk itself
  io = (4 + 2) * NB * NH * TQ * HD  # fp32 out, fp16 queries (the cache reads depend on the runtime length)
  est = Estimates(ops=2 * 2 * NB * NH * TQ * TQ * HD // 2, lds=io, mem=io)
  sink = UOp.sink(O.base, Q.base, KV.base, P.base, threads, qblocks, heads, arg=KernelInfo(name, estimates=est, dyn_shared_mem=cfg.smem(HD)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

def _int_tensor(x:int|UOp) -> Tensor|None:
  # a symbolic int (sums/products of bound Variables and consts, e.g. start_pos + toks) as a 1-element int32 Tensor
  if isinstance(x, int): return Tensor([x], dtype=dtypes.int32)
  if x.op is Ops.CONST: return Tensor([int(x.arg)], dtype=dtypes.int32)
  if x.is_bound_var: return Tensor(x).cast(dtypes.int32).reshape(1)
  if x.op in (Ops.ADD, Ops.MUL) and len(x.src) == 2 and (a:=_int_tensor(x.src[0])) is not None and (b:=_int_tensor(x.src[1])) is not None:
    return a + b if x.op is Ops.ADD else a * b
  return None

def _decode_program(OUT:UOp, *srcs:UOp, stage:str, nb:int, nh:int, nkv:int, nctx:int, hd:int, nsplit:int) -> UOp:
  name = f"nv_flash_decode_{stage.lower()}_{nb}_{nh}_{nkv}_{nctx}_{hd}_s{nsplit}"
  defines = {stage: 1, "KNAME": name, "NB": nb, "NH": nh, "NKV": nkv, "NCTX": nctx, "HD": hd, "NSPLIT": nsplit}
  src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + DECODE_SRC
  dims:tuple[UOp, ...] = (UOp.special(128, "lidx0"), UOp.special(nsplit, "gidx0"), UOp.special(nb * nkv, "gidx1")) if stage == "PARTIAL" else \
    (UOp.special(hd, "lidx0"), UOp.special(nb * nh, "gidx0"))
  # the KV bytes read depend on the runtime length: count the query side only
  est = Estimates(ops=4 * nb * nh * hd, lds=4 * nb * nh * hd, mem=4 * nb * nh * hd)
  sink = UOp.sink(OUT.base, *[x.base for x in srcs], *dims, arg=KernelInfo(name, estimates=est))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

def nv_flash_decode(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp, nsms:int) -> Tensor|None:
  """One query token per sequence: q (B, H, 1, D) over the fp16 cache's keys [0, valid_end). (B, H, 1, D) fp32, or None."""
  B, H, _, D = (int(x) for x in q.shape)
  KVH, NCTX = int(assigned_kv.shape[2]), int(assigned_kv.shape[3])
  if (end_t:=_int_tensor(valid_end)) is None: return None
  # enough blocks to fill the GPU a few times over: each streams its share of the keys
  nsplit = max(1, min(64, -(-4 * nsms // (B * KVH))))
  def prg(stage:str): return functools.partial(_decode_program, stage=stage, nb=B, nh=H, nkv=KVH, nctx=NCTX, hd=D, nsplit=nsplit)
  part = Tensor.empty(B * H * nsplit * (D + 2), dtype=dtypes.float32, device=q.device)
  part = Tensor.custom_kernel(part, q.cast(dtypes.half).contiguous(), assigned_kv, end_t.contiguous(), fxn=prg("PARTIAL"))[0]
  out = Tensor.invalids(B, 1, H, D, dtype=dtypes.float32, device=q.device)
  return Tensor.custom_kernel(out, part, fxn=prg("COMBINE"))[0].transpose(1, 2)

def nv_flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp) -> Tensor|None:
  """Causal attention of q (B, H, T, D) over the fp16 KV cache assigned_kv (2, B, KvH, max_context, D), whose keys [0, valid_end)
  are valid; query t sits at position valid_end - T + t. Returns (B, H, T, D) fp32, or None if the kernel doesn't apply
  (unsupported device or shapes). One token (decode) uses flash decoding. A symbolic T is padded to its max, and the padding is
  sliced off."""
  if not nv_gemm_supported(q.device) or assigned_kv.dtype != dtypes.half or q.ndim != 4 or assigned_kv.ndim != 5: return None
  if q.uop.axis is not None or assigned_kv.uop.axis is not None: return None
  B, H, T, D = q.shape
  _, Bk, KVH, NCTX, Dk = assigned_kv.shape
  if not all(isinstance(v, int) for v in (B, H, D, Bk, KVH, NCTX, Dk)): return None
  B, H, D, KVH, NCTX = int(B), int(H), int(D), int(KVH), int(NCTX)
  if B != Bk or D != Dk or H % KVH or D not in (64, 128): return None
  if isinstance(T, int) and T == 1: return nv_flash_decode(q, assigned_kv, valid_end, num_sms(q.device))
  T_max = q.max_shape[2]
  if not isinstance(T_max, int) or T_max < 16: return None  # a few tokens (e.g. speculative decode) stay on tinygrad's kernels
  cfg = FlashCfg(WARPS=4 if T_max >= 64 else 2)
  bm = 16 * cfg.WARPS
  if NCTX % cfg.BN or cfg.smem(D) > MAX_SMEM: return None
  TQ = -(-T_max // bm) * bm
  qp = q if isinstance(T, int) and T == TQ else q.pad_to((B, H, T_max, D)).pad((None, None, (0, TQ - T_max), None))
  if (end_t:=_int_tensor(valid_end)) is None or (start_t:=_int_tensor(valid_end - T)) is None: return None
  params = end_t.cat(start_t).contiguous()
  out = Tensor.invalids(B, TQ, H, D, dtype=dtypes.float32, device=q.device)
  out = Tensor.custom_kernel(out, qp.cast(dtypes.half).contiguous(), assigned_kv, params,
                             fxn=functools.partial(_flash_program, cfg=cfg))[0].transpose(1, 2)
  return out if isinstance(T, int) and T == TQ else out.shrink((None, None, (0, T), None))

# ******** GEMV on GGUF quantized weights (decode) ********

Q8_0, Q4_K, Q6_K = 8, 12, 14  # ggml types
# K must be a multiple of this: the quant block, and for Q6_K/Q8_0 enough of them for 8-byte aligned rows (the kernel's ld16)
NV_QUANT_UNIT = {Q8_0: 256, Q4_K: 256, Q6_K: 1024}
NV_QUANT_BYTES = {Q8_0: 34 / 32, Q4_K: 144 / 256, Q6_K: 210 / 256}  # bytes per weight
NV_QUANT_STEP = {Q8_0: 512, Q4_K: 1024, Q6_K: 1024}  # weights per warp per loop iteration
NV_QUANT_ALIGN = {Q8_0: 8, Q4_K: 16, Q6_K: 8}  # byte alignment of the weights the kernel's loads need (Q4_K: 128-bit loads)
NV_QUANT_MAX_TOKENS = 8  # each warp keeps one accumulator per token; prefill dequantizes to fp16 for the GEMM instead

def _byte_offset(t:Tensor) -> int|None:
  # offset of t's data in its buffer, in bytes (a GGUF tensor is a view into the whole file's buffer)
  return None if (view:=t.uop.contiguous_view()) is None else view[1] * view[0].dtype.itemsize

def _quant_program(OUT:UOp, *srcs:UOp, stage:str, k:int, ntok:int, n:int=0, qtype:int=0, ksplit:int=1) -> UOp:
  if stage == "DEQUANT":
    name, defines = f"nv_q{qtype}_dequant_{n}_{k}", {"DEQUANT": 1, "QTYPE": qtype}
    dims = (UOp.special(128, "lidx0"), UOp.special(n * k // 1024, "gidx0"))
    io = int(n * k * (NV_QUANT_BYTES[qtype] + 2))
    est = Estimates(ops=3 * n * k, lds=io, mem=io)
  elif stage == "QUANTIZE":
    name, defines = f"nv_q8_quantize_{ntok}_{k}", {"QUANTIZE": 1}
    dims = (UOp.special(128, "lidx0"), UOp.special(ntok * k // 128, "gidx0"))
    io = ntok * k * (4 + 1)
    est = Estimates(ops=3 * ntok * k, lds=io, mem=io)
  else:
    name, defines = f"nv_q{qtype}_gemv_{ntok}_{n}_{k}_ks{ksplit}", {"GEMV": 1, "QTYPE": qtype, "N_DIM": n, "NTOK": ntok, "KSPLIT": ksplit}
    dims = (UOp.special(128, "lidx0"), UOp.special(-(-n // (4 // ksplit)), "gidx0"))
    io = int(n * k * NV_QUANT_BYTES[qtype]) + ntok * (k + 4 * n)
    est = Estimates(ops=2 * ntok * n * k, lds=io, mem=io)
  src = "".join(f"#define {kk} {v}\n" for kk, v in {**defines, "KNAME": name, "K_DIM": k}.items()) + QUANT_SRC
  sink = UOp.sink(OUT.base, *[x.base for x in srcs], *dims, arg=KernelInfo(name, estimates=est))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

def nv_quant_linear(x:Tensor, packed:Tensor, qtype:int, out_features:int, in_features:int) -> Tensor|None:
  """x @ W.T for a few tokens, W (out_features, in_features) stored as packed GGUF blocks of type qtype (Q4_K, Q6_K or Q8_0,
  as find_packed returns them). The result is in x's dtype (accumulated in fp32), or None if the kernel doesn't apply."""
  if qtype not in NV_QUANT_UNIT or not nv_gemm_supported(x.device) or x.device != packed.device: return None
  if x.uop.axis is not None or packed.uop.axis is not None or x.shape[-1] != in_features: return None
  tokens, N, K = prod(x.shape[:-1]), out_features, in_features
  if not isinstance(tokens, int) or not 1 <= tokens <= NV_QUANT_MAX_TOKENS or K % NV_QUANT_UNIT[qtype] or K % 128: return None
  if packed.nbytes() != int(N * K * NV_QUANT_BYTES[qtype]): return None
  if (off:=_byte_offset(packed)) is None or off % NV_QUANT_ALIGN[qtype]: return None
  # split each row's K over up to 4 warps when there are too few rows to keep every SM streaming (~64 warps per SM: a long row
  # per warp leaves the SMs idle at the end, bench/quant_bench.py)
  ksplit = 1
  while ksplit < 4 and N * ksplit < getenv("NV_QUANT_WARPS_PER_SM", 64) * num_sms(x.device) and K // NV_QUANT_STEP[qtype] >= 2 * ksplit: ksplit *= 2
  prg = functools.partial(_quant_program, k=K, ntok=tokens)
  out = Tensor.empty(tokens, N, dtype=dtypes.float32, device=x.device)
  out = Tensor.custom_kernel(out, packed, _quantize_x(x, tokens, K), fxn=functools.partial(prg, stage="GEMV", n=N, qtype=qtype, ksplit=ksplit))[0]
  return out.reshape(*x.shape[:-1], N).cast(x.dtype)

def nv_dequant(packed:Tensor, qtype:int, out_features:int, in_features:int, dep:Tensor) -> Tensor|None:
  """The (out_features, in_features) fp16 weight of packed GGUF blocks (Q4_K, Q6_K or Q8_0), or None if unsupported.
  The kernel waits for dep (the layer's input, otherwise unused) so each layer's copy is made right before its GEMM."""
  N, K = out_features, in_features
  if qtype not in NV_QUANT_UNIT or K % NV_QUANT_UNIT[qtype] or (N * K) % 1024 or packed.nbytes() != int(N * K * NV_QUANT_BYTES[qtype]): return None
  out = Tensor.empty(N, K, dtype=dtypes.half, device=packed.device)
  return Tensor.custom_kernel(out, packed, dep, fxn=functools.partial(_quant_program, stage="DEQUANT", k=K, ntok=0, n=N, qtype=qtype))[0]

_XQ_CACHE:dict[UOp, Tensor] = {}
def _quantize_x(x:Tensor, tokens:int, K:int) -> Tensor:
  # int8 activations for the GEMV. The q/k/v projections (and gate/up) take the same x one after another: quantize it once.
  # (A fresh Tensor.empty output per call would make each quantization a distinct kernel.) Keyed on the UOp, a few entries.
  if (xq:=_XQ_CACHE.get(x.uop)) is not None: return xq
  xq = Tensor.empty(tokens * (K // 4 + K // 16), dtype=dtypes.int32, device=x.device)
  xq = Tensor.custom_kernel(xq, x.reshape(tokens, K).float().contiguous(),
                            fxn=functools.partial(_quant_program, stage="QUANTIZE", k=K, ntok=tokens))[0]
  if len(_XQ_CACHE) >= 4: _XQ_CACHE.pop(next(iter(_XQ_CACHE)))
  _XQ_CACHE[x.uop] = xq
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
