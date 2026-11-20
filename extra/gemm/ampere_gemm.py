"""
Hand-written Ampere fp16 GEMM (extra/gemm/ampere_gemm.cu) plugged into tinygrad with Tensor.custom_kernel.

  from extra.gemm.ampere_gemm import ampere_matmul
  c = ampere_matmul(a, b)          # a: (M,K) half, b: (K,N) half -> (M,N) half, fp32 accumulate

Falls back to a.matmul(b) when the device/dtype/shape isn't supported.
With Cfg.SK (default), tiles that would leave the last wave partly empty are split along K across every SM (stream-K),
which needs a small fp32 workspace and a zeroed flag buffer per call.
Run as a script for a correctness + speed check: DEV=NV python extra/gemm/ampere_gemm.py
"""
import ctypes, functools, pathlib
from dataclasses import dataclass
from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates

SRC = (pathlib.Path(__file__).parent / "ampere_gemm.cu").read_text()

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

# Sustained (power capped) benchmarks on a 3080: the big tile with stream-K was the best or tied on every shape tried
# (bench/logs/soak_sweep.log); big tiles move less data per FLOP, so the card holds a higher clock under its power limit.
# Smaller tiles are fallbacks for shapes the big tile doesn't divide.
AUTO_CFGS = (Cfg(), Cfg(BM=128, WN=32), Cfg(BM=128, BN=64, WN=32))

def auto_cfg(M:int, N:int, K:int) -> Cfg:
  return next((c for c in AUTO_CFGS if M % c.BM == 0 and N % c.BN == 0 and K % c.BK == 0), DEFAULT)

@functools.cache
def num_sms(device:str) -> int:
  dev = Device[device]
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

def supported(a:Tensor, b:Tensor, cfg:Cfg=DEFAULT) -> bool:
  if not (isinstance(a.device, str) and a.device.split(":")[0] in {"NV", "CUDA"}): return False
  if a.dtype != dtypes.half or b.dtype != dtypes.half or a.ndim != 2 or b.ndim != 2: return False
  if cfg.smem > MAX_SMEM: return False
  (M, K), (K2, N) = a.shape, b.shape
  return all(isinstance(x, int) for x in (M, K, N)) and K == K2 and M % cfg.BM == 0 and N % cfg.BN == 0 and K % cfg.BK == 0

@functools.cache
def _source(M:int, N:int, K:int, cfg:Cfg, sk_p:int) -> tuple[str, str]:
  name = f"ampere_gemm_{M}_{N}_{K}_{cfg}" + (f"{sk_p}" if sk_p else "")
  defines = dict(KNAME=name, M_DIM=M, N_DIM=N, K_DIM=K, BM=cfg.BM, BN=cfg.BN, BK=cfg.BK, WM=cfg.WM, WN=cfg.WN, STAGES=cfg.STAGES,
                 GROUP_M=cfg.GROUP_M, SK_P=sk_p)
  return name, "".join(f"#define {k} {v}\n" for k, v in defines.items()) + SRC

def _program(C:UOp, A:UOp, B:UOp, WS:UOp, FLAGS:UOp, cfg:Cfg, sp:Split) -> UOp:
  (M, K), N = A.shape, B.shape[1]
  name, src = _source(M, N, K, cfg, sp.sk_p)
  threads, blocks = UOp.special(cfg.threads, "lidx0"), UOp.special(sp.grid, "gidx0")
  est = Estimates(ops=2 * M * N * K, lds=2 * (M * K + K * N + M * N), mem=2 * (M * K + K * N + M * N))
  sink = UOp.sink(C.base, A.base, B.base, WS.base, FLAGS.base, threads, blocks, arg=KernelInfo(name, estimates=est, dyn_shared_mem=cfg.smem))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

@functools.cache
def _flags(device:str) -> Tensor:
  # Stream-K handoff flags. They must be 0 when a launch starts; every launch resets each flag it sets before it finishes, so
  # one zeroed buffer per device is reused (a fresh zeros() per call costs an extra kernel launch, ~5% on a 2048^3 GEMM).
  # This relies on kernels on a device running in order, one after another, as tinygrad's NV (dependent QMDs) and
  # CUDA (one stream) runtimes do. A launch that faults can leave a flag set; the next one would then read a stale partial.
  return Tensor.zeros(MAX_SK_WORKERS, dtype=dtypes.int32, device=device).contiguous().realize()
MAX_SK_WORKERS = 4096  # >= SMs x blocks per SM on any GPU this runs on

def ampere_matmul(a:Tensor, b:Tensor, cfg:Cfg|None=None) -> Tensor:
  """Forward only (no grad_fxn yet). Falls back to a.matmul(b) for unsupported inputs."""
  if cfg is None and a.ndim == 2 and b.ndim == 2 and all(isinstance(x, int) for x in (*a.shape, b.shape[1])):
    cfg = auto_cfg(a.shape[0], b.shape[1], a.shape[1])
  if cfg is None or not supported(a, b, cfg): return a.matmul(b)
  a, b = a.contiguous(), b.contiguous()
  (M, K), N = a.shape, b.shape[1]
  sp = split(M, N, K, cfg, num_sms(a.device) if cfg.SK else 0)
  assert sp.sk_workers <= MAX_SK_WORKERS
  out = Tensor.invalids(M, N, dtype=dtypes.half, device=a.device)
  # stream-K scratch: fp32 partial tiles (one per worker), and the per-device flag buffer (one-element dummy when stream-K is off)
  ws = Tensor.empty(sp.sk_workers * cfg.BM * cfg.BN if sp.sk_workers else 1, dtype=dtypes.float32, device=a.device)
  flags = _flags(a.device) if sp.sk_workers else Tensor.empty(1, dtype=dtypes.int32, device=a.device)
  return Tensor.custom_kernel(out, a, b, ws, flags, fxn=functools.partial(_program, cfg=cfg, sp=sp))[0]

if __name__ == "__main__":
  import numpy as np
  from tinygrad.helpers import getenv
  N = getenv("N", 4096)
  a, b = Tensor.randn(N, N, dtype=dtypes.half).realize(), Tensor.randn(N, N, dtype=dtypes.half).realize()
  c = ampere_matmul(a, b).realize()
  ref = a.float().numpy() @ b.float().numpy()
  err = np.abs(c.float().numpy() - ref).max()
  print(f"{Device.DEFAULT} N={N} cfg={auto_cfg(N, N, N)}  max abs err vs fp32 numpy: {err:.4f}")
  np.testing.assert_allclose(c.float().numpy(), ref, atol=0.5, rtol=1e-2)
  print("correct")
