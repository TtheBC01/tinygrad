"""
Hand-written Ampere fp16 GEMM (extra/gemm/ampere_gemm.cu) plugged into tinygrad with Tensor.custom_kernel.

  from extra.gemm.ampere_gemm import ampere_matmul
  c = ampere_matmul(a, b)          # a: (M,K) half, b: (K,N) half -> (M,N) half, fp32 accumulate

Falls back to a.matmul(b) when the device/dtype/shape isn't supported.
Run as a script for a correctness + speed check: DEV=NV python extra/gemm/ampere_gemm.py
"""
import functools, pathlib
from dataclasses import dataclass
from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates

SRC = (pathlib.Path(__file__).parent / "ampere_gemm.cu").read_text()

@dataclass(frozen=True)
class Cfg:
  BM: int = 128
  BN: int = 128
  BK: int = 32
  WM: int = 64
  WN: int = 64
  STAGES: int = 3
  GROUP_M: int = 0
  @property
  def threads(self) -> int: return (self.BM // self.WM) * (self.BN // self.WN) * 32
  @property
  def smem(self) -> int: return self.STAGES * (self.BM * self.BK + self.BK * self.BN) * 2
  def __str__(self): return f"{self.BM}x{self.BN}x{self.BK}_w{self.WM}x{self.WN}_s{self.STAGES}" + (f"_g{self.GROUP_M}" if self.GROUP_M else "")

DEFAULT = Cfg()
MAX_SMEM = 99 * 1024  # per-block shared memory limit on sm_86/sm_89 (sm_80 allows 163 KB)

def auto_cfg(M:int, N:int, K:int) -> Cfg:
  # 128x128x32 / 4 warps / 4 stages (64 KB dynamic smem) was the best or within 2% of the best config on every shape
  # benchmarked on a 3080 (bench/logs/ampere_bench_dyn.log). Grouped rasterization only paid off on very large grids.
  return Cfg(STAGES=4, GROUP_M=8 if (M // DEFAULT.BM) * (N // DEFAULT.BN) >= 2048 else 0)

def supported(a:Tensor, b:Tensor, cfg:Cfg=DEFAULT) -> bool:
  if not (isinstance(a.device, str) and a.device.split(":")[0] in {"NV", "CUDA"}): return False
  if a.dtype != dtypes.half or b.dtype != dtypes.half or a.ndim != 2 or b.ndim != 2: return False
  if cfg.smem > MAX_SMEM: return False
  (M, K), (K2, N) = a.shape, b.shape
  return all(isinstance(x, int) for x in (M, K, N)) and K == K2 and M % cfg.BM == 0 and N % cfg.BN == 0 and K % cfg.BK == 0

@functools.cache
def _source(M:int, N:int, K:int, cfg:Cfg) -> tuple[str, str]:
  name = f"ampere_gemm_{M}_{N}_{K}_{cfg}"
  defines = dict(KNAME=name, M_DIM=M, N_DIM=N, K_DIM=K, BM=cfg.BM, BN=cfg.BN, BK=cfg.BK, WM=cfg.WM, WN=cfg.WN, STAGES=cfg.STAGES, GROUP_M=cfg.GROUP_M)
  return name, "".join(f"#define {k} {v}\n" for k, v in defines.items()) + SRC

def _program(C:UOp, A:UOp, B:UOp, cfg:Cfg) -> UOp:
  (M, K), N = A.shape, B.shape[1]
  name, src = _source(M, N, K, cfg)
  threads = UOp.special(cfg.threads, "lidx0")
  bx, by = UOp.special(N // cfg.BN, "gidx0"), UOp.special(M // cfg.BM, "gidx1")
  est = Estimates(ops=2 * M * N * K, lds=2 * (M * K + K * N + M * N), mem=2 * (M * K + K * N + M * N))
  sink = UOp.sink(C.base, A.base, B.base, threads, bx, by, arg=KernelInfo(name, estimates=est, dyn_shared_mem=cfg.smem))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

def ampere_matmul(a:Tensor, b:Tensor, cfg:Cfg|None=None) -> Tensor:
  """Forward only (no grad_fxn yet). Falls back to a.matmul(b) for unsupported inputs."""
  if cfg is None and a.ndim == 2 and b.ndim == 2 and all(isinstance(x, int) for x in (*a.shape, b.shape[1])):
    cfg = auto_cfg(a.shape[0], b.shape[1], a.shape[1])
  if cfg is None or not supported(a, b, cfg): return a.matmul(b)
  a, b = a.contiguous(), b.contiguous()
  out = Tensor.invalids(a.shape[0], b.shape[1], dtype=dtypes.half, device=a.device)
  return Tensor.custom_kernel(out, a, b, fxn=functools.partial(_program, cfg=cfg))[0]

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
