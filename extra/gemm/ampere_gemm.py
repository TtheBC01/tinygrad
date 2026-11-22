# The Ampere fp16 GEMM lives in tinygrad/llm/kernels/nv.py (kernel: nv_gemm.cu), where the LLM's Linear layers use it.
from tinygrad.llm.kernels.nv import *  # noqa: F401,F403
from tinygrad.llm.kernels.nv import _gemm, _flags, _dims  # noqa: F401
