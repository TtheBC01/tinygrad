import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.kernels.nv import ampere_matmul, nv_linear, nv_flash_attention, nv_gemm_supported, _gemm, Cfg, BIG, MID, SMALL, ROWS32, ROWS64
from tinygrad.llm.kernels.nv import nv_quant_linear, Q8_0, Q4_K, Q6_K
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear

def _rand(*shape, seed=0): return np.random.default_rng(seed).standard_normal(shape).astype(np.float16)
def _tol(k:int): return dict(atol=0.01 * np.sqrt(k) + 0.02, rtol=2e-3)  # one fp16 rounding of values ~sqrt(k), plus summation order

class TestNVGemm(unittest.TestCase):
  def setUp(self):
    if not nv_gemm_supported(Tensor.empty(1).device): self.skipTest("NV/CUDA sm_80+ with the CUDA C renderer required")

  def _check(self, M, K, N, cfg=None, ta=False, tb=False):
    an, bn = _rand(*((K, M) if ta else (M, K)), seed=1), _rand(*((N, K) if tb else (K, N)), seed=2)
    a32, b32 = an.astype(np.float32), bn.astype(np.float32)
    ref = (a32.T if ta else a32) @ (b32.T if tb else b32)
    jf = TinyJit(lambda a, b: _gemm(a, b, ta, tb, cfg).realize())
    for _ in range(3):  # capture + replays: stream-K flags are back at 0 after every launch
      np.testing.assert_allclose(jf(Tensor(an), Tensor(bn)).float().numpy(), ref, **_tol(K))

  def test_configs(self):
    for cfg in (BIG, MID, SMALL, ROWS32, ROWS64):
      for sk in (True, False):
        with self.subTest(cfg=str(cfg), sk=sk): self._check(512, 1024, 768, Cfg(**{**cfg.__dict__, "SK": sk}))

  def test_stream_k_shapes(self):
    # fewer tiles than SMs, a partial last wave, and a small K split across many workers
    for M, K, N in ((256, 8192, 256), (4096, 1024, 1536), (384, 96, 640)):
      with self.subTest(shape=(M, K, N)): self._check(M, K, N)

  def test_transposed(self):
    for ta, tb in ((False, True), (True, False), (True, True)):
      with self.subTest(ta=ta, tb=tb): self._check(512, 768, 1024, ta=ta, tb=tb)

  def test_bf16(self):
    def bf16(x):  # round to the nearest bf16, ties to even (numpy has no bf16)
      u = x.view(np.uint32)
      return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)
    rng = np.random.default_rng(0)
    a32, b32 = rng.standard_normal((512, 1024)).astype(np.float32), rng.standard_normal((1024, 768)).astype(np.float32)
    ab, bb = bf16(a32), bf16(b32)
    for tb in (False, True):
      a, b = Tensor(a32).cast(dtypes.bfloat16), Tensor(b32.T.copy() if tb else b32).cast(dtypes.bfloat16)
      out = _gemm(a, b, tb=tb)
      self.assertEqual(out.dtype, dtypes.bfloat16)
      np.testing.assert_allclose(out.float().numpy(), ab @ bb, atol=0.1 * np.sqrt(1024) / 8 + 0.1, rtol=1e-2)
      np.testing.assert_allclose(_gemm(a, b, tb=tb, out_dtype=dtypes.float32).numpy(), ab @ bb, **_tol(1024))

  def test_unsupported_falls_back(self):
    a, b = Tensor(_rand(100, 64)), Tensor(_rand(64, 72, seed=1))
    np.testing.assert_allclose(ampere_matmul(a, b).numpy(), (a.float() @ b.float()).numpy(), atol=0.05, rtol=1e-2)

  def test_gradients(self):
    an, bn, gn = _rand(256, 512), _rand(512, 384, seed=1), _rand(256, 384, seed=2)
    a, b = Tensor(an).realize(), Tensor(bn).realize()
    a.requires_grad, b.requires_grad = True, True
    (ampere_matmul(a, b).float() * Tensor(gn).float()).sum().backward()
    g32 = gn.astype(np.float32)
    np.testing.assert_allclose(a.grad.float().numpy(), g32 @ bn.astype(np.float32).T, **_tol(384))
    np.testing.assert_allclose(b.grad.float().numpy(), an.astype(np.float32).T @ g32, **_tol(256))

class TestNVLinear(unittest.TestCase):
  def setUp(self):
    if not nv_gemm_supported(Tensor.empty(1).device): self.skipTest("NV/CUDA sm_80+ with the CUDA C renderer required")

  def test_fp32_activations(self):
    # the LLM's residual stream is fp32: cast to fp16 for the tensor cores, fp32 out
    x, w = np.random.default_rng(0).standard_normal((2, 37, 512)).astype(np.float32), _rand(768, 512, seed=1)
    out = nv_linear(Tensor(x), Tensor(w))
    assert out is not None and out.dtype == dtypes.float32 and out.shape == (2, 37, 768)
    np.testing.assert_allclose(out.numpy(), x.astype(np.float16).astype(np.float32) @ w.astype(np.float32).T, **_tol(512))

  def test_decode_uses_tinygrad(self):
    self.assertIsNone(nv_linear(Tensor.empty(1, 1, 512), Tensor.empty(768, 512, dtype=dtypes.half)))

  def test_symbolic_tokens_jit(self):
    layer = Linear(512, 768, bias=True)
    layer.weight, layer.bias = Tensor(_rand(768, 512)).realize(), Tensor.randn(768).realize()
    jf = TinyJit(lambda x: layer(x).pad_to((1, 64, 768)).realize())  # back to a static shape to read it
    v = UOp.variable("toks", 1, 64)
    for n in (64, 13, 40, 7):
      x = np.random.default_rng(n).standard_normal((1, 64, 512)).astype(np.float32)
      out = jf(Tensor(x)[:, :v.bind(n)].contiguous()).numpy()
      ref = x[:, :n].astype(np.float16).astype(np.float32) @ layer.weight.numpy().astype(np.float32).T + layer.bias.numpy()
      np.testing.assert_allclose(out[:, :n], ref, **_tol(512))

def _attention_ref(q, cache, valid_end):
  # q (B,H,T,D) fp32, cache (2,B,KVH,N,D): causal attention of the T queries (positions valid_end-T..) over keys [0, valid_end)
  B, H, T, D = q.shape
  k, v = (np.repeat(cache[i, :, :, :valid_end].astype(np.float32), H // cache.shape[2], axis=1) for i in range(2))
  s = q.astype(np.float16).astype(np.float32) @ k.transpose(0, 1, 3, 2) / np.sqrt(D)
  pos = valid_end - T + np.arange(T)[:, None]
  s = np.where(np.arange(valid_end)[None, :] <= pos, s, -np.inf)
  p = np.exp(s - s.max(-1, keepdims=True))
  return (p / p.sum(-1, keepdims=True)) @ v

class TestNVFlashAttention(unittest.TestCase):
  def setUp(self):
    if not nv_gemm_supported(Tensor.empty(1).device): self.skipTest("NV/CUDA sm_80+ with the CUDA C renderer required")

  def _check(self, B, H, KVH, D, T, prefix, N=512):
    rng = np.random.default_rng(T + prefix + D)
    q = rng.standard_normal((B, H, T, D)).astype(np.float32)
    cache = rng.standard_normal((2, B, KVH, N, D)).astype(np.float16)  # the tail past valid_end is stale: must be ignored
    out = nv_flash_attention(Tensor(q), Tensor(cache), prefix + T)
    assert out is not None and out.shape == (B, H, T, D) and out.dtype == dtypes.float32
    np.testing.assert_allclose(out.numpy(), _attention_ref(q, cache, prefix + T), atol=3e-3, rtol=1e-2)

  def test_shapes(self):
    for B, H, KVH, D, T, prefix in ((1, 8, 2, 64, 128, 0), (1, 8, 8, 128, 64, 0), (2, 4, 2, 128, 64, 100), (1, 8, 2, 64, 32, 37),
                                    (1, 4, 1, 64, 256, 200)):
      with self.subTest(B=B, H=H, KVH=KVH, D=D, T=T, prefix=prefix): self._check(B, H, KVH, D, T, prefix)

  def test_decode(self):
    # one query token (flash decoding): KV lengths below, at and above the split count, GQA groups of 1-4, a stale cache tail
    for B, H, KVH, D, valid_end in ((1, 24, 8, 128, 1025), (1, 24, 8, 128, 3), (2, 8, 2, 64, 300), (1, 8, 8, 64, 1), (1, 32, 8, 64, 4096)):
      with self.subTest(B=B, H=H, KVH=KVH, D=D, valid_end=valid_end):
        rng = np.random.default_rng(valid_end + D)
        q = rng.standard_normal((B, H, 1, D)).astype(np.float32)
        cache = rng.standard_normal((2, B, KVH, 4096, D)).astype(np.float16)
        out = nv_flash_attention(Tensor(q), Tensor(cache), valid_end)
        assert out is not None and out.shape == (B, H, 1, D)
        np.testing.assert_allclose(out.numpy(), _attention_ref(q, cache, valid_end), atol=3e-3, rtol=1e-2)

  def test_decode_symbolic_jit(self):
    B, H, KVH, D, N = 1, 24, 8, 128, 2048
    rng = np.random.default_rng(1)
    cache = Tensor(rng.standard_normal((2, B, KVH, N, D)).astype(np.float16)).realize()
    vp = UOp.variable("pos", 0, N - 1)
    jf = TinyJit(lambda q, pos: nv_flash_attention(q, cache, pos + 1).realize())
    for pos in (0, 700, 5, 2047):
      q = rng.standard_normal((B, H, 1, D)).astype(np.float32)
      np.testing.assert_allclose(jf(Tensor(q), vp.bind(pos)).numpy(), _attention_ref(q, cache.numpy(), pos + 1), atol=3e-3, rtol=1e-2)

  def test_symbolic_jit(self):
    # chunked prefill: a symbolic chunk length T at a symbolic position, padded to the static max inside
    B, H, KVH, D, N, CH = 1, 8, 2, 64, 512, 64
    rng = np.random.default_rng(0)
    cache = Tensor(rng.standard_normal((2, B, KVH, N, D)).astype(np.float16)).realize()
    vt, vp = UOp.variable("toks", 1, CH), UOp.variable("pos", 0, N - CH)
    jf = TinyJit(lambda q, pos, toks: nv_flash_attention(q, cache, pos + toks).pad_to((B, H, CH, D)).realize())
    for T, pos in ((64, 0), (17, 64), (64, 81), (5, 300)):
      q = rng.standard_normal((B, H, CH, D)).astype(np.float32)
      t, p = vt.bind(T), vp.bind(pos)
      out = jf(Tensor(q)[:, :, :t].contiguous(), p, t).numpy()[:, :, :T]
      np.testing.assert_allclose(out, _attention_ref(q[:, :, :T], cache.numpy(), pos + T), atol=3e-3, rtol=1e-2)

def _quant_blocks(qtype:int, n:int, k:int, seed=0) -> np.ndarray:
  # random GGUF blocks for an (n, k) weight, with small finite fp16 scales (random bytes elsewhere: every bit pattern is a valid weight)
  bsz, nbytes = {Q8_0: (32, 34), Q4_K: (256, 144), Q6_K: (256, 210)}[qtype]
  rng = np.random.default_rng(seed)
  blocks = rng.integers(0, 256, (n * k // bsz, nbytes), dtype=np.uint8)
  def put_half(off, lo, hi): blocks[:, off:off+2] = rng.uniform(lo, hi, len(blocks)).astype(np.float16).view(np.uint8).reshape(-1, 2)
  if qtype == Q8_0: put_half(0, 0.001, 0.01)
  if qtype == Q4_K:
    for off in (0, 2): put_half(off, 0.001, 0.01)
  if qtype == Q6_K: put_half(208, 0.0005, 0.005)
  return blocks.reshape(-1)

class TestNVQuant(unittest.TestCase):
  def setUp(self):
    if not nv_gemm_supported(Tensor.empty(1).device): self.skipTest("NV/CUDA sm_80+ with the CUDA C renderer required")

  def _check(self, qtype, n, k, tokens):
    raw = _quant_blocks(qtype, n, k)
    w = ggml_data_to_tensor(Tensor(raw), n * k, qtype).reshape(n, k).numpy()  # tinygrad's reference dequantization
    x = np.random.default_rng(1).standard_normal((tokens, k)).astype(np.float32)
    word = dtypes.uint32 if qtype == Q4_K else dtypes.uint16
    out = nv_quant_linear(Tensor(x), Tensor(raw).bitcast(word).contiguous(), qtype, n, k)
    assert out is not None
    ref = x @ w.T
    # the activations are rounded to int8 per 32 values (~0.4% of each block's max): compare against the output's scale
    np.testing.assert_allclose(out.numpy(), ref, atol=0.01 * np.abs(ref).max(), rtol=0)

  def test_types_shapes(self):
    for qtype in (Q8_0, Q4_K, Q6_K):
      for n, k, tokens in ((512, 2048, 1), (3072, 3072, 1), (1024, 8192, 3), (130, 1024, 8)):
        with self.subTest(qtype=qtype, n=n, k=k, tokens=tokens): self._check(qtype, n, k, tokens)

  def test_linear_finds_packed(self):
    # a Linear whose weight is the GGUF dequantization runs the quant kernel for decode and the fp16 GEMM for prefill
    n, k = 768, 1024
    for qtype in (Q8_0, Q4_K, Q6_K):
      with self.subTest(qtype=qtype):
        raw = _quant_blocks(qtype, n, k, seed=2)
        lin = Linear(k, n, bias=False)
        lin.weight = ggml_data_to_tensor(Tensor(raw).realize(), n * k, qtype).reshape(n, k).cast(dtypes.half)
        assert lin._nv_packed() is not None and lin._nv_packed()[0] == qtype
        w = lin.weight.float().numpy()
        for tokens in (1, 64):
          x = np.random.default_rng(3).standard_normal((1, tokens, k)).astype(np.float32)
          ref = x @ w.T
          np.testing.assert_allclose(lin(Tensor(x)).numpy(), ref, atol=0.01 * np.abs(ref).max(), rtol=0)

  def test_rope_permuted_weight(self):
    # a GGUF q/k projection: the loader permutes the rows for RoPE, the kernel reads the unpermuted blocks and permutes its output
    from tinygrad.llm.model import permute_rope
    n, k = 1024, 2048
    for n_heads, rope_dim in ((16, 64), (8, 128), (16, 32)):  # q (rope over the whole head), k, and partial rope (a head prefix)
      with self.subTest(n_heads=n_heads, rope_dim=rope_dim):
        w0 = ggml_data_to_tensor(Tensor(_quant_blocks(Q4_K, n, k, seed=4)).realize(), n * k, Q4_K).reshape(n, k).cast(dtypes.half)
        lin = Linear(k, n, bias=False)
        lin.weight, lin.rope_source = permute_rope(w0, n_heads, rope_dim), (w0, n_heads, rope_dim)
        assert lin._nv_packed() is not None
        w = lin.weight.float().numpy()
        for tokens in (1, 32):  # decode GEMV, prefill on the dequantized weight
          x = np.random.default_rng(5).standard_normal((1, tokens, k)).astype(np.float32)
          ref = x @ w.T
          np.testing.assert_allclose(lin(Tensor(x)).numpy(), ref, atol=0.01 * np.abs(ref).max(), rtol=0)

  def test_prefill_dequant_waits_for_input(self):
    # each layer's fp16 copy of a quant weight is made after the previous layer's GEMM (it waits for the layer's input), so the
    # copies don't all live at once: two chained layers schedule as dequant, gemm, dequant, gemm
    from tinygrad.uop.ops import Ops
    n = k = 1024
    layers = []
    for i in range(2):
      lin = Linear(k, n, bias=False)
      lin.weight = ggml_data_to_tensor(Tensor(_quant_blocks(Q4_K, n, k, seed=i)).realize(), n * k, Q4_K).reshape(n, k).cast(dtypes.half)
      layers.append(lin)
    out = layers[1](layers[0](Tensor.randn(1, 64, k).realize()))
    names = [u.src[0].src[0].arg.name for u in out.schedule_linear().src if u.src[0].op is Ops.PROGRAM]
    order = [nm.split("_")[0] + ("_dequant" if "dequant" in nm else "") for nm in names]
    self.assertEqual(order, ["nv_dequant", "ampere", "nv_dequant", "ampere"])

  def test_unsupported_type(self):
    self.assertIsNone(nv_quant_linear(Tensor.empty(1, 256), Tensor.empty(256 * 84 // 256, dtype=dtypes.uint8), 10, 1, 256))  # Q2_K

if __name__ == "__main__":
  unittest.main()
