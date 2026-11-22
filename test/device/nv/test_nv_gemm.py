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

def _heads(out:np.ndarray, H:int) -> np.ndarray: return out.reshape(*out.shape[:2], H, -1).transpose(0, 2, 1, 3)  # (B,T,H*D) -> (B,H,T,D)

class TestNVFlashAttention(unittest.TestCase):
  def setUp(self):
    if not nv_gemm_supported(Tensor.empty(1).device): self.skipTest("NV/CUDA sm_80+ with the CUDA C renderer required")

  def _check(self, B, H, KVH, D, T, prefix, N=512):
    rng = np.random.default_rng(T + prefix + D)
    q = rng.standard_normal((B, H, T, D)).astype(np.float32)
    cache = rng.standard_normal((2, B, KVH, N, D)).astype(np.float16)  # the tail past valid_end is stale: must be ignored
    out = nv_flash_attention(Tensor(q), Tensor(cache), prefix + T)
    assert out is not None and out.shape == (B, T, H * D) and out.dtype == dtypes.float32
    np.testing.assert_allclose(_heads(out.numpy(), H), _attention_ref(q, cache, prefix + T), atol=3e-3, rtol=1e-2)

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
        assert out is not None and out.shape == (B, 1, H * D)
        np.testing.assert_allclose(_heads(out.numpy(), H), _attention_ref(q, cache, valid_end), atol=3e-3, rtol=1e-2)

  def test_decode_symbolic_jit(self):
    B, H, KVH, D, N = 1, 24, 8, 128, 2048
    rng = np.random.default_rng(1)
    cache = Tensor(rng.standard_normal((2, B, KVH, N, D)).astype(np.float16)).realize()
    vp = UOp.variable("pos", 0, N - 1)
    jf = TinyJit(lambda q, pos: nv_flash_attention(q, cache, pos + 1).realize())
    for pos in (0, 700, 5, 2047):
      q = rng.standard_normal((B, H, 1, D)).astype(np.float32)
      np.testing.assert_allclose(_heads(jf(Tensor(q), vp.bind(pos)).numpy(), H), _attention_ref(q, cache.numpy(), pos + 1), atol=3e-3,
                                 rtol=1e-2)

  def test_symbolic_jit(self):
    # chunked prefill: a symbolic chunk length T at a symbolic position, padded to the static max inside
    B, H, KVH, D, N, CH = 1, 8, 2, 64, 512, 64
    rng = np.random.default_rng(0)
    cache = Tensor(rng.standard_normal((2, B, KVH, N, D)).astype(np.float16)).realize()
    vt, vp = UOp.variable("toks", 1, CH), UOp.variable("pos", 0, N - CH)
    jf = TinyJit(lambda q, pos, toks: nv_flash_attention(q, cache, pos + toks).pad_to((B, CH, H * D)).realize())
    for T, pos in ((64, 0), (17, 64), (64, 81), (5, 300)):
      q = rng.standard_normal((B, H, CH, D)).astype(np.float32)
      t, p = vt.bind(T), vp.bind(pos)
      out = _heads(jf(Tensor(q)[:, :, :t].contiguous(), p, t).numpy()[:, :T], H)
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

  def _shared_linears(self, specs, k, seed=0, bias=False):
    # Linears whose weights are GGUF blocks of the given (qtype, rows), stored back to back in one buffer like a GGUF file
    raws = [_quant_blocks(qtype, n, k, seed=seed + i) for i, (qtype, n) in enumerate(specs)]
    pads = [(-sum(len(r) for r in raws[:i]) - 64) % 16 for i in range(len(raws))]  # 16-byte aligned starts (Q4_K), after a header
    buf = Tensor(np.concatenate([np.zeros(64, np.uint8)] + [np.concatenate([np.zeros(p, np.uint8), r]) for p, r in zip(pads, raws)])).realize()
    lins, off = [], 64
    for (qtype, n), p, raw in zip(specs, pads, raws):
      off += p
      lin = Linear(k, n, bias=bias)
      lin.weight = ggml_data_to_tensor(buf[off:off+len(raw)], n * k, qtype).reshape(n, k).cast(dtypes.half)
      if bias: lin.bias = Tensor(np.random.default_rng(seed + n).standard_normal(n).astype(np.float32)).realize()
      assert lin._nv_quant_seg() is not None
      lins.append(lin)
      off += len(raw)
    return lins

  def test_fused_linear(self):
    # one GEMV (decode) or one dequantize + GEMM (prefill) for weights of different types and row counts (blocks of a segment
    # boundary are partly idle), views into one buffer, with biases; the same as each Linear on its own
    from tinygrad.llm.kernels.amd import fused_linear
    k = 1024
    for specs in (((Q4_K, 96), (Q6_K, 36), (Q8_0, 64)), ((Q4_K, 2048), (Q4_K, 512), (Q6_K, 512)), ((Q8_0, 100), (Q8_0, 28))):
      lins = self._shared_linears(specs, k, bias=True)
      for tokens in (1, 3, 32):
        with self.subTest(specs=specs, tokens=tokens):
          x = np.random.default_rng(7).standard_normal((1, tokens, k)).astype(np.float32)
          outs = fused_linear(Tensor(x), *lins)
          for lin, out in zip(lins, outs):
            ref = x @ lin.weight.float().numpy().T + lin.bias.numpy()
            np.testing.assert_allclose(out.numpy(), ref, atol=0.01 * np.abs(ref).max(), rtol=0)

  def test_fused_kernel_count(self):
    # decode: q/k/v in one quantize and one GEMV; prefill: one dequantize and one GEMM
    from tinygrad.llm.kernels.amd import fused_linear
    from tinygrad.uop.ops import Ops
    lins = self._shared_linears(((Q4_K, 512), (Q4_K, 128), (Q6_K, 128)), 1024)
    for tokens, expect in ((1, ["nv_q8_quantize", "nv_q12x12x14_gemv"]), (64, ["nv_q12x12x14_dequant", "ampere"])):
      with self.subTest(tokens=tokens):
        outs = fused_linear(Tensor.randn(1, tokens, 1024).realize(), *lins)
        names = [u.src[0].src[0].arg.name for u in Tensor.schedule_linear(*outs).src if u.src[0].op is Ops.PROGRAM]
        kinds = [next((e for e in expect if nm.startswith(e)), nm) for nm in names if nm.startswith(("nv_", "ampere"))]
        self.assertEqual(kinds, expect)

  def test_glu_ffn(self):
    # down(gated_activation(gate(x), up(x))): in decode the activation runs in down's input quantization, in prefill in the gate/up
    # GEMM's epilogue (on rows dequantized interleaved); Llama's SwiGLU and the gpt-oss form (alpha 1.702, clamp 7, up bias 1)
    from tinygrad.llm.kernels.amd import glu_ffn
    from tinygrad.uop.ops import Ops
    k, h = 1024, 2048
    for qtypes, glu in (((Q4_K, Q4_K, Q6_K), (1.0, None, 0.0)), ((Q8_0, Q8_0, Q8_0), (1.702, 7.0, 1.0))):
      gate, up = self._shared_linears(((qtypes[0], h), (qtypes[1], h)), k, seed=10)
      down = self._shared_linears(((qtypes[2], k),), h, seed=20)[0]
      alpha, limit, up_bias = glu
      def act(g:Tensor, u:Tensor) -> Tensor:
        if limit is not None: g, u = g.clamp(max_=limit), u.clamp(-limit, limit)
        return g * (g * alpha).sigmoid() * (u + up_bias)
      wg, wu, wd = (lin.weight.float().numpy() for lin in (gate, up, down))
      for tokens in (1, 2, 32, 100):
        with self.subTest(qtypes=qtypes, tokens=tokens):
          x = np.random.default_rng(8).standard_normal((1, tokens, k)).astype(np.float32)
          x *= 4 / np.std(x @ wg.T)  # gate pre-activations ~4: through the sigmoid's curve and the clamp, and fp16 range in prefill
          g, u = x @ wg.T, x @ wu.T
          if limit is not None: g, u = np.minimum(g, limit), np.clip(u, -limit, limit)
          ref = (g / (1 + np.exp(-alpha * g)) * (u + up_bias)) @ wd.T
          if tokens == 32:
            sched = Tensor.schedule_linear(glu_ffn(Tensor(x).realize(), gate, up, down, act, glu))
            names = [u.src[0].src[0].arg.name for u in sched.src if u.src[0].op is Ops.PROGRAM]
            self.assertEqual(len(names), 4, names)  # the interleaved dequantize, the SwiGLU GEMM, down's dequantize and GEMM
            self.assertTrue(names[0].endswith("_il") and "_glu" in names[1] and names[2].startswith("nv_q") and "_f32_" in names[3], names)
          out = glu_ffn(Tensor(x), gate, up, down, act, glu)
          self.assertEqual(out.dtype, dtypes.float32)
          np.testing.assert_allclose(out.numpy(), ref, atol=0.02 * np.abs(ref).max(), rtol=0)

  def test_rmsnorm_fused(self):
    # q/k/v of an RMSNorm's output: in decode the quantize kernel applies the norm from its input (the norm's own kernels and the
    # weight's cast never run), with the weight fp32 as stored or fp32 cast to fp16 as the GGUF loader makes it; prefill as before
    from tinygrad.llm.kernels.amd import RMSNorm, fused_linear
    from tinygrad.uop.ops import Ops
    k = 1024
    lins = self._shared_linears(((Q4_K, 256), (Q6_K, 128), (Q8_0, 64)), k, seed=30)
    w32 = np.random.default_rng(31).uniform(0.5, 1.5, k).astype(np.float32)
    for xdt, wcast in ((dtypes.half, True), (dtypes.float32, False), (dtypes.half, False)):
      norm = RMSNorm(k, eps=1e-5)
      norm.weight = Tensor(w32).realize().cast(dtypes.half) if wcast else Tensor(w32).realize()
      wn = norm.weight.float().numpy()
      for tokens in (1, 3, 32):
        with self.subTest(xdt=xdt, wcast=wcast, tokens=tokens):
          x = (np.random.default_rng(32).standard_normal((1, tokens, k)) * 3).astype(np.float32)
          xt = Tensor(x).cast(xdt).realize()
          xr = xt.float().numpy()
          h = xr / np.sqrt((xr * xr).mean(-1, keepdims=True) + 1e-5) * wn
          if tokens == 1:  # (on its own input: a scheduled but unrun quantize must not be the cached one used below)
            sched = Tensor.schedule_linear(*fused_linear(norm(Tensor(x).cast(xdt).realize()), *lins))
            names = [u.src[0].src[0].arg.name for u in sched.src if u.src[0].op is Ops.PROGRAM]
            self.assertEqual(len(names), 2)  # nothing but the norm-applying quantize and the fused GEMV
            self.assertTrue(names[0].startswith("nv_q8_quantize_1_1024_norm") and names[1].startswith("nv_q12x14x8_gemv"), names)
          outs = fused_linear(norm(xt), *lins)
          for lin, out in zip(lins, outs):
            ref = h @ lin.weight.float().numpy().T
            np.testing.assert_allclose(out.float().numpy(), ref, atol=0.015 * np.abs(ref).max(), rtol=0)

  def test_mmq_prefill(self):
    # prefill (>= 128 tokens) on the int8 GEMM: activations int8 per 32 values, weights requantized exactly; mixed formats in one
    # launch (the Q6_K tiles take the two-scale path), a padded and a symbolic token count, and SwiGLU with a Q6_K down
    from tinygrad.llm.kernels.amd import fused_linear, glu_ffn
    from tinygrad.uop.ops import Ops
    k, h = 1024, 2048
    lins = self._shared_linears(((Q4_K, 256), (Q6_K, 128), (Q8_0, 128)), k, seed=50)
    gate, up = self._shared_linears(((Q4_K, h), (Q4_K, h)), k, seed=60)
    down = self._shared_linears(((Q6_K, k),), h, seed=70)[0]
    wg, wu, wd = (lin.weight.float().numpy() for lin in (gate, up, down))
    for tokens in (128, 200, 512):
      with self.subTest(tokens=tokens):
        x = np.random.default_rng(tokens).standard_normal((1, tokens, k)).astype(np.float32)
        if tokens == 512:
          names = [u.src[0].src[0].arg.name for u in Tensor.schedule_linear(*fused_linear(Tensor(x).realize(), *lins)).src
                   if u.src[0].op is Ops.PROGRAM]
          # Q4_K and Q8_0 read as packed blocks, Q6_K requantized
          self.assertTrue(any(n.startswith("nv_q14_requant_128_1024") for n in names) and
                          any(n.startswith("nv_mmq_512_512_1024") and n.endswith("_r12xi6xr8") for n in names), names)
        for lin, out in zip(lins, fused_linear(Tensor(x), *lins)):
          ref = x @ lin.weight.float().numpy().T
          np.testing.assert_allclose(out.numpy(), ref, atol=0.02 * np.abs(ref).max(), rtol=0)
        xg = x * (4 / np.std(x @ wg.T))
        g, u = xg @ wg.T, xg @ wu.T
        ref = (g / (1 + np.exp(-g)) * u) @ wd.T
        out = glu_ffn(Tensor(xg), gate, up, down, lambda a, b: a * a.sigmoid() * b, (1.0, None, 0.0)).numpy()
        np.testing.assert_allclose(out, ref, atol=0.02 * np.abs(ref).max(), rtol=0)
    vt = UOp.variable("toks", 1, 512)
    jf = TinyJit(lambda x: fused_linear(x, *lins)[0].pad_to((1, 512, 256)).realize())
    for tokens in (512, 300):
      with self.subTest(jit_tokens=tokens):
        x = np.random.default_rng(tokens + 1).standard_normal((1, 512, k)).astype(np.float32)
        out = jf(Tensor(x)[:, :vt.bind(tokens)].contiguous()).numpy()[:, :tokens]
        ref = x[:, :tokens] @ lins[0].weight.float().numpy().T
        np.testing.assert_allclose(out, ref, atol=0.02 * np.abs(ref).max(), rtol=0)

  def test_decode_attention_fused(self):
    # the output projection of flash decoding's output: its quantize kernel merges the attention splits (the merge kernel never
    # runs), for KV lengths that leave splits empty or fill all of them, batch 2, and a symbolic length under JIT
    from tinygrad.uop.ops import Ops
    for B, H, KVH, D in ((1, 16, 4, 64), (2, 8, 8, 128)):
      lin = self._shared_linears(((Q4_K, 512),), H * D, seed=40 + D)[0]
      wn = lin.weight.float().numpy()
      rng = np.random.default_rng(D)
      cache = Tensor(rng.standard_normal((2, B, KVH, 4096, D)).astype(np.float16)).realize()
      def ref(q, ve): return _attention_ref(q, cache.numpy(), ve).transpose(0, 2, 1, 3).reshape(B, 1, H * D) @ wn.T
      for ve in (1, 37, 1000, 4096):
        with self.subTest(B=B, D=D, valid_end=ve):
          q = rng.standard_normal((B, H, 1, D)).astype(np.float32)
          if ve == 1000:
            sched = Tensor.schedule_linear(lin(nv_flash_attention(Tensor(q), cache, ve)))
            names = [u.src[0].src[0].arg.name for u in sched.src if u.src[0].op is Ops.PROGRAM]
            self.assertEqual([n.split("_")[:4] for n in names if n.startswith("nv_")],
                             [["nv", "flash", "decode", "partial"], ["nv", "q8", "quantize", str(B)], ["nv", "q12", "gemv", str(B)]], names)
            self.assertIn("_attn", names[-2])
          r = ref(q, ve)
          np.testing.assert_allclose(lin(nv_flash_attention(Tensor(q), cache, ve)).numpy(), r, atol=0.015 * np.abs(r).max(), rtol=0)
      vp = UOp.variable("pos", 0, 4095)
      jf = TinyJit(lambda q, pos: lin(nv_flash_attention(q, cache, pos + 1)).realize())
      for pos in (0, 700, 5, 4095):
        with self.subTest(B=B, D=D, jit_pos=pos):
          q = rng.standard_normal((B, H, 1, D)).astype(np.float32)
          r = ref(q, pos + 1)
          np.testing.assert_allclose(jf(Tensor(q), vp.bind(pos)).numpy(), r, atol=0.015 * np.abs(r).max(), rtol=0)

  def test_unsupported_type(self):
    self.assertIsNone(nv_quant_linear(Tensor.empty(1, 256), Tensor.empty(256 * 84 // 256, dtype=dtypes.uint8), 10, 1, 256))  # Q2_K

if __name__ == "__main__":
  unittest.main()
