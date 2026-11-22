import unittest
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig

def _model(seed:int, scale:float=0.02) -> Transformer:
  # a small llama-shaped model with random fp16 weights (the head dims the NV attention kernels take)
  cfg = TransformerConfig(num_blocks=2, dim=256, hidden_dim=512, n_heads=4, n_kv_heads=2, head_dim=64, vocab_size=512, norm_eps=1e-5,
                          rope_theta=10000.0, rope_dim=64, v_head_dim=64, max_context=256)
  m = Transformer(cfg)
  Tensor.manual_seed(seed)
  for k, p in nn.state.get_state_dict(m).items():
    v = Tensor.ones(*p.shape, dtype=p.dtype).contiguous() if "norm" in k else ((Tensor.rand(*p.shape) - 0.5) * 2 * scale).cast(p.dtype)
    p.replace(v).realize()
  return m

class TestSpeculative(unittest.TestCase):
  def _check(self, draft_seed:int, k:int, n:int=40):
    target, draft = _model(1, 0.2), _model(draft_seed, 0.2)
    prompt = [3, 17, 99, 250, 7, 42, 8]
    ref = [t for _, t in zip(range(n), target.generate(list(prompt)))]
    target._cached_tokens = []
    out = [t for _, t in zip(range(n), target.generate_speculative(draft, list(prompt), k))]
    self.assertEqual(out, ref)

  def test_same_model_draft(self):
    # a draft with the target's weights agrees with it: every round accepts all k tokens
    for k in (1, 4):
      with self.subTest(k=k): self._check(1, k)

  def test_other_draft(self):
    # an unrelated draft: rounds keep a few tokens or only the target's correction, the output is still the target's greedy output
    for k in (2, 5):
      with self.subTest(k=k): self._check(7, k)

  def test_generate_uses_draft(self):
    target, draft = _model(1, 0.2), _model(7, 0.2)
    ref = [t for _, t in zip(range(20), target.generate([5, 6, 7]))]
    target._cached_tokens, target.draft, target.draft_k = [], draft, 3
    self.assertEqual([t for _, t in zip(range(20), target.generate([5, 6, 7]))], ref)

if __name__ == "__main__":
  unittest.main()
