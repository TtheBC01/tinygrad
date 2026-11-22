import unittest, threading
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict
from tinygrad.llm.model import Transformer, TransformerConfig, DecodeBatch
from tinygrad.llm.serve import LLMServer, Request

CONFIG = TransformerConfig(num_blocks=2, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2, norm_eps=1e-5, vocab_size=100, head_dim=32,
                           rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=128)
PROMPTS = [[5, 6, 7, 8], [1, 2, 3], list(range(10, 50)), [9, 9, 9, 9, 9, 9]]

def _model() -> Transformer:
  model = Transformer(CONFIG)
  rng = np.random.RandomState(1234)
  for t in get_state_dict(model).values(): t.assign(Tensor(rng.uniform(-0.5, 0.5, t.shape).astype(np.float32))).realize()
  return model

def _greedy(prompt:list[int], n:int) -> list[int]:
  g = _model().generate(list(prompt))
  return [next(g) for _ in range(n)]

class TestDecodeBatch(unittest.TestCase):
  def test_matches_single(self):
    # rows decode together (tinygrad's attention kernels here), each row's greedy output is the single-sequence one; a row that
    # idles for a while (slot 1) and rows that join later keep their outputs too
    db = DecodeBatch(_model(), 4)
    outs = {0: [db.prefill(0, list(PROMPTS[0]))], 2: [db.prefill(2, list(PROMPTS[2]))]}
    for i in range(12):
      if i == 3: outs[1] = [db.prefill(1, list(PROMPTS[1]))]
      if i == 6: outs[3] = [db.prefill(3, list(PROMPTS[3]))]
      nxt = db.step([outs[s][-1] if s in outs and len(outs[s]) < 10 else None for s in range(4)])
      for s, t in enumerate(nxt):
        if t is not None: outs[s].append(t)
    for s, out in outs.items():
      with self.subTest(slot=s): self.assertEqual(out, _greedy(PROMPTS[s], len(out)))

  def test_rejects_models_that_ran(self):
    model = _model()
    next(model.generate([1, 2]))
    with self.assertRaises(AssertionError): DecodeBatch(model, 2)

class _Tok:
  def is_end(self, t:int) -> bool: return False

class TestServerBatching(unittest.TestCase):
  def _serve(self, reqs:list[Request], slots:int) -> list[list[int]]:
    srv = LLMServer(("127.0.0.1", 0), _model(), "test", _Tok(), None, DecodeBatch(_model(), slots))
    try:
      for r in reqs: srv.waiting.put(r)
      srv.waiting.put(None)  # the loop finishes the running requests, then returns
      outs:list[list[int]] = [[] for _ in reqs]
      def collect(i, r):
        while (t:=r.out.get()) is not None: outs[i].append(t)
      threads = [threading.Thread(target=collect, args=(i, r)) for i, r in enumerate(reqs)]
      for t in threads: t.start()
      th = threading.Thread(target=srv.serve_batched, args=(srv.batch,))
      th.start()
      for t in threads: t.join(timeout=120)
      th.join(timeout=120)
      return outs
    finally: srv.server_close()

  def test_concurrent_requests(self):
    # more requests than rows: they decode together, later ones join as rows free up; each gets its single-sequence output
    reqs = [Request(list(p), lambda: False, max_tokens=n) for p, n in zip(PROMPTS + [[3, 4]], [6, 9, 4, 7, 5])]
    outs = self._serve(reqs, 2)
    for i, (r, out) in enumerate(zip(reqs, outs)):
      with self.subTest(req=i):
        self.assertEqual(out, _greedy(r.ids, r.max_tokens))
        self.assertEqual(r.finish_reason, "length")

if __name__ == "__main__":
  unittest.main()
