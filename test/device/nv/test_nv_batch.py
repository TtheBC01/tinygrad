import unittest
from tinygrad.llm.model import DecodeBatch
from test.device.nv.test_nv_spec import _model

def _greedy(prompt:list[int], n:int) -> list[int]:
  m = _model(1, 0.2)
  return [t for _, t in zip(range(n), m.generate(list(prompt)))]

def _batched(db:DecodeBatch, prompts:dict[int, list[int]], n:int, temps:dict[int, float]|None=None) -> dict[int, list[int]]:
  # each slot's prompt prefilled, then n-1 steps of all rows together (slots not in prompts idle)
  temps = temps or {}
  outs = {s: [db.prefill(s, list(p), temps.get(s, 0.0))] for s, p in prompts.items()}
  for _ in range(n - 1):
    nxt = db.step([outs[s][-1] if s in outs else None for s in range(db.slots)], [temps.get(s, 0.0) for s in range(db.slots)])
    for s in outs: outs[s].append(nxt[s])
  return outs

class TestDecodeBatch(unittest.TestCase):
  prompts = {0: [3, 17, 99, 250, 7, 42, 8], 1: [5, 6, 7], 3: list(range(40, 90))}

  def test_matches_single(self):
    # each row's greedy output is the single-sequence one, whatever the other rows hold (slot 2 idles)
    db = DecodeBatch(_model(1, 0.2), 4)
    outs = _batched(db, self.prompts, 30)
    for s, p in self.prompts.items():
      with self.subTest(slot=s): self.assertEqual(outs[s], _greedy(p, 30))

  def test_sampled_rows_and_prefix_reuse(self):
    db = DecodeBatch(_model(1, 0.2), 4)
    # a sampled row next to greedy ones doesn't change them
    outs = _batched(db, self.prompts, 20, temps={1: 1.0})
    for s in (0, 3): self.assertEqual(outs[s], _greedy(self.prompts[s], 20))
    # a follow-up prompt in slot 0 reuses what the row holds: same output as a fresh run of the whole prompt
    follow = self.prompts[0] + outs[0][:-1] + [11, 12]
    self.assertEqual(db.prefix_len(0, follow), len(self.prompts[0]) + 19)
    out = _batched(db, {0: follow}, 15)[0]
    self.assertEqual(out, _greedy(follow, 15))

if __name__ == "__main__":
  unittest.main()
