import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.model import spec_accept

class TestSpecAccept(unittest.TestCase):
  # spec_accept's kept tokens are distributed as the target's p whatever the draft's q, checked over many independent rows
  B, V = 200_000, 6
  P = [[0.40, 0.30, 0.20, 0.10, 0.00, 0.00], [0.05, 0.05, 0.10, 0.20, 0.25, 0.35], [0.00, 0.50, 0.00, 0.50, 0.00, 0.00]]
  Q = [[0.10, 0.10, 0.10, 0.10, 0.30, 0.30], [0.30, 0.00, 0.30, 0.00, 0.40, 0.00], [0.00, 0.50, 0.00, 0.50, 0.00, 0.00]]

  def _run(self, P, Q):
    Tensor.manual_seed(0)
    k = len(P)
    p, q = Tensor(P).reshape(1, k, self.V).expand(self.B, k, self.V), Tensor(Q).reshape(1, k, self.V).expand(self.B, k, self.V)
    props = (Tensor.rand(self.B, k, 1) > Tensor(np.cumsum(Q, -1).tolist()).reshape(1, k, self.V)).sum(-1).cast(dtypes.int32)
    out, n = spec_accept(p, q, props)
    return props.numpy(), out.numpy(), n.numpy()[:, 0]

  def _hist(self, x): return np.bincount(x, minlength=self.V) / len(x)

  def test_kept_tokens_follow_target(self):
    props, out, n = self._run(self.P, self.Q)
    for i in range(len(self.P)):
      # token i is output (kept or replaced) when the drafted tokens before it were all kept
      sel = n >= i
      with self.subTest(i=i): np.testing.assert_allclose(self._hist(out[sel, i]), self.P[i], atol=0.006)
    # each output is the drafted token while kept, then the replacement
    for i in range(len(self.P)): np.testing.assert_array_equal(out[n > i, i], props[n > i, i])
    # draft token 0 is kept with probability sum(min(p, q)); a draft equal to the target is always kept
    self.assertAlmostEqual(float((n >= 1).mean()), float(np.minimum(self.P[0], self.Q[0]).sum()), delta=0.005)
    self.assertTrue((n[n >= 2] == 3).all())  # row 2 has p == q

  def test_same_distribution_keeps_all(self):
    _, out, n = self._run(self.P[:1] * 2, self.P[:1] * 2)
    self.assertTrue((n == 2).all())
    np.testing.assert_allclose(self._hist(out[:, 1]), self.P[0], atol=0.006)

if __name__ == "__main__":
  unittest.main()
