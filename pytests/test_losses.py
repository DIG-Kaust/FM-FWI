"""Tests of fm_fwi.losses beyond the gradient checks in test_baselines.py.

- the stage-1 time-space mask: one before the direct arrival plus the offset, zero after it, tapered in between;
- chunked against unchunked execution of the SeisLM feature loss (loss and gradient);
- equivalence with the module that produced the 3D results (pytests/reference/losses_3d_june2026.py): same loss and
  gradient with the settings of the 3D runs (three input channels, trace standardization, masked data, both poolings).
  The one difference between the two is in the interface: fm_fwi.losses takes the observed data already masked
  (b = M d_obs, as all the 2D drivers pass it), the reference module applies Op to b itself.
The SeisLM tests run in float64 on the real encoder and are skipped when the checkpoint is absent.
"""
import os
import unittest

import numpy as np
import torch

from test_baselines import CKPT, _ricker


class TestMask(unittest.TestCase):
  def test_time_space_mask(self):
    from devitofwi.preproc.masking import TimeSpaceMasking
    nt, dt, toff, nsmooth = 1000, 0.004, 0.4, 50
    offsets = np.array([[0.0, 1.5, 3.0]])                       # km, one source, three receivers
    m = TimeSpaceMasking(offsets / 1.5, nt, dt, toff, nsmooth=nsmooth).mask[0]   # (nr, nt)
    t = np.arange(nt) * dt
    for k, tc in enumerate(offsets[0] / 1.5 + toff):
      before, after = t < tc - nsmooth * dt, t > tc + nsmooth * dt
      self.assertTrue(np.allclose(m[k, before], 1.0), f"receiver {k}: mask not 1 before the cut")
      self.assertTrue(np.allclose(m[k, after], 0.0, atol=1e-12), f"receiver {k}: mask not 0 after the cut")
      self.assertTrue(np.all(np.diff(m[k]) <= 1e-12), f"receiver {k}: mask not non-increasing")


@unittest.skipUnless(os.path.exists(CKPT), "SeisLM checkpoint not available")
class TestSeisLMLoss(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from fm_fwi.model.pretrained import load_pretrained_multidim_wav2vec2
    from pylops.basicoperators import Diagonal
    cls.model = load_pretrained_multidim_wav2vec2(CKPT).to(torch.float64).eval()
    nt, nr, ns, dt = 400, 5, 2, 0.004
    rng = np.random.default_rng(0)
    obs = [np.stack([_ricker(nt, dt, 0.4 + 0.05 * k + 0.03 * s) for k in range(nr)]).T for s in range(ns)]
    cls.syn = [np.roll(o, 8, axis=0) + 0.01 * rng.standard_normal(o.shape) for o in obs]   # (nt, nr)
    mask = np.clip(np.linspace(1.5, -0.5, nt), 0, 1)[:, None] * np.ones((1, nr))
    cls.Op = [Diagonal(mask.ravel(), dtype="float64") for _ in range(ns)]
    cls.b_raw = np.stack([o.ravel() for o in obs])
    cls.b = np.stack([cls.Op[i] @ cls.b_raw[i] for i in range(ns)])   # masked observed data, as fm_fwi.losses expects
    cls.nt, cls.nr, cls.ns = nt, nr, ns

  def _run(self, cls_, b=None, **kw):
    loss = cls_(self.model, Op=self.Op, b=self.b if b is None else b, nr=self.nr, nt=self.nt, device="cpu", normalize_traces=True,
                n_input_channels=3, **kw)
    out = []
    for i in range(self.ns):
      f = float(loss(self.syn[i].ravel(), i))
      out.append((f, np.asarray(loss.grad(self.syn[i].ravel(), i), dtype=np.float64).ravel()))
    return out

  def assertSame(self, a, b, tol, msg, gtol=None):
    for (fa, ga), (fb, gb) in zip(a, b):
      self.assertAlmostEqual(fa / fb, 1.0, delta=tol, msg=f"{msg}: loss {fa} vs {fb}")
      self.assertLess(np.linalg.norm(ga - gb) / np.linalg.norm(gb), gtol or tol, msg=f"{msg}: gradient")

  def test_chunked_equals_unchunked(self):
    from fm_fwi.losses import SeisLMFeatureLoss
    for pool in ("none", "mean"):
      self.assertSame(self._run(SeisLMFeatureLoss, pool=pool, trace_chunk=2),
                      self._run(SeisLMFeatureLoss, pool=pool, trace_chunk=None), 1e-10, f"chunked, pool={pool}")

  def test_matches_3d_run_module(self):
    from fm_fwi.losses import SeisLMFeatureLoss
    from reference.losses_3d_june2026 import SeisLMFeatureLoss as Reference
    for pool in ("mean", "none"):
      kw = dict(pool=pool, trace_chunk=2, cache_obs_embeddings=True, cache_obs_on_cpu=True)
      # gradients agree to ~4e-8: the reference applies the mask adjoint through pylops' TorchOperator (float32 step)
      self.assertSame(self._run(SeisLMFeatureLoss, **kw), self._run(Reference, b=self.b_raw, **kw), 1e-10,
                      f"3D module, pool={pool}", gtol=1e-6)


if __name__ == "__main__":
  unittest.main()
