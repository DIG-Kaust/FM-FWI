"""float64 gradient checks for the revision losses (C8).

Baselines: torch.autograd.gradcheck on small random traces. SeisLM and hybrid: a
directional finite-difference check in float64 on the real encoder, skipped when the
checkpoint is absent.
"""

import os
import unittest

import numpy as np
import torch

from fm_fwi import baselines as bl

CKPT = os.path.join(os.path.dirname(__file__), "..",
                    "pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt")


def _ricker(nt, dt, t0, f0=5.0):
    t = np.arange(nt) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


class TestBaselineGradients(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(0)
    nr, nt, self.dt = 3, 64, 0.004
    obs = np.stack([_ricker(nt, self.dt, 0.08 + 0.02 * k) for k in range(nr)])
    self.obs = torch.tensor(obs, dtype=torch.float64)
    self.syn = (self.obs.roll(3, -1) + 0.05 * torch.randn(nr, nt, dtype=torch.float64)).requires_grad_()

  def _check(self, f):
    self.assertTrue(torch.autograd.gradcheck(lambda s: f(s, self.obs), (self.syn,), eps=1e-6, atol=1e-5))

  def test_l2(self):
    self._check(bl.l2)

  def test_standardized_l2(self):
    self._check(bl.standardized_l2)

  def test_envelope(self):
    for p in (2.0, 1.5, 1.0):
      for pad in (True, False):
        self._check(lambda s, o: bl.envelope_l2(s, o, p=p, pad=pad))

  def test_analytic_padding(self):
    """Padded and circular Hilbert transforms agree away from the trace ends."""
    a = bl._analytic(self.obs, pad=True).abs()
    b = bl._analytic(self.obs, pad=False).abs()
    np.testing.assert_allclose(a[:, 16:48].numpy(), b[:, 16:48].numpy(), atol=0.05)

  def test_wasserstein2(self):
    self._check(lambda s, o: bl.wasserstein2(s, o, dt=self.dt))

  def test_lowpass(self):
    for pad in (True, False):
      self._check(lambda s, o: bl.lowpass_l2(s, o, self.dt, fc=3.0, pad=pad))

  def test_wasserstein2_exact_two_samples(self):
    """Exact discrete OT: mass 10/11 moves from t=1 to t=0, so W2^2 = 10/11 (old code gave 18.2)."""
    syn = torch.tensor([[-1., 1.]], dtype=torch.float64)
    obs = torch.tensor([[1., -1.]], dtype=torch.float64)
    self.assertAlmostEqual(float(bl.wasserstein2(syn, obs, dt=1.0, shift=1.1)), 10 / 11)

  def test_lowpass_gain_at_cutoff(self):
    """A sinusoid on an FFT bin at fc keeps 1/sqrt(2) of its amplitude: the loss vs zero halves."""
    nt, dt = 256, 0.004
    fc = 8 / (nt * dt)                                       # exactly on bin 8 (circular filter)
    x = torch.sin(2 * np.pi * fc * torch.arange(nt, dtype=torch.float64) * dt)[None]
    z = torch.zeros_like(x)
    self.assertAlmostEqual(float(bl.lowpass_l2(x, z, dt, fc, pad=False) / bl.l2(x, z)), 0.5, places=10)

  def test_wasserstein2_zero_at_match(self):
    self.assertLess(float(bl.wasserstein2(self.obs, self.obs, dt=self.dt)), 1e-12)

  def test_trace_misfit_matches_l2(self):
    """Wrapper: masked adjoint source equals the analytic L2 one, M (M d - b)."""
    rng = np.random.default_rng(0)
    nt, nr = 16, 4
    d, mask, b = rng.standard_normal((nt, nr)), rng.uniform(size=(nt, nr)), rng.standard_normal((1, nt * nr))
    loss = bl.TraceMisfit(bl.l2, b, nt, nr, mask=[mask])
    f = loss(d.ravel(), 0)
    r = mask * d - b[0].reshape(nt, nr)
    self.assertAlmostEqual(f, 0.5 * np.sum(r ** 2))
    np.testing.assert_allclose(loss.grad(d.ravel(), 0), (mask * r).ravel())

  def test_noise_snr(self):
    rng = np.random.default_rng(1)
    d = rng.standard_normal((2, 500, 10))
    for kind in ("band", "coherent", "matched", "white", "flat"):
      n = bl.add_noise(d, 5.0, 0.004, kind=kind, band=(1.0, 12.0)) - d
      snr = 10 * np.log10(np.mean(d ** 2, axis=(1, 2)) / np.mean(n ** 2, axis=(1, 2)))
      np.testing.assert_allclose(snr, 5.0)

  def test_noise_premute(self):
    # noise is zero before each trace's first arrival and unchanged (same realisation and scale) after it
    d = np.zeros((1, 400, 3))
    for j, k in enumerate((50, 120, 300)):
      d[0, k:k + 20, j] = np.hanning(20)
    n0 = bl.add_noise(d, 10.0, 0.004, kind="matched") - d
    n1 = bl.add_noise(d, 10.0, 0.004, kind="matched", premute=1e-3) - d
    for j, k in enumerate((50, 120, 300)):
      k += 1                                  # np.hanning(20)[0] == 0
      self.assertFalse(n1[0, :k, j].any())
      np.testing.assert_array_equal(n1[0, k:, j], n0[0, k:, j])

  def test_matched_noise_spectrum(self):
    # matched noise has the data's spectrum: SNR per frequency is flat at the nominal value
    rng = np.random.default_rng(2)
    t = np.arange(1000) * 0.004
    w = (1 - 2 * (np.pi * 5 * (t - 0.4)) ** 2) * np.exp(-(np.pi * 5 * (t - 0.4)) ** 2)  # 5 Hz Ricker
    d = np.stack([np.stack([np.roll(w, k) for k in rng.integers(0, 400, 200)], 1)])
    n = bl.add_noise(d, 10.0, 0.004, kind="matched") - d
    pd, pn = (np.mean(np.abs(np.fft.rfft(x[0], axis=0)) ** 2, axis=1) for x in (d, n))
    band = (pd > 0.1 * pd.max())
    np.testing.assert_allclose(10 * np.log10(pd[band] / pn[band]), 10.0, atol=1.5)

  def test_flat_noise_spectrum(self):
    # flat noise: no power outside the band, flat inside it; white noise: flat to Nyquist
    rng = np.random.default_rng(3)
    d = rng.standard_normal((1, 1000, 200))
    f = np.fft.rfftfreq(1000, 0.004)
    for kind, inside in (("flat", (f >= 1.0) & (f <= 11.0)), ("white", np.ones_like(f, bool))):
      n = bl.add_noise(d, 10.0, 0.004, kind=kind, band=(1.0, 11.0)) - d
      pn = np.mean(np.abs(np.fft.rfft(n[0], axis=0)) ** 2, axis=1)
      self.assertLess(pn[~inside].max(initial=0.0), 1e-20 * pn.max())
      self.assertLess(np.std(pn[inside]) / np.mean(pn[inside]), 0.15)


@unittest.skipUnless(os.path.exists(CKPT), "SeisLM checkpoint not available")
class TestSeisLMGradient(unittest.TestCase):
  """Directional derivative of the SeisLM and hybrid losses vs central differences, float64."""

  def _directional(self, loss, x, i, h=1e-4):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(x.shape)
    v *= np.linalg.norm(x) / np.linalg.norm(v)
    loss(x, i)
    g = np.asarray(loss.grad(x, i), dtype=np.float64).ravel()
    fd = (loss(x + h * v, i) - loss(x - h * v, i)) / (2 * h)
    return float(g @ v), fd

  def test_feature_and_hybrid(self):
    from fm_fwi.losses import SeisLMFeatureLoss, L2, HybridLoss
    from fm_fwi.model.pretrained import load_pretrained_multidim_wav2vec2
    from pylops.basicoperators import Identity
    nt, nr = 400, 4
    obs = np.stack([_ricker(nt, 0.004, 0.4 + 0.05 * k) for k in range(nr)]).T  # (nt, nr)
    syn = np.roll(obs, 10, axis=0) + 0.01 * np.random.default_rng(1).standard_normal((nt, nr))
    b = obs.reshape(1, -1)
    model = load_pretrained_multidim_wav2vec2(CKPT).to(torch.float64)
    for pool in ("none", "mean"):
      feat = SeisLMFeatureLoss(model, Op=Identity(nt * nr, dtype="float64"), b=b, nr=nr, nt=nt, pool=pool)
      an, fd = self._directional(feat, syn.ravel(), 0)
      self.assertAlmostEqual(an / fd, 1.0, delta=1e-3, msg=f"pool={pool}: {an} vs {fd}")
    # resampling to 100 Hz (R4.7): exact chain rule through the resampling and a diagonal mask
    from pylops.basicoperators import Diagonal
    mask = np.clip(np.linspace(-0.5, 1.5, nt), 0, 1)[:, None] * np.ones((1, nr))
    for Op in (Diagonal(mask.ravel(), dtype="float64"), None):
      feat_rs = SeisLMFeatureLoss(model, Op=Op, b=(mask * obs).reshape(1, -1), nr=nr, nt=nt, pool="none",
                                  dt=0.004, resample=True, trace_chunk=2)
      an, fd = self._directional(feat_rs, syn.ravel(), 0)
      self.assertAlmostEqual(an / fd, 1.0, delta=1e-3, msg=f"resample, Op={type(Op).__name__}: {an} vs {fd}")
    hyb = HybridLoss(L2(Identity(nt * nr, dtype="float64"), b), feat, alpha=0.3, scale_l2=1.0, scale_seislm=1.0)
    an, fd = self._directional(hyb, syn.ravel(), 0)
    self.assertAlmostEqual(an / fd, 1.0, delta=1e-3, msg=f"hybrid: {an} vs {fd}")


if __name__ == "__main__":
  unittest.main()
