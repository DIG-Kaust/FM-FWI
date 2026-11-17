"""Baseline misfits and data perturbations.

Every misfit is a torch function ``misfit(syn, obs) -> scalar`` of traces shaped
``[nr, nt]``. :class:`TraceMisfit` wraps one as a devitofwi loss (``__call__(x, i)``
and ``grad(x, i)``) and gets the adjoint source from autograd, so all baselines share
one code path and one gradient test (pytests/test_baselines.py).
"""

__all__ = ["TraceMisfit", "l2", "standardized_l2", "envelope_l2", "wasserstein2",
           "lowpass_l2", "add_noise", "fix_tslearn_softdtw"]

import numpy as np
import torch


class TraceMisfit:
    """devitofwi-style loss around a torch trace misfit.

    Parameters
    ----------
    misfit : callable
        ``misfit(syn, obs)`` on ``[nr, nt]`` tensors, returning a scalar tensor.
    b : numpy.ndarray
        Observed shots, ``b[i]`` flattenable to ``(nt, nr)`` (already masked).
    nt, nr : int
        Samples per trace and receivers per shot.
    mask : list of numpy.ndarray, optional
        Per-shot ``(nt, nr)`` mask applied to the modelled data (and to the
        adjoint source, since the mask is diagonal).
    """

    def __init__(self, misfit, b, nt, nr, mask=None):
        self.misfit, self.b, self.nt, self.nr, self.mask = misfit, b, nt, nr, mask
        self._loss_value = None
        self._grad_cache = None

    def __call__(self, x, i):
        d = np.asarray(x).reshape(self.nt, self.nr)
        if self.mask is not None:
            d = d * self.mask[i]
        syn = torch.tensor(d.T, dtype=torch.float64, requires_grad=True)
        obs = torch.as_tensor(np.asarray(self.b[i]).reshape(self.nt, self.nr).T, dtype=torch.float64)
        f = self.misfit(syn, obs)
        (g,) = torch.autograd.grad(f, syn)
        g = g.numpy().T
        if self.mask is not None:
            g = g * self.mask[i]
        # The engine always calls grad(x, i) right after __call__(x, i) on the same x,
        # so the adjoint source is computed here and cached by shot index.
        self._grad_cache = (i, g.ravel().astype(np.asarray(x).dtype))
        self._loss_value = float(f)
        return self._loss_value

    def grad(self, x, i):
        if self._grad_cache is None or self._grad_cache[0] != i:
            self(x, i)
        return self._grad_cache[1]


class _SoftDTW64(torch.autograd.Function):
    """tslearn 0.9.0's soft-DTW autograd Function (metrics/soft_dtw_loss_pytorch.py) with two fixes.

    1. backward passes D padded by one zero row and column to _njit_soft_dtw_grad, which takes m = D.shape[0] - 1
       (tslearn's numpy SoftDTW pads; the PyTorch backward does not, so its gradient was that of the series without
       their last sample, and it zeroed the last row and column of the saved D in place). On the Overthrust traces
       the normalized loss's gradient was off by 20-100% (scripts/diag_softdtw_fd.py).
    2. R stays float64 (torch.Tensor cast it to float32).
    """

    @staticmethod
    def forward(ctx, D, gamma):
        from tslearn.metrics.soft_dtw_fast import _njit_soft_dtw_batch
        D_ = D.detach().cpu().numpy().astype(np.float64)
        R_ = np.zeros((D_.shape[0], D_.shape[1] + 2, D_.shape[2] + 2))
        _njit_soft_dtw_batch(D_, R_, gamma)
        ctx.save_for_backward(D, torch.from_numpy(R_))
        ctx.gamma = gamma
        return torch.from_numpy(R_[:, -2, -2].copy()).to(D)

    @staticmethod
    def backward(ctx, grad_output):
        from tslearn.metrics.soft_dtw_fast import _njit_soft_dtw_grad_batch
        D, R = ctx.saved_tensors
        b, m, n = D.shape
        D_ = np.pad(D.detach().cpu().numpy().astype(np.float64), ((0, 0), (0, 1), (0, 1)))
        E_ = np.zeros((b, m + 2, n + 2))
        _njit_soft_dtw_grad_batch(D_, R.numpy().copy(), E_, ctx.gamma)
        E = torch.from_numpy(E_[:, 1:m + 1, 1:n + 1].copy()).to(D)
        return grad_output.view(-1, 1, 1) * E, None


def fix_tslearn_softdtw():
    """Make tslearn's SoftDTWLossPyTorch (used by devitofwi's SoftDTW) use _SoftDTW64."""
    # Module-global swap, read at call time by SoftDTWLossPyTorch.forward.
    import tslearn.metrics.soft_dtw_loss_pytorch as sdl
    sdl._SoftDTWLossPyTorch = _SoftDTW64


def l2(syn, obs):
    return 0.5 * torch.sum((syn - obs) ** 2)


def standardized_l2(syn, obs, eps=1e-6):
    """L2 after the per-trace standardization SeisLMFeatureLoss applies."""
    def std(x):
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, unbiased=False).clamp_min(eps)
    return l2(std(syn), std(obs))


def _analytic(x, pad=True):
    """Analytic signal along the last axis (FFT Hilbert transform).

    pad=True zero-pads to twice the length first, which reduces (does not remove) wraparound
    of late energy to early times (as in ADFWI); pad=False is the circular transform of
    scipy.signal.hilbert.
    """
    nt = x.shape[-1]
    n = 2 * nt if pad else nt
    h = torch.zeros(n, dtype=x.dtype, device=x.device)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return torch.fft.ifft(torch.fft.fft(x, n=n, dim=-1) * h, dim=-1)[..., :nt]


def envelope_l2(syn, obs, p=2.0, pad=True, eps=1e-3, eps_abs=1e-30):
    """L2 between envelopes raised to the power p: 0.5 sum (E_syn^p - E_obs^p)^2.

    p=2 is the squared envelope (Wu et al., 2014), p=1 the plain envelope (Yuan et al., 2015;
    SeisFlows), p=1.5 ADFWI's default. For p<2 the envelope sqrt(re^2+im^2) is smoothed with
    eps * max|obs| (floored at eps_abs) so its derivative stays finite where the trace is zero.
    """
    def e2(x):
        z = _analytic(x, pad)
        return z.real.square() + z.imag.square()
    if p == 2.0:
        return l2(e2(syn), e2(obs))
    d2 = (eps * obs.detach().abs().max()).clamp_min(eps_abs).square()
    return l2((e2(syn) + d2) ** (p / 2), (e2(obs) + d2) ** (p / 2))


def wasserstein2(syn, obs, dt=1.0, shift=1.1, eps_abs=1e-30):
    """Trace-wise quadratic Wasserstein distance: sum over traces of W2^2 in s^2.

    Positivity: linear shift by c = shift * max|obs| of the shot, then per-trace normalization
    to unit mass on the sample times t_k = k dt. W2^2 is the exact discrete optimal-transport
    cost between the two sets of point masses, from their quantile functions (as in POT's
    wasserstein_1d): W2^2 = sum over the merged CDF breakpoints of du * (F^-1(u) - G^-1(u))^2.
    Samples of syn below -c are clamped (zero gradient there); c is chosen from obs so this
    is rare.
    """
    c = (shift * obs.detach().abs().max()).clamp_min(eps_abs)

    def mass(x):
        m = (x + c).clamp_min(1e-6 * c)
        return m / m.sum(-1, keepdim=True)

    nt = syn.shape[-1]
    t = torch.arange(nt, dtype=syn.dtype, device=syn.device) * dt
    F, G = mass(syn).cumsum(-1), mass(obs).cumsum(-1).detach()
    qs, _ = torch.sort(torch.cat([F, G], -1), -1)                  # merged CDF breakpoints
    du = torch.diff(qs, dim=-1, prepend=torch.zeros_like(qs[..., :1]))
    quantile = lambda cdf: t[torch.searchsorted(cdf.detach().contiguous(), qs.detach().contiguous()).clamp(max=nt - 1)]
    return torch.sum(du * (quantile(F) - quantile(G)) ** 2)


def lowpass_l2(syn, obs, dt, fc=None, order=4, pad=True):
    """L2 after a zero-phase low-pass at fc Hz (None = full band), for frequency continuation.

    The filter is the analog Butterworth magnitude 1/sqrt(1 + (f/fc)^(2 order)) sampled on the
    FFT grid (applied once, zero phase; not scipy's butter + filtfilt, which squares the
    magnitude). pad=True zero-pads to twice the length before filtering, which reduces
    circular wraparound of early energy to late times. Autograd gives the adjoint L^T L r.
    """
    if fc is None:
        return l2(syn, obs)
    nt = syn.shape[-1]
    n = 2 * nt if pad else nt
    freqs = torch.fft.rfftfreq(n, d=dt, dtype=syn.dtype, device=syn.device)
    h = 1.0 / torch.sqrt(1.0 + (freqs / fc) ** (2 * order))
    lp = lambda x: torch.fft.irfft(torch.fft.rfft(x, n=n, dim=-1) * h, n=n, dim=-1)[..., :nt]
    return l2(lp(syn), lp(obs))


def add_noise(d, snr_db, dt, kind="flat", band=(1.0, 11.0), seed=0, premute=None):
    """Add noise to shots ``d`` (ns, nt, nr) at ``snr_db``, defined per shot on unmasked data.

    kind="white": Gaussian noise, independent per sample, flat up to Nyquist (``band`` is ignored).
    kind="flat": white noise with every frequency outside ``band`` removed (sharp edges, flat
    inside), so all of it lies in the data's band without the data's spectral shape.
    SNR = 10 log10(mean(d^2) / mean(n^2)) over every sample of each shot. Returns float64.
    premute: if set, the noise is zeroed before each trace's first arrival, picked as the first sample
    where |d| exceeds ``premute`` times the trace's max. The noise is scaled before the mute, so after the
    pick it equals the unmuted noise (the whole-record SNR is then higher than ``snr_db``).
    """
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty_like(d)
    for i, shot in enumerate(d):
        w = rng.standard_normal(shot.shape)
        if kind == "white":
            n = w
        elif kind == "flat":
            f = np.fft.rfftfreq(shot.shape[0], dt)[:, None]
            n = np.fft.irfft(np.fft.rfft(w, axis=0) * ((f >= band[0]) & (f <= band[1])), n=shot.shape[0], axis=0)
        else:
            raise ValueError(f"unknown noise kind '{kind}'")
        ps = np.mean(shot ** 2)
        if ps == 0:                     # SNR undefined for a silent shot
            out[i] = shot
            continue
        n *= np.sqrt(ps / np.mean(n ** 2)) * 10.0 ** (-snr_db / 20.0)
        if premute is not None:
            # Hard mute at an amplitude pick of the first arrival.
            a = np.abs(shot)
            pick = (a > premute * a.max(axis=0, keepdims=True)).argmax(axis=0)
            n[np.arange(shot.shape[0])[:, None] < pick[None, :]] = 0.0
        out[i] = shot + n
    return out
