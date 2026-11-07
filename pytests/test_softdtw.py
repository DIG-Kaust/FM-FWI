"""The soft-DTW baseline path (devitofwi SoftDTW -> tslearn SoftDTWLossPyTorch) against an exact float64 reference.

tslearn 0.9.0's PyTorch gradient is wrong (unpadded D in the backward, R cast to float32); fwi2d.py patches it with
fm_fwi.baselines.fix_tslearn_softdtw(). The reference is a plain-torch float64 soft-DTW divergence differentiated by
autograd. On the real data the patched gradient also matches float64 finite differences (scripts/diag_softdtw_fd.py).
Needs tslearn:  python -m pytest -q -s pytests/test_softdtw.py
"""

import unittest

import numpy as np
import torch

from fm_fwi import baselines as bl

try:
    from devitofwi.loss.dtw import SoftDTW
    from pylops.basicoperators import Diagonal
except ImportError:
    SoftDTW = None


def _ricker(nt, dt, t0, f0=5.0):
    t = np.arange(nt) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _sdtw(x, y, gamma):
    """Soft-DTW of 1D series x, y (torch float64), squared-Euclidean cost, by the plain recursion."""
    D = (x[:, None] - y[None, :]) ** 2
    m, n = D.shape
    inf = torch.tensor(float("inf"), dtype=x.dtype)
    R = [[inf] * (n + 1) for _ in range(m + 1)]
    R[0][0] = torch.zeros((), dtype=x.dtype)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            r = torch.stack([R[i - 1][j - 1], R[i - 1][j], R[i][j - 1]])
            R[i][j] = D[i - 1, j - 1] - gamma * torch.logsumexp(-r / gamma, 0)
    return R[m][n]


def _divergence(x, y, gamma):
    return _sdtw(x, y, gamma) - 0.5 * (_sdtw(x, x, gamma) + _sdtw(y, y, gamma))


@unittest.skipIf(SoftDTW is None, "tslearn / devitofwi not available")
class TestSoftDTW(unittest.TestCase):
    def test_patched_against_float64_reference(self):
        bl.fix_tslearn_softdtw()
        nt, nr, dt = 64, 3, 0.008
        rng = np.random.default_rng(0)
        obs = np.stack([_ricker(nt, dt, 0.15 + 0.03 * k) for k in range(nr)]).T          # (nt, nr)
        syn = np.roll(obs, 3, axis=0) + 0.01 * rng.standard_normal((nt, nr))
        mask = np.clip(np.linspace(1.5, -0.5, nt), 0, 1)[:, None] * np.ones((1, nr))    # as fwi2d: Op = mask
        b = (mask * obs).reshape(1, -1)
        x = syn.ravel()
        for t_sub in (1, 4):
            for gamma in (1.0, 10.0, 100.0):
                loss = SoftDTW(Op=[Diagonal(mask.ravel(), dtype="float64")], b=b, nr=nr, nt=nt, gamma=gamma,
                               normalize=True, t_sub=t_sub)
                f = loss(x, 0)
                g = np.asarray(loss.grad(x, 0), dtype=np.float64).ravel()
                xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                s = (torch.tensor(mask.ravel()) * xt).reshape(nt, nr).T[:, ::t_sub]
                o = torch.tensor(b[0]).reshape(nt, nr).T[:, ::t_sub]
                fr = torch.stack([_divergence(s[k], o[k], gamma) for k in range(nr)]).mean()
                (gr,) = torch.autograd.grad(fr, xt)
                rel = np.linalg.norm(g - gr.numpy()) / np.linalg.norm(gr.numpy())
                msg = f"t_sub {t_sub}, gamma {gamma}: loss {f} vs {float(fr)}, grad rel diff {rel:.2e}"
                print(msg)
                self.assertAlmostEqual(f, float(fr), delta=1e-10 * max(1.0, abs(float(fr))), msg=msg)
                self.assertLess(rel, 1e-8, msg=msg)


if __name__ == "__main__":
    unittest.main()
