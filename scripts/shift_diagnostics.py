"""Time-shift diagnostics (Figs 12a and 18a, SI Fig S4), pretrained SeisLM encoder.

ricker: ADFWI's Ricker-shift misfit test (one 6 Hz Ricker, circular shifts of +-0.3 s), SeisLM curve of Fig 12a
    (fig12_baselines.py computes the other misfits itself).
rigid: the masked middle 2D Overthrust shot gather shifted rigidly in time over +-2.5 periods (Fig 18a).
sampling: the same test with the shot resampled to 1, 2, 4, 8 and 10 ms (Fig S4a).
window: the same test with the shot zero-padded from its 4 s record to 6 and 8 s (Fig S4b).
    Masked data are zero after the mute, so padding is exactly a longer masked record.

Each mode saves, per encoder layer, the unnormalized full-sequence and time-averaged feature distances in
<out>/<mode>.npz.

    python scripts/shift_diagnostics.py rigid --out runs/diagnostics
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from fm_fwi.losses import SeisLMFeatureLoss  # noqa: E402
from fm_fwi.model.pretrained import load_pretrained_multidim_wav2vec2  # noqa: E402

CKPT = os.path.join(REPO, "pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOWS = [4.0, 6.0, 8.0]                     # window: record lengths (s)
DTS = [0.001, 0.002, 0.004, 0.008, 0.010]     # sampling: intervals (s); 0.010 = the 100 Hz of pretraining


def shift(x, s):
    """Shift traces x [..., nt] by s samples with zero fill."""
    out = np.zeros_like(x)
    if s >= 0:
        out[..., s:] = x[..., :x.shape[-1] - s]
    else:
        out[..., :s] = x[..., -s:]
    return out


class Encoder:
    """All hidden states of the frozen pretrained SeisLM for traces [B, nt], with the loss's preprocessing."""

    def __init__(self, chunk=256):
        model = load_pretrained_multidim_wav2vec2(CKPT, map_location=DEVICE)
        # Reuse SeisLMFeatureLoss's standardization, channel copy and encoder call; the
        # depth comes from one forward pass.
        model = model.to(DEVICE).eval()
        with torch.no_grad():
            nl = len(model.wav2vec2(torch.zeros(1, 3, 400, device=DEVICE), output_hidden_states=True).hidden_states) - 1
        self.loss = SeisLMFeatureLoss(model, Op=None, b=None, nr=1, nt=1, device=DEVICE,
                                      pool="none", feature_layers=list(range(nl)))
        self.chunk = chunk

    @torch.no_grad()
    def __call__(self, traces):
        zs = None
        for s in range(0, len(traces), self.chunk):
            x = self.loss._prepare_batch_3c(torch.as_tensor(traces[s:s + self.chunk], dtype=torch.float32))
            z = [h.float().cpu() for h in self.loss._encode(x)]
            zs = [[h] for h in z] if zs is None else [a + [h] for a, h in zip(zs, z)]
        return [torch.cat(a) for a in zs]  # per layer [B, T', D]


def feature_curves(enc, syn_list, obs):
    """Per-layer MSE (full sequence and mean pooled) of each syn in syn_list against obs [nr, nt]."""
    zo = enc(obs)
    full, mean = [], []
    for syn in syn_list:
        zm = enc(syn)
        full.append([float(((a - b) ** 2).mean()) for a, b in zip(zm, zo)])
        mean.append([float(((a.mean(1) - b.mean(1)) ** 2).mean()) for a, b in zip(zm, zo)])
    return np.array(full).T, np.array(mean).T  # [layer, n]


def overthrust_shot():
    """Masked middle shot gather of the 2D Overthrust example [nr, nt] and its rigid shifts over +-2.5 periods,
    re-masked as in the inversion (the modelled data are masked)."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from overthrust2d import build_problem
    p = build_problem(SimpleNamespace(acq="surface", init="linear", init_sigma=[50.0, 40.0], init_scale=1.0,
                                      noise_snr=None, noise_type="flat", noise_seed=0, toff=0.4))
    obs = p["dobs_masked"][p["par"]["ns"] // 2].T.astype(np.float64)
    dt, period = p["dt"], 1.0 / p["par"]["freq"]
    smax = int(round(2.5 * period / dt))
    samples = np.arange(-smax, smax + 1, max(1, smax // 40))
    mask = p["masks"][p["par"]["ns"] // 2].T
    syn_list = [shift(obs, s) * mask for s in samples]
    return obs, syn_list, dt, {"shifts_s": samples * dt, "period_s": period}


def rigid():
    obs, syn_list, dt, res = overthrust_shot()
    res["pretrained_full_layers"], res["pretrained_mean_layers"] = feature_curves(Encoder(), syn_list, obs)
    return res


def ricker():
    """ADFWI's Ricker-shift test (examples/acoustic/02-misfit-functions-test/00-Ricker-Test/01_misfit_wavelets-shift):
    a 6 Hz Ricker (nt 1000, dt 3 ms, t0 0.5 s rolled by 330 samples, peak at 1.49 s) against copies of itself rolled
    circularly by -100..99 samples."""
    nt, dt, f0 = 1000, 0.003, 6.0
    t = np.arange(nt) * dt
    a = (np.pi * f0 * (t - 0.5)) ** 2
    obs = np.roll((1 - 2 * a) * np.exp(-a), 330)[None]                       # [1, nt]
    samples = np.arange(-100, 100)
    syn_list = [np.roll(obs, k, axis=-1) for k in samples]
    res = {"shifts_s": samples * dt, "period_s": 1.0 / f0, "dt": dt, "obs": obs[0]}
    res["pretrained_full_layers"], res["pretrained_mean_layers"] = feature_curves(Encoder(), syn_list, obs)
    return res


def window():
    obs, syn_list, dt, res = overthrust_shot()
    res["windows_s"] = np.array(WINDOWS)
    enc = Encoder()
    for T in WINDOWS:
        pad = lambda x: np.pad(x, ((0, 0), (0, max(0, int(round(T / dt)) - x.shape[1]))))
        res[f"full_{T:g}s"], res[f"mean_{T:g}s"] = feature_curves(enc, [pad(x) for x in syn_list], pad(obs))
    return res


def sampling():
    # shift and mask at the modelling interval, then resample; the data have no energy near the
    # coarsest Nyquist (50 Hz at 10 ms), so plain interpolation needs no anti-alias filter
    obs, syn_list, dt, res = overthrust_shot()
    t = np.arange(obs.shape[1]) * dt
    res["dts_s"] = np.array(DTS)
    enc = Encoder()
    for d in DTS:
        tn = np.arange(0.0, t[-1] + 1e-9, d)
        rs = lambda x: np.stack([np.interp(tn, t, tr) for tr in x])
        res[f"full_{d * 1e3:g}ms"], res[f"mean_{d * 1e3:g}ms"] = feature_curves(enc, [rs(x) for x in syn_list], rs(obs))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["rigid", "window", "sampling", "ricker"])
    ap.add_argument("--out", default=os.path.join(REPO, "runs/diagnostics"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {"rigid": rigid, "window": window, "sampling": sampling, "ricker": ricker}[args.mode]()
    np.savez(os.path.join(args.out, f"{args.mode}.npz"), **res)
    print("saved", os.path.join(args.out, f"{args.mode}.npz"))


if __name__ == "__main__":
    main()
