"""Time-shift diagnostics for the revision (Fig 3 / C6, Fig 4 style, baseline tuning).

twoshift: 1D trace of two 3 Hz Ricker wavelets; the modelled trace shifts each event
    independently over +-2.5 dominant periods (paper Fig 3). Misfit surfaces for L2,
    SeisLM (pretrained / random, full sequence / mean pooled) and the baselines.
ricker: ADFWI's Ricker-shift misfit test (one 6 Hz Ricker, circular shifts of +-0.3 s), our misfits and SeisLM.
sampling: the rigid-shift test on the same masked shot resampled to 1, 2, 4, 8 and 10 ms (R3.m3).
    Pretrained encoder, full sequence vs mean pooled, every layer.
window: the rigid-shift test on the same masked shot zero-padded from its 4 s record to 6 and 8 s
    (R4.5). Masked data are zero after the mute, so padding is exactly a longer masked record.
    Pretrained encoder, full sequence vs mean pooled, every layer.
rigid: one masked 2D Overthrust shot gather shifted rigidly in time over +-2.5 periods.
    Curves for every objective, every encoder layer (pretrained / random, pooled / full),
    and the baseline hyperparameter sweeps (soft-DTW gamma, W2 shift, low-pass cut-off).

Each curve is saved unnormalized in <out>/<mode>.npz; plots normalize by the maximum.

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
from fm_fwi import baselines as bl  # noqa: E402
from fm_fwi.losses import SeisLMFeatureLoss  # noqa: E402
from fm_fwi.model.pretrained import (  # noqa: E402
    load_pretrained_multidim_wav2vec2, load_random_multidim_wav2vec2)

CKPT = os.path.join(REPO, "pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def shift(x, s):
    """Shift traces x [..., nt] by s samples with zero fill."""
    out = np.zeros_like(x)
    if s >= 0:
        out[..., s:] = x[..., :x.shape[-1] - s]
    else:
        out[..., :s] = x[..., -s:]
    return out


class Encoder:
    """All hidden states of a frozen SeisLM for traces [B, nt], with the loss's preprocessing."""

    def __init__(self, kind, seed=0, chunk=256):
        model = (load_pretrained_multidim_wav2vec2(CKPT, map_location=DEVICE) if kind == "pretrained"
                 else load_random_multidim_wav2vec2(CKPT, map_location=DEVICE, seed=seed))
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


def baseline_curves(syn_list, obs, dt):
    """Baseline misfits (and hyperparameter sweeps) for each syn against obs [nr, nt]."""
    o = torch.as_tensor(obs, dtype=torch.float64)
    scale = 1.0 / (np.sqrt((obs ** 2).mean()) + 1e-30)  # soft-DTW on ~unit-RMS data, as in the driver
    from tslearn.metrics import SoftDTWLossPyTorch
    fns = {"l2": bl.l2, "stdl2": bl.standardized_l2, "envelope": bl.envelope_l2}
    for c in (0.5, 1.1, 2.0):
        fns[f"w2_shift{c}"] = lambda s, o, c=c: bl.wasserstein2(s, o, dt=dt, shift=c)
    for fc in (2.0, 3.0, 4.0):
        fns[f"lowpass_{fc:g}Hz"] = lambda s, o, fc=fc: bl.lowpass_l2(s, o, dt, fc)
    for g in (1.0, 10.0, 100.0, 1000.0):
        sdtw = SoftDTWLossPyTorch(gamma=g, normalize=True)
        fns[f"softdtw_g{g:g}"] = lambda s, o, sdtw=sdtw: sdtw(
            (scale * s)[:, ::4, None].to(DEVICE), (scale * o)[:, ::4, None].to(DEVICE)).mean()
    out = {k: [] for k in fns}
    for syn in syn_list:
        s = torch.as_tensor(syn, dtype=torch.float64)
        for k, f in fns.items():
            out[k].append(float(f(s, o)))
    return {k: np.array(v) for k, v in out.items()}


def twoshift(args):
    dt, nt, f0 = 0.004, 1000, 3.0
    t = np.arange(nt) * dt
    ricker = lambda t0: (1 - 2 * (np.pi * f0 * (t - t0)) ** 2) * np.exp(-(np.pi * f0 * (t - t0)) ** 2)
    t1, t2 = 1.5, 2.5
    obs = (ricker(t1) + ricker(t2))[None]
    period = 1.0 / f0
    shifts = np.linspace(-2.5 * period, 2.5 * period, args.n)
    syn = np.stack([ricker(t1 + a) + ricker(t2 + b) for a in shifts for b in shifts])
    res = {"shifts_s": shifts, "t1": t1, "t2": t2, "f0": f0, "dt": dt}
    for kind, seed in (("pretrained", 0), ("random", 0)):
        enc = Encoder(kind, seed)
        zo = enc(obs)
        zm = enc(syn)
        res[f"{kind}_full"] = (((zm[-1] - zo[-1]) ** 2).mean((1, 2))).numpy().reshape(args.n, args.n)
        res[f"{kind}_mean"] = (((zm[-1].mean(1) - zo[-1].mean(1)) ** 2).mean(1)).numpy().reshape(args.n, args.n)
    base = baseline_curves(list(syn[:, None, :]), obs, dt)
    res.update({k: v.reshape(args.n, args.n) for k, v in base.items()})
    return res


def rigid(args):
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from fwi2d import build_problem
    p = build_problem(SimpleNamespace(acq="surface", init="linear", init_sigma=[50.0, 40.0], init_scale=1.0,
                                      noise_snr=args.noise_snr, noise_type="band", noise_band=None,
                                      noise_seed=0, toff=0.4))
    obs = p["dobs_masked"][p["par"]["ns"] // 2].T.astype(np.float64)  # [nr, nt]
    dt, period = p["dt"], 1.0 / p["par"]["freq"]
    smax = int(round(2.5 * period / dt))
    samples = np.arange(-smax, smax + 1, max(1, smax // 40))
    mask = p["masks"][p["par"]["ns"] // 2].T
    # rigid shift of the masked gather, re-masked as in inversion (the modelled data are masked)
    syn_list = [shift(obs, s) * mask for s in samples]
    res = {"shifts_s": samples * dt, "period_s": period}
    for kind in ("pretrained", "random"):
        full, mean = feature_curves(Encoder(kind), syn_list, obs)
        res[f"{kind}_full_layers"], res[f"{kind}_mean_layers"] = full, mean
    res.update(baseline_curves(syn_list, obs, dt))
    return res


def ricker(args):
    """ADFWI's Ricker-shift test (examples/acoustic/02-misfit-functions-test/00-Ricker-Test/01_misfit_wavelets-shift):
    a 6 Hz Ricker (nt 1000, dt 3 ms, t0 0.5 s rolled by 330 samples, peak at 1.49 s) against copies of itself rolled
    circularly by -100..99 samples, through our misfits and SeisLM."""
    nt, dt, f0 = 1000, 0.003, 6.0
    t = np.arange(nt) * dt
    a = (np.pi * f0 * (t - 0.5)) ** 2
    obs = np.roll((1 - 2 * a) * np.exp(-a), 330)[None]                       # [1, nt]
    samples = np.arange(-100, 100)
    syn_list = [np.roll(obs, k, axis=-1) for k in samples]
    o = torch.as_tensor(obs, dtype=torch.float64)
    S = [torch.as_tensor(x, dtype=torch.float64) for x in syn_list]
    res = {"shifts_s": samples * dt, "period_s": 1.0 / f0, "dt": dt, "obs": obs[0],
           "l1": np.array([float((x - o).abs().sum()) for x in S]),             # ADFWI L1 / SmoothL1, as torch one-liners
           "smoothl1": np.array([float(torch.nn.functional.smooth_l1_loss(x, o, reduction="sum", beta=1.0)) for x in S]),
           "envelope_p1": np.array([float(bl.envelope_l2(x, o, p=1.0)) for x in S])}
    from tslearn.metrics import SoftDTWLossPyTorch
    scale = 1.0 / np.sqrt((obs ** 2).mean())
    for g in (5.0, 30.0):                                                     # t_sub 2 as in ADFWI's example
        sdtw = SoftDTWLossPyTorch(gamma=g, normalize=True)
        res[f"softdtw_tsub2_g{g:g}"] = np.array([float(sdtw((scale * x)[:, ::2, None], (scale * o)[:, ::2, None]).mean()) for x in S])
    res.update(baseline_curves(syn_list, obs, dt))
    for kind in ("pretrained", "random"):
        full, mean = feature_curves(Encoder(kind), syn_list, obs)
        res[f"{kind}_full_layers"], res[f"{kind}_mean_layers"] = full, mean
    return res


def _problem():
    """Clean 2D Overthrust testbed from the driver (linear start; the start does not enter these tests)."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from fwi2d import build_problem
    return build_problem(SimpleNamespace(acq="surface", init="linear", init_sigma=[50.0, 40.0], init_scale=1.0,
                                         noise_snr=None, noise_type="band", noise_band=None, noise_seed=0, toff=0.4))


def window(args):
    p = _problem()
    obs = p["dobs_masked"][p["par"]["ns"] // 2].T.astype(np.float64)  # [nr, nt]
    dt, period = p["dt"], 1.0 / p["par"]["freq"]
    smax = int(round(2.5 * period / dt))
    samples = np.arange(-smax, smax + 1, max(1, smax // 40))
    mask = p["masks"][p["par"]["ns"] // 2].T
    syn_list = [shift(obs, s) * mask for s in samples]
    res = {"shifts_s": samples * dt, "period_s": period, "windows_s": np.array(args.windows)}
    enc = Encoder("pretrained")
    for T in args.windows:
        pad = lambda x: np.pad(x, ((0, 0), (0, max(0, int(round(T / dt)) - x.shape[1]))))
        full, mean = feature_curves(enc, [pad(x) for x in syn_list], pad(obs))
        res[f"full_{T:g}s"], res[f"mean_{T:g}s"] = full, mean
    return res


def sampling(args):
    p = _problem()
    obs = p["dobs_masked"][p["par"]["ns"] // 2].T.astype(np.float64)  # [nr, nt]
    dt, period = p["dt"], 1.0 / p["par"]["freq"]
    smax = int(round(2.5 * period / dt))
    samples = np.arange(-smax, smax + 1, max(1, smax // 40))
    mask = p["masks"][p["par"]["ns"] // 2].T
    # shift and mask at the modelling interval, then resample; the data have no energy near the
    # coarsest Nyquist (50 Hz at 10 ms), so plain interpolation needs no anti-alias filter
    syn_list = [shift(obs, s) * mask for s in samples]
    t = np.arange(obs.shape[1]) * dt
    res = {"shifts_s": samples * dt, "period_s": period, "dts_s": np.array(args.dts), "dt_model_s": dt}
    enc = Encoder("pretrained")
    for d in args.dts:
        tn = np.arange(0.0, t[-1] + 1e-9, d)
        rs = lambda x: np.stack([np.interp(tn, t, tr) for tr in x])
        full, mean = feature_curves(enc, [rs(x) for x in syn_list], rs(obs))
        res[f"full_{d * 1e3:g}ms"], res[f"mean_{d * 1e3:g}ms"] = full, mean
    return res


def plot(res, mode, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = lambda v: (v - v.min()) / max(v.max() - v.min(), 1e-30)
    if mode in ("window", "sampling"):
        # window: unnormalized (same sequence length); sampling: each curve normalized, since the
        # sequence length and so the distance scale change with the interval
        vals, unit, lab, f = ((res["windows_s"], "s", "{:g} s record", lambda v: v) if mode == "window" else
                              (res["dts_s"] * 1e3, "ms", "{:g} ms sampling", norm))
        tp = res["shifts_s"] / res["period_s"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for i, T in enumerate(vals):
            c = plt.cm.viridis(i / max(1, len(vals) - 1))
            for ax, k in zip(axs, ("full", "mean")):
                ax.plot(tp, f(res[f"{k}_{T:g}{unit}"][-1]), color=c, label=lab.format(T))
        state = "unnormalized" if mode == "window" else "normalized"
        axs[0].set_title(f"SeisLM full sequence, last layer ({state})")
        axs[1].set_title(f"SeisLM mean pooled, last layer ({state})")
        axs[1].legend()
        for ax in axs:
            ax.set_xlabel("shift (dominant periods)")
    elif mode == "twoshift":
        keys = [k for k, v in res.items() if isinstance(v, np.ndarray) and v.ndim == 2]
        n = len(keys)
        fig, axs = plt.subplots((n + 4) // 5, 5, figsize=(16, 3.2 * ((n + 4) // 5)))
        ext = [res["shifts_s"][0], res["shifts_s"][-1]] * 2
        for ax, k in zip(axs.flat, keys):
            ax.imshow(norm(res[k]).T, origin="lower", extent=ext, cmap="viridis")
            ax.set_title(k, fontsize=8)
        for ax in list(axs.flat)[n:]:
            ax.axis("off")
    elif mode == "ricker":
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in ("l1", "l2", "smoothl1", "envelope_p1", "softdtw_tsub2_g5", "w2_shift1.1", "lowpass_2Hz"):
            ax.plot(res["shifts_s"], norm(res[k]), lw=1, label=k)
        ax.plot(res["shifts_s"], norm(res["pretrained_full_layers"][-1]), "k", lw=2, label="SeisLM")
        ax.legend(fontsize=7); ax.set_xlabel("time shift (s)")
    else:
        tp = res["shifts_s"] / res["period_s"]
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        for kind, ls in (("pretrained", "-"), ("random", "--")):
            for j, row in enumerate(res[f"{kind}_full_layers"]):
                axs[0].plot(tp, norm(row), ls, color=plt.cm.viridis(j / len(res[f"{kind}_full_layers"])))
            for j, row in enumerate(res[f"{kind}_mean_layers"]):
                axs[1].plot(tp, norm(row), ls, color=plt.cm.viridis(j / len(res[f"{kind}_mean_layers"])))
        axs[0].set_title("SeisLM full sequence, per layer (solid pretrained, dashed random)")
        axs[1].set_title("SeisLM mean pooled, per layer")
        for k, v in res.items():
            if isinstance(v, np.ndarray) and v.ndim == 1 and k not in ("shifts_s",):
                axs[2].plot(tp, norm(v), label=k, lw=1)
        axs[2].legend(fontsize=6, ncol=2)
        axs[2].set_title("Baselines")
        for ax in axs:
            ax.set_xlabel("shift (dominant periods)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["twoshift", "rigid", "window", "sampling", "ricker"])
    ap.add_argument("--out", default=os.path.join(REPO, "runs/diagnostics"))
    ap.add_argument("--n", type=int, default=41, help="twoshift grid size per axis")
    ap.add_argument("--noise-snr", type=float, default=None, help="rigid: add band noise to the data")
    ap.add_argument("--windows", type=float, nargs="+", default=[4.0, 6.0, 8.0], help="window: record lengths (s)")
    ap.add_argument("--dts", type=float, nargs="+", default=[0.001, 0.002, 0.004, 0.008, 0.010],
                    help="sampling: intervals (s) fed to SeisLM; 0.010 = the 100 Hz of pretraining")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.mode + (f"_snr{args.noise_snr:g}" if args.noise_snr is not None else "")
    res = {"twoshift": twoshift, "rigid": rigid, "window": window, "sampling": sampling, "ricker": ricker}[args.mode](args)
    np.savez(os.path.join(args.out, f"{tag}.npz"), **res)
    plot(res, args.mode, os.path.join(args.out, f"{tag}.png"))
    print("saved", os.path.join(args.out, tag))


if __name__ == "__main__":
    main()
