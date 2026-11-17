"""1D two-Ricker time-shift experiment of the paper (Fig 3).

Setup:
3 Hz Ricker wavelets at 1/3 and 2/3 of a 4107-sample trace, dt 3 ms, each shifted independently
over +-2.5 dominant periods in steps of 4 samples (137 x 137 grid), three-channel copy of the trace.

Adds, on the same traces:
  - every encoder layer, full sequence and mean pooled
  - random encoders with 3 seeds
  - two input preparations: "raw" (as in the paper's Fig 3) and "std" (per-trace standardization,
    as in the inversion loss)
  - the baselines with the hyperparameters used in the inversions (and the sweeps)
  - a rigid shift of the whole two-event trace (Fig 4 style, 1D version)

All values are the objective as minimized (squared distances; mean over feature entries for SeisLM).

    python scripts/boa1d.py --out runs/diagnostics/boa1d          # compute (GPU) + plot
    python scripts/boa1d.py --out runs/diagnostics/boa1d --plot   # plot from boa1d.npz only
"""

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt")
F0, DT, NT, STRIDE = 3.0, 0.003, 4107, 4
C1, C2 = NT // 3, 2 * NT // 3
SEEDS = (0, 1, 2)


def ricker(center):
    u = (np.pi * F0 * (np.arange(NT) - center) * DT) ** 2
    return (1 - 2 * u) * np.exp(-u)


def shift(x, s):
    out = np.zeros_like(x)
    if s >= 0:
        out[..., s:] = x[..., :x.shape[-1] - s]
    else:
        out[..., :s] = x[..., -s:]
    return out


def traces(stride=STRIDE):
    w1, w2 = ricker(C1), ricker(C2)
    ref = w1 + w2
    smax = int((2.5 / F0) / DT) // stride * stride  # symmetric grid that contains zero shift
    sh = np.arange(-smax, smax + 1, stride)
    grid = (np.stack([shift(w1, s) for s in sh])[:, None] + np.stack([shift(w2, s) for s in sh])[None])
    rigid = np.stack([shift(ref, s) for s in sh])
    return ref, sh, grid.reshape(-1, NT), rigid


def compute(args):
    import torch
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from fm_fwi.model.pretrained import load_pretrained_multidim_wav2vec2, load_random_multidim_wav2vec2
    from shift_diagnostics import baseline_curves
    dev = "cuda"
    ref, sh, grid, rigid = traces(args.stride)
    t0 = time.time()
    print(f"{len(sh)} shifts, {len(grid)} grid traces", flush=True)
    res = {"shifts_s": sh * DT, "ref": ref, "w1": ricker(C1), "w2": ricker(C2),
           "f0": F0, "dt": DT}

    def prep(x, kind):
        x = torch.as_tensor(x, dtype=torch.float32, device=dev)[:, None].repeat(1, 3, 1)
        if kind == "std":
            x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-10)
        return x

    @torch.no_grad()
    def dists(model, x, kind, chunk=64):
        """Per-layer [full, mean-pooled] squared distances of every trace in x to ref: [L, 2, N]."""
        enc = lambda t: model.wav2vec2(prep(t, kind), output_hidden_states=True).hidden_states
        zr = enc(ref[None])
        out = []
        for s in range(0, len(x), chunk):
            z = enc(x[s:s + chunk])
            out.append(torch.stack([torch.stack([((a - b) ** 2).mean((1, 2)), ((a.mean(1) - b.mean(1)) ** 2).mean(1)])
                                    for a, b in zip(z, zr)]).cpu())
        return torch.cat(out, -1).numpy()

    models = [("pretrained", lambda: load_pretrained_multidim_wav2vec2(CKPT, map_location=dev))]
    models += [(f"random{s}", lambda s=s: load_random_multidim_wav2vec2(CKPT, map_location=dev, seed=s)) for s in SEEDS]
    for name, load in models:
        m = load().to(dev).eval()
        for kind in ("raw", "std"):
            res[f"{name}_{kind}_grid"] = dists(m, grid, kind)    # [L, 2, N*N]
            res[f"{name}_{kind}_rigid"] = dists(m, rigid, kind)  # [L, 2, N]
            print(name, kind, f"done {time.time() - t0:.0f} s", flush=True)
        del m
        torch.cuda.empty_cache()
    for tag, x in (("grid", grid), ("rigid", rigid)):
        res[f"bl_l2_{tag}"] = 0.5 * ((x - ref) ** 2).sum(-1)  # = bl.l2 per trace, vectorized
    # Save before the per-trace CPU baselines: they keep the GPU idle for hours (job 347788 was cancelled
    # in that phase), so they only run with --baselines all.
    np.savez(os.path.join(args.out, "boa1d.npz"), **res)
    print(f"saved encoder maps {time.time() - t0:.0f} s", flush=True)
    if args.baselines == "all":
        for tag, x in (("grid", grid), ("rigid", rigid)):
            for k, v in baseline_curves(list(x[:, None, :]), ref[None], DT).items():
                res[f"bl_{k}_{tag}"] = v
            print("baselines", tag, f"done {time.time() - t0:.0f} s", flush=True)
    return res


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    n = len(res["shifts_s"])
    ms = res["shifts_s"] * 1e3
    T0 = 1e3 / float(res["f0"])
    norm = lambda v: (v - v.min()) / max(v.max() - v.min(), 1e-30)
    last = lambda k, pool=0: res[k][-1, pool]
    g = lambda v: norm(v).reshape(n, n)

    # Fig 3, revised: traces, then maps for the inversion objectives
    panels = [("L2", res["bl_l2_grid"]),
              ("SeisLM, full sequence", last("pretrained_std_grid")),
              ("Random encoder, full sequence", last("random0_std_grid")),
              ("SeisLM, mean pooled", last("pretrained_std_grid", 1)),
              ("Envelope", res["bl_envelope_grid"]),
              ("Soft-DTW, γ = 100", res["bl_softdtw_g100_grid"]),
              ("W2", res["bl_w2_shift1.1_grid"]),
              ("Standardized L2", res["bl_stdl2_grid"])]
    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.6, 1, 1], hspace=0.35, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    t = np.arange(NT) * DT * 1e3
    ax.plot(t, res["ref"], "k", lw=1.4, label="reference")
    i0 = n // 2
    for (a, b), c in (((i0 + 14, i0), "C3"), ((i0, i0 - 14), "C2"), ((i0 + 42, i0 - 42), "C4")):
        sa, sb = int(round(ms[a] / (DT * 1e3))), int(round(ms[b] / (DT * 1e3)))
        ax.plot(t, shift(res["w1"], sa) + shift(res["w2"], sb), c, lw=1,
                label=f"Δt₁ = {ms[a]:+.0f} ms, Δt₂ = {ms[b]:+.0f} ms")
    ax.set(xlim=(3000, 9500), xlabel="Time (ms)", ylabel="Amplitude")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    for i, (title, v) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        im = ax.imshow(g(v).T, origin="lower", extent=[ms[0], ms[-1]] * 2, cmap="RdBu_r", vmin=0, vmax=1)
        ax.plot(0, 0, "w+", ms=7)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([-800, 0, 800]); ax.set_yticks([-800, 0, 800])
        if i % 4 == 0: ax.set_ylabel("Δt₂ (ms)")
        else: ax.set_yticklabels([])
        if i // 4 == 1: ax.set_xlabel("Δt₁ (ms)")
    fig.colorbar(im, ax=fig.axes[1:], shrink=0.6, label="Normalized misfit")
    fig.savefig(os.path.join(out, "boa1d_maps.png"), dpi=200, bbox_inches="tight")

    # Paper Fig 3 inputs (raw traces) next to the inversion's standardized inputs
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.6), constrained_layout=True)
    for r, kind in enumerate(("raw", "std")):
        for c, (title, k, p) in enumerate((("SeisLM, full", "pretrained", 0), ("Random 0, full", "random0", 0),
                                           ("Random 1, full", "random1", 0), ("SeisLM, mean pooled", "pretrained", 1))):
            ax = axs[r, c]
            ax.imshow(g(last(f"{k}_{kind}_grid", p)).T, origin="lower", extent=[ms[0], ms[-1]] * 2,
                      cmap="RdBu_r", vmin=0, vmax=1)
            ax.set_title(f"{title} ({'raw input, as Fig 3' if kind == 'raw' else 'standardized input'})", fontsize=8)
            ax.set_xticks([-800, 0, 800]); ax.set_yticks([-800, 0, 800])
    fig.savefig(os.path.join(out, "boa1d_inputs.png"), dpi=200)

    # Fig 4 style on the 1D trace: rigid shift, per layer
    tp = res["shifts_s"] * float(res["f0"])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, constrained_layout=True)
    nl = res["pretrained_std_rigid"].shape[0]
    cm = plt.cm.viridis
    for ax, (title, keys, pool) in zip(axs, (("(a) Pretrained, full sequence", ["pretrained"], 0),
                                             ("(b) Random (3 seeds), full sequence", [f"random{s}" for s in SEEDS], 0),
                                             ("(c) Pretrained, mean pooled", ["pretrained"], 1))):
        for k in keys:
            for j in range(nl):
                ax.plot(tp, norm(res[f"{k}_std_rigid"][j, pool]), color=cm(j / (nl - 1)), lw=1,
                        label=("conv. projection" if j == 0 else f"block {j}") if k == keys[0] else None)
        ax.plot(tp, norm(res["bl_l2_rigid"]), "k:", lw=1.2, label="L2 (data)")
        ax.set(title=title, xlabel="Shift (dominant periods)")
    axs[0].set_ylabel("Normalized distance")
    axs[0].legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(out, "boa1d_layers.png"), dpi=200)

    # Cross-section at Δt₂ = 0 for every objective, with the monotonic interval around zero
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    rows = {"L2": res["bl_l2_grid"], "SeisLM": last("pretrained_std_grid"),
            "Random 0": last("random0_std_grid"), "SeisLM pooled": last("pretrained_std_grid", 1),
            "Envelope": res["bl_envelope_grid"], "Soft-DTW γ100": res["bl_softdtw_g100_grid"],
            "W2": res["bl_w2_shift1.1_grid"], "Std L2": res["bl_stdl2_grid"]}
    for i, (k, v) in enumerate(rows.items()):
        y = v.reshape(n, n)[:, n // 2]
        lo, hi = monotonic(y)
        ax.plot(ms / T0, norm(y), color=f"C{i}", lw=1.2, label=f"{k}  [{ms[lo] / T0:+.2f}, {ms[hi] / T0:+.2f}]")
    ax.set(xlabel="Δt₁ (dominant periods), Δt₂ = 0", ylabel="Normalized misfit")
    ax.legend(fontsize=7, frameon=False, title="monotonic interval", title_fontsize=7)
    fig.savefig(os.path.join(out, "boa1d_section.png"), dpi=200)


def monotonic(y):
    """Indices of the widest interval around the minimum where y rises monotonically on both sides."""
    i = lo = hi = int(np.argmin(y))
    while lo > 0 and y[lo - 1] >= y[lo]: lo -= 1
    while hi < len(y) - 1 and y[hi + 1] >= y[hi]: hi += 1
    return lo, hi


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "runs/diagnostics/boa1d"))
    ap.add_argument("--plot", action="store_true", help="only plot from <out>/boa1d.npz")
    ap.add_argument("--no-plot", action="store_true", help="compute and save only")
    ap.add_argument("--stride", type=int, default=STRIDE, help="shift step in samples (coarser for a smoke test)")
    ap.add_argument("--baselines", choices=["l2", "all"], default="all",
                    help="l2: vectorized L2 only; all: also the per-trace baseline sweeps (CPU, slow)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.plot:
        res = dict(np.load(os.path.join(args.out, "boa1d.npz")))
    else:
        res = compute(args)
        np.savez(os.path.join(args.out, "boa1d.npz"), **res)
    if not args.no_plot:
        plot(res, args.out)
    print("saved", args.out)
