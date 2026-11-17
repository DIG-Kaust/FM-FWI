# Paper figure: established misfits on the 2D Overthrust from the linear start, and SI Fig S5 (models).
# (a) ADFWI's Ricker-shift test (runs/diagnostics/ricker.npz: 6 Hz Ricker, circular shifts of +-0.3 s; SeisLM curve from the
#     shift_diagnostics.py ricker run) with each baseline at its inversion setting, recomputed here with fm_fwi.baselines; graph-space W1 exact.
# (b) Velocity error of the inversions (run.json of runs/<run>: S1 masked + S2 L2, accepted iterates).
# Usage: python fig12_baselines.py <outdir>
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(H, "../.."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md
from fm_fwi import baselines as bl  # noqa: E402

OUT = sys.argv[1]
# Base style of the paper's figure, then the panel style.
INK, SLATE, HAIR, CANVAS = "#1d1d1f", "#707070", "#d6d6d6", "#ffffff"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"], "font.size": 15,
                     "axes.edgecolor": HAIR, "axes.labelcolor": INK, "xtick.color": SLATE, "ytick.color": SLATE,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ededed", "grid.linewidth": 1, "lines.linewidth": 2.2,
                     "legend.frameon": False, "figure.facecolor": CANVAS, "axes.facecolor": CANVAS})
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
# label, colour, line style, inversion run
OBJ = [("L2", "#E8A317", "-", "ref_l2"), ("SeisLM", "#1B9E9E", "-", "ref_feature"),
       ("L2, standardized traces", "#7F7F7F", "--", "bl_stdl2"), ("Envelope", "#D62728", "-", "bl_envelope_p1"),
       ("W2", "#3B5BA5", "-", "bl_w2_exact"), ("Frequency continuation", "#2CA02C", "-", "bl_multiscale_pad"),
       ("Soft-DTW", "#8E44AD", "-", "sdtw_g30_linear"), ("Graph-space W1", "#8C564B", "-.", None)]


# Reference curves of panel (a) for the two misfits not in fm_fwi.baselines.
def sdtw_div(x, y, gamma):
    """Soft-DTW divergence (Blondel et al., 2021) of 1D series, squared-Euclidean cost, float64, anti-diagonal sweep."""
    def sdtw(a, b):
        D = (a[:, None] - b[None, :]) ** 2
        m, n = D.shape
        R = np.full((m + 1, n + 1), np.inf); R[0, 0] = 0.0
        for d in range(2, m + n + 1):
            i = np.arange(max(1, d - n), min(m, d - 1) + 1); j = d - i
            r = np.stack([R[i - 1, j - 1], R[i - 1, j], R[i, j - 1]]) / -gamma
            mx = r.max(0)
            R[i, j] = D[i - 1, j - 1] - gamma * (mx + np.log(np.exp(r - mx).sum(0)))
        return R[m, n]
    return sdtw(x, y) - 0.5 * (sdtw(x, x) + sdtw(y, y))


def gsot_w1(x, y, dt):
    """Graph-space optimal transport (Metivier et al., 2019): W1 between the point clouds (t_k, x_k) and (t_k, y_k),
    equal weights, Euclidean cost. Exact (an assignment problem); ADFWI's Misfit_wasserstein_sinkhorn (p = 1,
    blur 0.01) approximates it, and ADFWI reports dt * W1^2."""
    from scipy.optimize import linear_sum_assignment
    t = np.arange(len(x)) * dt
    P, Q = np.c_[t, x], np.c_[t, y]
    M = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    i, j = linear_sum_assignment(M)
    return M[i, j].mean()


r = np.load(os.path.join(RUNS, "diagnostics/ricker.npz"))
o, dt, x = r["obs"], float(r["dt"]), r["shifts_s"]
ks = np.round(x / dt).astype(int)
O, S = torch.as_tensor(o[None]), [torch.as_tensor(np.roll(o, k)[None]) for k in ks]
sc = 1 / np.sqrt((o ** 2).mean())        # soft-DTW sees unit-RMS data, as in overthrust2d.py
curve = {"L2": [float(bl.l2(s, O)) for s in S],
         "SeisLM": r["pretrained_full_layers"][-1],
         "L2, standardized traces": [float(bl.standardized_l2(s, O)) for s in S],
         "Envelope": [float(bl.envelope_l2(s, O, p=1.0)) for s in S],
         "W2": [float(bl.wasserstein2(s, O, dt=dt, shift=0.75)) for s in S],
         "Frequency continuation": [float(bl.lowpass_l2(s, O, dt, fc=2.0)) for s in S],
         "Soft-DTW": [sdtw_div(sc * np.roll(o, k)[::4], sc * o[::4], 30.0) for k in ks],
         "Graph-space W1": [gsot_w1(np.roll(o, k), o, dt) for k in ks]}
norm = lambda v: (np.asarray(v) - np.min(v)) / (np.max(v) - np.min(v))
i0 = int(np.argmin(np.abs(x)))
for k, v in curve.items():                # monotonic interval around zero shift, printed for the text
    v, j, m = np.asarray(v), i0, i0
    while j + 1 < len(v) and v[j + 1] > v[j]:
        j += 1
    while m > 0 and v[m - 1] > v[m]:
        m -= 1
    print(f"{k:26s} monotonic {x[m]:+.3f} .. {x[j]:+.3f} s")

run = lambda name: json.load(open(os.path.join(RUNS, name, "run.json")))
fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0))
for lab, c, ls, name in OBJ:
    lw = 3 if lab == "SeisLM" else 2.2
    if lab != "L2, standardized traces":  # one trace: standardization is a constant factor, same curve as L2
        a.plot(x, norm(curve[lab]), color=c, ls=ls, lw=lw, label=lab)
    if name:
        j = run(name)
        e1, e2 = j["stage1"]["err"], j["stage2"]["err"]
        b.plot(np.arange(len(e1)), e1, color=c, ls=ls, lw=lw, label=lab)
        b.plot(len(e1) - 1 + np.arange(len(e2)), e2, color=c, ls=ls, lw=lw)
        b.plot(len(e1) - 1, e2[0], "o", color=c, mfc="white", mew=2, ms=7, zorder=6)
b.axhline(0.172, color="#555555", lw=1, ls=(0, (2, 2)))
for ax, ttl, xl, yl in ((a, "(a) Ricker time-shift test", "time shift (s)", "Normalized misfit"),
                        (b, "(b) 2D Overthrust, linear start", "accepted L-BFGS iteration", "RMS relative velocity error")):
    ax.set_title(ttl, fontsize=16, pad=10)
    ax.set_xlabel(xl, fontsize=14)
    ax.set_ylabel(yl, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
h, l = a.get_legend_handles_labels()
hb, lb = b.get_legend_handles_labels()
h += [hb[lb.index("L2, standardized traces")]]
l += ["L2, standardized traces"]
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig(os.path.join(OUT, "fig_baselines.png"), dpi=170, bbox_inches="tight")

# SI Fig S5: stage-1 and stage-2 models of the five baselines, colour scale of Fig 11
xg, zg = np.arange(501) * 0.025, np.arange(207) * 0.025
base = [o_ for o_ in OBJ[2:] if o_[3]]
fig, axs = plt.subplots(len(base), 2, figsize=(20, 4.4 * len(base)))
for row, (lab, _, _, name) in enumerate(base):
    for col, (f, st) in enumerate((("vp_inv.npy", "stage 1"), ("vp_inv_refl.npy", "stage 2"))):
        ax = axs[row, col]
        im = ax.imshow(np.load(os.path.join(RUNS, name, f)).T, vmin=1.5, vmax=6.0, cmap="jet",
                       extent=(xg[0], xg[-1], zg[-1], zg[0]), aspect="auto")
        ax.set_title(f"({'abcdefghij'[2 * row + col]}) {lab}, {st}", fontsize=19)
        ax.set_xlabel("X [km]", fontsize=14)
        ax.set_ylabel("Z [km]", fontsize=14)
        ax.set_xticks(np.arange(0, 12.6, 2))
        ax.set_yticks(np.arange(0, 5.1, 1))
        ax.grid(False)
        ax.tick_params(colors="black", labelsize=14)
        for s in ax.spines.values():
            s.set_edgecolor("black")
fig.subplots_adjust(hspace=0.45, wspace=0.08, bottom=0.08)
cax = fig.add_axes([0.33, 0.025, 0.34, 0.012])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[2, 3, 4, 5, 6])
cb.set_label("Velocity [km/s]", fontsize=20)
cb.ax.tick_params(labelsize=20, colors="black")
cb.outline.set_edgecolor("black")
cb.ax.grid(False)
fig.savefig(os.path.join(OUT, "figS5_baseline_models.png"), dpi=110, bbox_inches="tight")
print("saved")
