# Paper figure (R1.3, R3.m4, R4.5): full versus time-averaged SeisLM features.
# (a) Rigid time-shift test on the masked middle 2D Overthrust shot, last transformer block, pretrained encoder
#     (runs/diagnostics/rigid.npz from shift_diagnostics.py), each curve normalized to 0-1.
# (b) 3D Overthrust from the linear start, windowed velocity error: time-averaged = the paper run
#     (3d/overthrust_feat_linear_init, stage 2 _refl + _refl2, 100 + 200 it); full sequence = the July run
#     (3d/overthrust_feat_water_linear_init_pool_none100, stage 2 s2_300, one 300-it run).
# Usage: python fig18_pooling.py <outdir>
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("FMFWI_RUNS", os.path.join(H, "../../runs"))   # driver outputs, see REPRODUCE.md
OUT = sys.argv[1]
FULL_C, MEAN_C = "#1B9E9E", "#7B3294"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
err = lambda d: np.load(os.path.join(R, d, "vp_error_local_history.npy"))
norm = lambda v: (v - v.min()) / (v.max() - v.min())

r = np.load(os.path.join(R, "diagnostics/rigid.npz"))
x = r["shifts_s"] / float(r["period_s"])
fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0))
for key, c, lab in (("pretrained_full_layers", FULL_C, "full sequence"), ("pretrained_mean_layers", MEAN_C, "time average")):
    v = norm(r[key][-1])
    a.plot(x, v, color=c, lw=2.8, label=lab)
    a.plot(x[v.argmin()], 0, "v", color=c, ms=10, clip_on=False, zorder=6)
    print(f"{lab}: minimum at {x[v.argmin()]:+.2f} periods")
a.axvline(0, color="#555555", lw=1, ls=(0, (2, 2)))

for (s1, s2), c in (((["3d/overthrust_feat_linear_init"],
                      ["3d/overthrust_feat_linear_init_refl", "3d/overthrust_feat_linear_init_refl2"]), MEAN_C),
                    ((["3d/overthrust_feat_water_linear_init_pool_none100"], ["3d/s2_300"]), FULL_C)):
    e1 = err(s1[0])
    e2 = np.concatenate([err(d)[1:] for d in s2])
    b.plot(np.arange(len(e1) + len(e2)), np.concatenate([e1, e2]), color=c, lw=2.8)
    b.plot(len(e1) - 1, e1[-1], "o", color=c, mfc="white", mew=2.2, ms=8, zorder=6)
    print(f"3D {c}: S1 {e1[-1]:.3f}, S2 {e2[-1]:.3f}")
for ax, ttl, xl, yl in ((a, "(a) Rigid time-shift test, 2D", "time shift (dominant periods of the 5-Hz wavelet)",
                         "Normalized feature distance"),
                        (b, "(b) 3D Overthrust, linear start", "accepted L-BFGS iteration", "RMS relative velocity error")):
    ax.set_title(ttl, fontsize=16, pad=10)
    ax.set_xlabel(xl, fontsize=14)
    ax.set_ylabel(yl, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
h = [mlines.Line2D([], [], color=FULL_C, lw=2.8), mlines.Line2D([], [], color=MEAN_C, lw=2.8),
     mlines.Line2D([], [], marker="o", color="#555555", mfc="white", mew=2.2, ms=8, ls="none")]
fig.legend(h, ["full feature sequence", "time-averaged features", "stage 2 start"], loc="lower center", ncol=3,
           frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(os.path.join(OUT, "fig_pooling.png"), dpi=170, bbox_inches="tight")
print("saved")
