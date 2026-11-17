# SI Figure S12 (R4.5, R1.3): rigid time shift of the masked middle Marmousi shot gather, last transformer block,
# full feature sequence against time-averaged features, pretrained and randomly initialized encoder, with the
# data-space L2 distance. Same data as Fig 4 (runs/diagnostics/layerwise/layerwise_marmousi_pool{none,mean}.npz).
# Usage: python figS7_layer_mean.py <outdir>
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md

H = os.path.dirname(os.path.abspath(__file__))
D, OUT = os.path.join(RUNS, "diagnostics/layerwise"), sys.argv[1]
L2_C, FULL_C, MEAN_C = "#E8A317", "#1B9E9E", "#7B3294"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
full = np.load(os.path.join(D, "layerwise_marmousi_poolnone.npz"))
mean = np.load(os.path.join(D, "layerwise_marmousi_poolmean.npz"))
x = full["shifts_s"] / (full["shifts_s"].max() / 2.5)          # shifts span +-2.5 dominant periods
n = lambda v: v / v.max()
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
for ax, key, ttl in ((axs[0], "pretrained_curves", "(a) Pre-trained encoder"),
                     (axs[1], "random_curves", "(b) Randomly initialized encoder")):
    ax.plot(x, n(full["l2"]), color=L2_C, lw=2.4, ls=":")
    ax.plot(x, n(full[key][-1]), color=FULL_C, lw=3)
    ax.plot(x, n(mean[key][-1]), color=MEAN_C, lw=3, ls="-.")
    ax.set_title(ttl, fontsize=16, pad=10)
    ax.set_xlabel("Time shift (dominant periods)", fontsize=14)
    ax.set_xlim(-2.5, 2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
axs[0].set_ylabel("Normalized distance", fontsize=14)
h = [mlines.Line2D([], [], color=L2_C, lw=2.4, ls=":"), mlines.Line2D([], [], color=FULL_C, lw=3),
     mlines.Line2D([], [], color=MEAN_C, lw=3, ls="-.")]
fig.legend(h, ["$L_2$ (data space)", "block 6, full sequence", "block 6, time-averaged"], loc="lower center",
           ncol=3, frameon=False, fontsize=12.5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(os.path.join(OUT, "figS7_layer_mean.png"), dpi=170, bbox_inches="tight")
print("saved")
