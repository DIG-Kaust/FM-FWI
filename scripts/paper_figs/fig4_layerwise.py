# Paper Fig 4: rigid time shift of the masked middle Marmousi shot gather, feature distance per encoder layer,
# pretrained and randomly initialized encoder. The paper's Fig 4 is the submitted PNG (its plotting script was not
# kept); this script draws the same curves from the same data in a close style, not pixel for pixel. With "mean" it
# also draws the time-averaged last-block distance (that curve is in SI Figure S12, figS7_layer_mean.py).
# Inputs: runs/diagnostics/layerwise/layerwise_marmousi_pool{none,mean}.npz (scripts/layerwise/layerwise_sensitivity.py
# --source marmousi --pool none / mean): full sequence and time average.
# Usage: python fig4_layerwise.py <outdir> [mean]
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(H, "../../runs"))   # driver outputs, see REPRODUCE.md
D, OUT = os.path.join(RUNS, "diagnostics/layerwise"), sys.argv[1]
L2_C, MEAN_C = "#E8A317", "#7B3294"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
full = np.load(os.path.join(D, "layerwise_marmousi_poolnone.npz"))
mean = np.load(os.path.join(D, "layerwise_marmousi_poolmean.npz"))
x = full["shifts_s"] / (full["shifts_s"].max() / 2.5)          # shifts span +-2.5 dominant periods
n = lambda v: v / v.max()
teal = [plt.cm.BuGn(0.35 + 0.1 * k) for k in range(6)]
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
for ax, key, ttl in ((axs[0], "pretrained_curves", "(a) Pre-trained encoder"),
                     (axs[1], "random_curves", "(b) Randomly initialized encoder")):
    ax.plot(x, n(full["l2"]), color=L2_C, lw=2.4, ls=":")
    ax.plot(x, n(full[key][0]), color="#777777", lw=2.2, ls="--")              # convolutional front end
    for k in range(1, 7):
        ax.plot(x, n(full[key][k]), color="#1B9E9E" if k == 6 else teal[k - 1], lw=3.2 if k == 6 else 2)
    if "mean" in sys.argv[2:]:
        ax.plot(x, n(mean[key][-1]), color=MEAN_C, lw=2.8, ls="-.")            # block 6, time-averaged
    ax.set_title(ttl, fontsize=16, pad=10)
    ax.set_xlabel("Time shift (dominant periods)", fontsize=14)
    ax.set_xlim(-2.5, 2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
axs[0].set_ylabel("Normalized feature distance", fontsize=14)
h = [mlines.Line2D([], [], color=L2_C, lw=2.4, ls=":"), mlines.Line2D([], [], color="#777777", lw=2.2, ls="--")]
h += [mlines.Line2D([], [], color=teal[k - 1], lw=2) for k in range(1, 6)]
h += [mlines.Line2D([], [], color="#1B9E9E", lw=3.2)]
lab = ["$L_2$ (data space)", "conv/proj"] + [f"block {k}" for k in range(1, 7)]
if "mean" in sys.argv[2:]:
    h.append(mlines.Line2D([], [], color=MEAN_C, lw=2.8, ls="-."))
    lab.append("block 6, time-averaged")
fig.legend(h, lab, loc="lower center", ncol=len(h), frameon=False, fontsize=11.5,
           bbox_to_anchor=(0.5, -0.04), handlelength=1.8, columnspacing=1.1)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(os.path.join(OUT, "fig_layerwise_timeshift.png"), dpi=170, bbox_inches="tight")
print("saved")
