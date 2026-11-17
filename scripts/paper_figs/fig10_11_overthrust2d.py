# Paper Figs 10-11 (2D Overthrust) from the runs ref_l2, ref_feature, ref_hybrid (overthrust2d.py; S1 100 it masked + S2 100 it L2;
# runs/<name>), in the style of pool_none/plot_overthrust_surface_lin_init.py.
# Usage: python <this> <outdir> [nohybrid]   (nohybrid: main-text Figs 10-11, L2 and SeisLM only; without it, SI Figs S9-S10)
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md

D = RUNS
OUT = sys.argv[1]
L2_C, SEISLM_C, HYBRID_C = "#E8A317", "#1B9E9E", "#9B59B6"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
ld = lambda run, f: np.load(os.path.join(D, run, f)).astype(float)
err = lambda run: np.concatenate([ld(run, "err.npy"), ld(run, "err_refl.npy")[1:]])   # 0..200, start included once

runs = [("L2", L2_C, "ref_l2", []), ("SeisLM", SEISLM_C, "ref_feature", []), ("Hybrid", HYBRID_C, "ref_hybrid", [])]
if "nohybrid" in sys.argv[2:]:
    runs = runs[:2]
fig, (axL, axE) = plt.subplots(1, 2, figsize=(12.5, 4.6))
for lab, c, run, reps in runs:
    l1, l2 = ld(run, "loss.npy"), ld(run, "loss_refl.npy")
    axL.plot(np.arange(len(l1)), l1 / l1[0], color=c, lw=3, label=lab)
    axL.plot(np.arange(len(l2)), l2 / l2[0], color=c, lw=3, ls="--")
    e = err(run)
    axE.plot(np.arange(len(e)), e, color=c, lw=3, label=lab)
    n1 = len(ld(run, "err.npy")) - 1
    axE.plot(n1, e[n1], "o", color=c, mfc="white", mew=2.2, ms=9, zorder=6)
for ax, ttl, ylab in [(axL, "(a) Loss", "Normalized loss"), (axE, "(b) Velocity error", "RMS relative velocity error")]:
    ax.set_title(ttl, fontsize=17, pad=10)
    ax.set_xlabel("accepted L-BFGS iteration", fontsize=15)
    ax.set_ylabel(ylab, fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
handles, labels = axE.get_legend_handles_labels()
extra = [mlines.Line2D([], [], color="#555555", lw=3, ls="-"), mlines.Line2D([], [], color="#555555", lw=3, ls="--"),
         mlines.Line2D([], [], marker="o", color="#555555", mfc="white", mew=2.2, ms=9, ls="none")]
fig.legend(handles + extra, labels + ["stage 1", "stage 2 (refl)", "stage 2 start"],
           loc="lower center", ncol=len(handles) + 3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.04),
           handlelength=1.9, columnspacing=1.5)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(os.path.join(OUT, "fig10_proposed.png"), dpi=170, bbox_inches="tight")

# Fig 11: six models, same layout and colour scale as the submitted figure
x = np.arange(501) * 0.025
z = np.arange(207) * 0.025
fig, axs = plt.subplots(len(runs), 2, figsize=(20, 4.5 * len(runs)))
for row, (lab, _, run, _) in enumerate(runs):
    for col, (f, st) in enumerate([("vp_inv.npy", "stage 1"), ("vp_inv_refl.npy", "stage 2")]):
        ax = axs[row, col]
        im = ax.imshow(ld(run, f).T, vmin=1.5, vmax=6.0, cmap="jet", extent=(x[0], x[-1], z[-1], z[0]), aspect="auto")
        ax.set_title(f"({'abcdef'[2 * row + col]}) {lab}, {st}", fontsize=20)
        ax.set_xlabel("X [km]", fontsize=14)
        ax.set_ylabel("Z [km]", fontsize=14)
        ax.set_xticks(np.arange(0, 12.6, 2))
        ax.set_yticks(np.arange(0, 5.1, 1))
        ax.grid(False)
        ax.tick_params(colors="black", labelsize=14)
        for s in ax.spines.values():
            s.set_edgecolor("black")
fig.subplots_adjust(hspace=0.35, wspace=0.08, bottom=0.12)
cax = fig.add_axes([0.33, 0.04, 0.34, 0.018])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[2, 3, 4, 5, 6])   # as Fig 9
cb.set_label("Velocity [km/s]", fontsize=20)
cb.ax.tick_params(labelsize=20, colors="black")
cb.outline.set_edgecolor("black")
cb.ax.grid(False)                                      # rcParams grid draws white lines on the colorbar
fig.savefig(os.path.join(OUT, "fig11_proposed.png"), dpi=110, bbox_inches="tight")
print("saved")
