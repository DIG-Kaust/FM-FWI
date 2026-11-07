# Supporting Information Figs S1-S2 (Text S1): pretrained vs random encoder and feature blocks, 2D Overthrust.
# Inputs: runs/<name> of overthrust2d.py (err.npy, err_refl.npy, vp_inv*.npy): rand_s0-2, layer0, layer3, ref_feature, ref_l2. Style as the paper's convergence and model figures.
# Usage: python figS1_S2_encoder.py <outdir>
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md

H = os.path.dirname(os.path.abspath(__file__))
D, R, OUT = RUNS, RUNS, sys.argv[1]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
L2_C, SEISLM_C, RAND_C = "#E8A317", "#1B9E9E", "#D62728"


def err(folder, run):
    e = np.load(os.path.join(folder, run, "err.npy"))
    p = os.path.join(folder, run, "err_refl.npy")
    return (e, np.load(p)) if os.path.exists(p) else (e, None)


def plot_run(ax, folder, run, c, lab, ls="-"):
    e1, e2 = err(folder, run)
    ax.plot(np.arange(len(e1)), e1, color=c, lw=2.6, ls=ls, label=lab)
    if e2 is not None:
        n1 = len(e1) - 1
        ax.plot(n1 + np.arange(len(e2)), e2, color=c, lw=2.6, ls=ls)
        ax.plot(n1, e2[0], "o", color=c, mfc="white", mew=2, ms=8, zorder=6)


fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
plot_run(a, R, "ref_l2", L2_C, "L2")
plot_run(a, R, "ref_feature", SEISLM_C, "SeisLM, pretrained")
for k, ls in zip((0, 1, 2), ("-", "--", ":")):
    plot_run(a, D, f"rand_s{k}", RAND_C, f"SeisLM, random seed {k}", ls)
plot_run(b, R, "ref_feature", SEISLM_C, "block 6 (last)")
plot_run(b, D, "layer3", "#3B5BA5", "block 4")
plot_run(b, D, "layer0", "#7F7F7F", "block 1")
for ax, ttl in ((a, "(a) Pretrained vs random encoder"), (b, "(b) Feature block, pretrained encoder")):
    ax.axhline(0.172, color="#555555", lw=1, ls=(0, (2, 2)))
    ax.set_title(ttl, fontsize=16, pad=10)
    ax.set_xlabel("accepted L-BFGS iteration", fontsize=14)
    ax.set_ylabel("RMS relative velocity error", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
b.set_xlim(a.get_xlim())
b.set_ylim(a.get_ylim())
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figS1_encoder_convergence.png"), dpi=170, bbox_inches="tight")

# Fig S2: stage-1 and stage-2 models, pretrained vs random (seed 0), same colour scale as Fig 11
x, z = np.arange(501) * 0.025, np.arange(207) * 0.025
panels = [(R, "ref_feature", "pretrained encoder"), (D, "rand_s0", "random encoder (seed 0)")]
fig, axs = plt.subplots(2, 2, figsize=(20, 9.5))
for row, (folder, run, lab) in enumerate(panels):
    for col, (f, st) in enumerate((("vp_inv.npy", "stage 1"), ("vp_inv_refl.npy", "stage 2"))):
        ax = axs[row, col]
        im = ax.imshow(np.load(os.path.join(folder, run, f)).T, vmin=1.5, vmax=6.0, cmap="jet",
                       extent=(x[0], x[-1], z[-1], z[0]), aspect="auto")
        ax.set_title(f"({'abcd'[2 * row + col]}) SeisLM, {lab}, {st}", fontsize=19)
        ax.set_xlabel("X [km]", fontsize=14)
        ax.set_ylabel("Z [km]", fontsize=14)
        ax.set_xticks(np.arange(0, 12.6, 2))
        ax.set_yticks(np.arange(0, 5.1, 1))
        ax.grid(False)
        ax.tick_params(colors="black", labelsize=14)
        for s in ax.spines.values():
            s.set_edgecolor("black")
fig.subplots_adjust(hspace=0.38, wspace=0.08, bottom=0.17)
cax = fig.add_axes([0.33, 0.05, 0.34, 0.025])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[2, 3, 4, 5, 6])
cb.set_label("Velocity [km/s]", fontsize=20)
cb.ax.tick_params(labelsize=20, colors="black")
cb.outline.set_edgecolor("black")
cb.ax.grid(False)
fig.savefig(os.path.join(OUT, "figS2_encoder_models.png"), dpi=110, bbox_inches="tight")
print("saved")
