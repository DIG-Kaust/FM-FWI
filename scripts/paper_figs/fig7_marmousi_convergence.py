#!/usr/bin/env python3
"""Paper Figure 7: loss and velocity error of the three Marmousi workflows (stage 1, then stage-2 L2), from
runs/marm/marm_{l2,feature,hybrid} of scripts/marm2d.py (err.npy, err_refl.npy, loss.npy, loss_refl.npy: one value per
accepted iteration; err.npy starts with the starting-model error).
Usage: python fig7_marmousi_convergence.py <outdir> [nohybrid]   (nohybrid: main-text Fig 7; without it, SI Fig S7)
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md
OUT = sys.argv[1]
L2_C, SEISLM_C, HYBRID_C = "#E8A317", "#1B9E9E", "#9B59B6"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor":   "#bdbdbd", "axes.linewidth": 1.0,
    "axes.grid":        True,      "grid.color":     "#dddddd",
    "grid.linewidth":   0.9,       "xtick.color":    "#555555",
    "ytick.color":      "#555555", "font.size":      13,
})


def load(name):
    d = os.path.join(RUNS, "marm", name)
    return tuple(np.load(os.path.join(d, f)).astype(float) for f in ("err.npy", "err_refl.npy", "loss.npy", "loss_refl.npy"))


def make_plot(out_path, runs):
    """runs: list of (label, color, err1, err2, loss1, loss2)"""
    fig, (axL, axE) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    init_err = np.mean([r[2][0] for r in runs])

    for lab, c, err1, err2, loss1, loss2 in runs:
        v1 = err1.copy(); v1[0] = init_err
        verr = np.concatenate([v1, err2])
        n1   = len(v1)
        axE.plot(np.arange(len(verr)), verr, color=c, lw=3, label=lab)
        axE.plot(n1, err2[0], "o", color=c, mfc="white", mew=2.2, ms=9, zorder=6)

        axL.plot(np.arange(len(loss1)), loss1 / loss1[0], color=c, lw=3, label=lab)
        if loss2.size > 1:
            axL.plot(np.arange(len(loss2)), loss2 / loss2[0], color=c, lw=3, ls="--")

    for ax, ttl, ylab in [
        (axL, "(a) Loss",           "Normalized loss"),
        (axE, "(b) Velocity error", "RMS relative velocity error"),
    ]:
        ax.set_title(ttl, fontsize=17, pad=10)
        ax.set_xlabel("accepted L-BFGS iteration", fontsize=15)
        ax.set_ylabel(ylab, fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=0)

    handles, labels = axE.get_legend_handles_labels()
    stage1_h = mlines.Line2D([], [], color="#555555", lw=3, ls="-",  label="stage 1")
    refl_h   = mlines.Line2D([], [], color="#555555", lw=3, ls="--", label="stage 2 (refl)")
    start_h  = mlines.Line2D([], [], marker="o", color="#555555", mfc="white",
                              mew=2.2, ms=9, ls="none", label="stage 2 start")
    fig.legend(handles + [stage1_h, refl_h, start_h],
               labels  + ["stage 1", "stage 2 (refl)", "stage 2 start"],
               loc="lower center", ncol=len(handles) + 3, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, -0.04), handlelength=1.9, columnspacing=1.5)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


runs = [("L2", L2_C, *load("marm_l2_rerun")), ("SeisLM", SEISLM_C, *load("marm_feature_rerun")),
        ("Hybrid", HYBRID_C, *load("marm_hybrid_rerun"))]
if "nohybrid" in sys.argv[2:]:
    runs = runs[:2]
make_plot(os.path.join(OUT, "fig_6_perf.png"), runs)
