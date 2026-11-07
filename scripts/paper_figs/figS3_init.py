# Supporting Information Fig S3 (Text S2): tolerance to the starting model, 2D Overthrust, L2 vs SeisLM.
# Inputs: runs/init_* of fwi2d.py (err.npy, err_refl.npy); the unscaled linear start (x1.00) is runs/ref_l2 and
# runs/ref_feature. Stage-2 points are drawn where stage 2 exists.
# Usage: python figS3_init.py <outdir>
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
L2_C, SEISLM_C = "#E8A317", "#1B9E9E"


def run(tag, obj):
    if tag == "s1.00":
        d = os.path.join(R, "ref_l2" if obj == "l2" else "ref_feature")
    else:
        b = "_b" if obj == "feature" and tag in ("s0.85", "s1.15") else ""
        d = os.path.join(D, f"init_{tag}_{obj}{b}")
    e = np.load(os.path.join(d, "err.npy"))
    p = os.path.join(d, "err_refl.npy")
    return e[0], e[-1], (np.load(p)[-1] if os.path.exists(p) else np.nan)


groups = [("(a) Scaled linear-gradient start", "scale factor of the starting model",
           [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15], lambda v: f"s{v:.2f}"),
          ("(b) Smoothed true model", "Gaussian smoothing length (km, horizontal)",
           [0.5, 1.25, 2.5], lambda v: {0.5: "sm20", 1.25: "sm50", 2.5: "sm100"}[v])]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
for ax, (ttl, xl, xs, tag) in zip(axs, groups):
    st = np.array([run(tag(v), "l2")[0] for v in xs])
    ax.plot(xs, st, color="#555555", lw=1.6, ls=(0, (2, 2)), marker="s", ms=6, label="starting model")
    for obj, c, lab in (("l2", L2_C, "L2"), ("feature", SEISLM_C, "SeisLM")):
        r = np.array([run(tag(v), obj) for v in xs])
        ax.plot(xs, r[:, 1], color=c, lw=2.6, marker="o", ms=7, label=f"{lab}, stage 1")
        ax.plot(xs, r[:, 2], color=c, lw=2.6, ls="--", marker="o", mfc="white", mew=2, ms=7, label=f"{lab}, stage 2")
    ax.set_title(ttl, fontsize=16, pad=10)
    ax.set_xlabel(xl, fontsize=14)
    ax.set_ylabel("RMS relative velocity error", fontsize=13)
    ax.set_xticks(xs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
axs[1].set_ylim(axs[0].get_ylim())
axs[1].legend(frameon=False, fontsize=11, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figS3_init_tolerance.png"), dpi=170, bbox_inches="tight")
print("saved")
