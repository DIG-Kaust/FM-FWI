# Paper Figure 8: Marmousi models after stage 1 (left) and stage 2 (right) for the L2, SeisLM and hybrid workflows, from
# runs/marm/marm_{l2,feature,hybrid}_rerun (scripts/marm2d.py: vp_inv.npy, vp_inv_refl.npy). Colour scale as
# Fig 5 (jet, 1.5-5.486 km/s).
# Usage: python fig8_marmousi_models.py <outdir> [nohybrid]   (nohybrid: main-text Fig 8; without it, SI Fig S8)
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md
x, z = np.arange(601) * 0.015, np.arange(221) * 0.015
runs = (("$L_2$", "marm_l2_rerun"), ("SeisLM", "marm_feature_rerun"), ("Hybrid", "marm_hybrid_rerun"))
if "nohybrid" in sys.argv[2:]:
    runs = runs[:2]
fig, axs = plt.subplots(len(runs), 2, figsize=(20, 4.5 * len(runs)))
for r, (lab, name) in enumerate(runs):
    for c, (f, st) in enumerate((("vp_inv.npy", "stage 1"), ("vp_inv_refl.npy", "stage 2"))):
        ax = axs[r, c]
        im = ax.imshow(np.load(os.path.join(RUNS, "marm", name, f)).T, vmin=1.5, vmax=5.486, cmap="jet",
                       extent=(x[0], x[-1], z[-1], z[0]), aspect="auto")
        ax.set_title(f"({'abcdef'[2 * r + c]}) {lab}, {st}", fontsize=20)
        ax.set_xlabel("X [km]", fontsize=14)
        ax.set_ylabel("Z [km]", fontsize=14)
        ax.set_xticks(np.arange(0, 9.1, 1))
        ax.set_yticks(np.arange(0, 3.4, 1))
        ax.tick_params(colors="black", labelsize=14)
fig.subplots_adjust(hspace=0.35, wspace=0.08, bottom=0.12)
cax = fig.add_axes([0.33, 0.04, 0.34, 0.018])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("Velocity [km/s]", fontsize=20)
cb.ax.tick_params(labelsize=20)
fig.savefig(os.path.join(sys.argv[1], "fig_7_marm_inv.png"), dpi=110, bbox_inches="tight")
print("saved")
