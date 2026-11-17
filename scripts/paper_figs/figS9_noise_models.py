# SI Figure S14: 2D Overthrust models at 5 dB of flat in-band noise (1-11 Hz) from the linear start,
# without and with the noise before the first arrival muted, after stage 1 and stage 2; layout and colour scale as Fig 11.
# Inputs: runs/{noise,nmute}_flat5_{l2,feature}/{vp_inv,vp_inv_refl,err,err_refl}.npy (the 5-dB runs of Fig 13 and
# Table 3). Usage: python figS9_noise_models.py <outdir>
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md

D = RUNS
OUT = sys.argv[1]
ld = lambda run, f: np.load(os.path.join(D, run, f)).astype(float)
rows = [("$L_2$", "noise_flat5_l2"), ("$L_2$, muted", "nmute_flat5_l2"),
        ("SeisLM", "noise_flat5_feature"), ("SeisLM, muted", "nmute_flat5_feature")]
x, z = np.arange(501) * 0.025, np.arange(207) * 0.025
fig, axs = plt.subplots(4, 2, figsize=(20, 18))
for r, (lab, run) in enumerate(rows):
    for c, (f, e, st) in enumerate([("vp_inv.npy", "err.npy", "stage 1"), ("vp_inv_refl.npy", "err_refl.npy", "stage 2")]):
        ax = axs[r, c]
        im = ax.imshow(ld(run, f).T, vmin=1.5, vmax=6.0, cmap="jet", extent=(x[0], x[-1], z[-1], z[0]), aspect="auto")
        ax.set_title(f"({'abcdefgh'[2 * r + c]}) {lab}, {st}: error {ld(run, e)[-1]:.3f}", fontsize=20)
        ax.set_xlabel("X [km]", fontsize=14)
        ax.set_ylabel("Z [km]", fontsize=14)
        ax.set_xticks(np.arange(0, 12.6, 2))
        ax.set_yticks(np.arange(0, 5.1, 1))
        ax.tick_params(colors="black", labelsize=14)
        for s in ax.spines.values():
            s.set_edgecolor("black")
fig.subplots_adjust(hspace=0.4, wspace=0.08, bottom=0.1)
cax = fig.add_axes([0.33, 0.035, 0.34, 0.014])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[2, 3, 4, 5, 6])   # as Figs 9 and 11
cb.set_label("Velocity [km/s]", fontsize=20)
cb.ax.tick_params(labelsize=20, colors="black")
fig.savefig(os.path.join(OUT, "figS9_noise_models.png"), dpi=110, bbox_inches="tight")
print("saved")
