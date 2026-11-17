# Paper Figs 5-6 (Marmousi model and first-iteration gradients), two panels side by side with one horizontal
# colorbar, laid out like the 2D Overthrust model figure (fig_2d_ovth2.png). Revision, 10 Nov 2026.
# True model data/Marm.bin; initial model as in notebooks/SeisLM_Marmousi.ipynb (gaussian_filter sigma (60, 40),
# water layer kept at 1.5 km/s); first gradients runs/marm/marm_{l2,feature}_rerun/grad.npy of scripts/marm2d.py
# (water muted and smoothed with sigma (4, 4)), each divided by its maximum absolute value. The paper used the
# grad.npy of the original notebook runs; marm2d.py saves the same quantity from its own run.
# Usage: python fig5_6_marmousi_side.py <outdir>
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "../.."); OUT = sys.argv[1]
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(H, "../../runs"))   # driver outputs, see REPRODUCE.md
nx, nz, d = 601, 221, 0.015
x, z = np.arange(nx) * d, np.arange(nz) * d
vt = np.fromfile(os.path.join(R, "data/Marm.bin"), np.float32).reshape(nz, nx).T
msk = (vt > 1.52).astype(int)
vi = gaussian_filter(vt, sigma=[60, 40]) * msk
vi[vi == 0] = 1.5
xs, xr = np.arange(30) * 0.3, np.arange(300) * 0.03
gl, gf = (np.load(os.path.join(RUNS, "marm", f"marm_{k}_rerun", "grad.npy")) for k in ("l2", "feature"))


def side(panels, cmap, vmin, vmax, clabel, name, acq=False, ticks=None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.3), gridspec_kw=dict(wspace=0.12))
    for ax, (t, m) in zip(axs, panels):
        im = ax.imshow(m.T, cmap=cmap, vmin=vmin, vmax=vmax, extent=(x[0], x[-1], z[-1], z[0]), aspect="auto")
        if acq:
            ax.plot(xr, np.full_like(xr, 0.01), "w.", ms=2, clip_on=False)
            ax.plot(xs, np.full_like(xs, 0.01), "ro", ms=4, clip_on=False)
        ax.set_title(t, fontsize=17); ax.set_xlabel("X [km]", fontsize=13); ax.set_ylabel("Z [km]", fontsize=13)
        ax.set_xticks(np.arange(0, 9.1, 1)); ax.tick_params(labelsize=12)
    cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.06, pad=0.2, aspect=40, shrink=0.45)
    cb.set_label(clabel, fontsize=16)
    if ticks is not None: cb.set_ticks(ticks); cb.ax.tick_params(labelsize=13)
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight"); plt.close(fig)


vmin, vmax = np.percentile(vt, [2, 98])   # as the notebook, so Fig 5 shares the colour scale of Fig 8
side([("(a) True model", vt), ("(b) Initial model", vi)], "jet", vmin, vmax, "Velocity [km/s]",
     "fig_4_marm.png", acq=True)
side([("(a) $L_2$", gl / np.abs(gl).max()), ("(b) SeisLM", gf / np.abs(gf).max())], "seismic", -0.1, 0.1,
     "g/max(|g|)", "fig_5_grad.png", ticks=[-0.1, -0.05, 0, 0.05, 0.1])
print("saved")
