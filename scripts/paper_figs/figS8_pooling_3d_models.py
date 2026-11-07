# SI Figure S8 (R1.3, R3.m4, R4.5): 3D Overthrust models from the linear start with time-averaged and full-sequence
# SeisLM features, after stage 1 (100 it) and after the 300-it L2 stage 2; layout and colour scale as Fig 17.
# Inputs (runs/3d/<name>, scripts/overthrust3d.py), vp_current.npy and vp_error_local_history.npy of
#   time average:  overthrust_feat_linear_init (S1), overthrust_feat_linear_init_refl2 (S2, after _refl)
#   full sequence: overthrust_feat_water_linear_init_pool_none100 (S1), s2_300 (S2)
# the runs of Figure 18(b). Usage: python figS8_pooling_3d_models.py <outdir>
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md

H = os.path.dirname(os.path.abspath(__file__))
R, OUT = os.path.join(RUNS, "3d"), sys.argv[1]
ld = lambda run, f: np.load(os.path.join(R, run, f + ".npy")).astype(float)
panels = [("(a) Time average, stage 1", "overthrust_feat_linear_init"),
          ("(b) Full sequence, stage 1", "overthrust_feat_water_linear_init_pool_none100"),
          ("(c) Time average, stage 2", "overthrust_feat_linear_init_refl2"),
          ("(d) Full sequence, stage 2", "s2_300")]
nx, ny, nz = ld(panels[0][1], "vp_current").shape
x, y, z = (np.arange(n) * 0.05 for n in (nx, ny, nz))
ix, iy, iz = nx // 2, ny // 2, nz // 4
Lx, Ly, Lz = x[-1], y[-1], z[-1]
plt.rcParams.update({"figure.facecolor": "white", "font.size": 20})
fig = plt.figure(figsize=(2 * (Lx + Ly) * 0.95 + 1.6, 2 * (Ly + Lz) * 0.95 + 1.6))
outer = gridspec.GridSpec(2, 2, wspace=0.08, hspace=0.16)
kw = dict(cmap="jet", vmin=1.5, vmax=6.0, aspect="auto", interpolation="nearest")
for k, (title, run) in enumerate(panels):
    v = ld(run, "vp_current")
    err = ld(run, "vp_error_local_history")[-1]
    g = gridspec.GridSpecFromSubplotSpec(2, 2, outer[k], width_ratios=[Lx, Ly], height_ratios=[Ly, Lz],
                                         hspace=0.05, wspace=0.05)
    am, axz, acb = fig.add_subplot(g[0, 0]), fig.add_subplot(g[1, 0]), fig.add_subplot(g[0, 1])
    acb.axis("off")
    acb.text(0.06, 0.95, f"Error: {err:.3f}", transform=acb.transAxes, va="top", fontsize=22)
    ayz = fig.add_subplot(g[1, 1], sharey=axz)
    am.imshow(v[:, :, iz].T, extent=[0, Lx, 0, Ly], origin="lower", **kw)
    am.axhline(y[iy], color="k", lw=1, ls="--")
    am.axvline(x[ix], color="m", lw=1, ls="--")
    am.set_ylabel("Crossline (km)")
    am.tick_params(labelbottom=False)
    am.set_title(title, fontsize=26, loc="left", pad=12)
    axz.imshow(v[:, iy, :].T, extent=[0, Lx, Lz, 0], **kw)
    axz.axvline(x[ix], color="m", lw=1, ls="--")
    axz.axhline(z[iz], color="#555555", lw=1, ls="--")
    axz.set_xlabel("Inline (km)")
    axz.set_ylabel("Depth (km)")
    im = ayz.imshow(v[ix, :, :].T, extent=[0, Ly, Lz, 0], **kw)
    ayz.axvline(y[iy], color="k", lw=1, ls="--")
    ayz.axhline(z[iz], color="#555555", lw=1, ls="--")
    ayz.set_xlabel("Crossline (km)")
    ayz.tick_params(labelleft=False)
    for a in (am, axz, ayz):
        a.grid(False)
        a.tick_params(colors="black")
cax = acb.inset_axes([0.08, 0.05, 0.08, 0.45])          # colorbar in the corner of panel (d)
cb = fig.colorbar(im, cax=cax)
cb.set_label("Velocity [km/s]", fontsize=20)
fig.savefig(os.path.join(OUT, "figS8_pooling_3d_models.png"), dpi=120, bbox_inches="tight")
print("saved")
