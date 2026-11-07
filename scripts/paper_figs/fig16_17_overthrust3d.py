# Paper Figs 16-17 (R4.4): 3D Overthrust, both workflows with the same stage-2 L2 refinement.
# Inputs (runs/3d/<name>, scripts/overthrust3d.py): loss_history.npy and vp_error_local_history.npy (one value per
# accepted iteration, initial model first for the error) and vp_current.npy (final model) of
#   overthrust_l2_water_linear_init           L2 stage 1 (17 it)
#   overthrust_l2_water_linear_init_refl/2    L2 stage 2 (100 + 200 it)
#   overthrust_feat_linear_init               SeisLM stage 1 (100 it)
#   overthrust_feat_linear_init_refl/2        L2 stage 2 (100 + 200 it)
# Usage: python fig16_17_overthrust3d.py <outdir>
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md

H = os.path.dirname(os.path.abspath(__file__))
D, OUT = os.path.join(RUNS, "3d"), sys.argv[1]
L2_C, SEISLM_C = "#E8A317", "#1B9E9E"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
ld = lambda run, f: np.load(os.path.join(D, run, f + ".npy")).astype(float)
L2, FE = "overthrust_l2_water_linear_init", "overthrust_feat_linear_init"


def chain(s1):
    """Stage-1 loss and error, stage-2 loss (one curve over both stage-2 runs) and error."""
    s2 = [s1 + "_refl", s1 + "_refl2"]
    return (ld(s1, "loss_history"), ld(s1, "vp_error_local_history"),
            np.concatenate([ld(r, "loss_history") for r in s2]),
            np.concatenate([ld(r, "vp_error_local_history")[1:] for r in s2]))


# Fig 15: velocity error only. The 3D L2 loss logs the misfit of the last shot evaluated (devitofwi L2._loss_value),
# not the total over shots, so the loss histories of the L2 runs are not plotted.
fig, axE = plt.subplots(figsize=(7.5, 4.6))
for run, c in ((L2, L2_C), (FE, SEISLM_C)):
    _, e1, _, e2 = chain(run)
    n1 = len(e1) - 1
    axE.plot(np.arange(len(e1) + len(e2)), np.concatenate([e1, e2]), color=c, lw=3)
    axE.plot(n1, e1[-1], "o", color=c, mfc="white", mew=2.2, ms=9, zorder=6)
axE.set_xlabel("accepted L-BFGS iteration", fontsize=15)
axE.set_ylabel("RMS relative velocity error", fontsize=14)
axE.spines["top"].set_visible(False)
axE.spines["right"].set_visible(False)
axE.tick_params(length=0)
h = [mlines.Line2D([], [], color=L2_C, lw=3), mlines.Line2D([], [], color=SEISLM_C, lw=3),
     mlines.Line2D([], [], marker="o", color="#555555", mfc="white", mew=2.2, ms=9, ls="none")]
axE.legend(h, ["$L_2$ workflow", "SeisLM workflow", "stage 2 start"], frameon=False, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_16_ovth_3d_perf.png"), dpi=170, bbox_inches="tight")

# Fig 16: final models of the two workflows, depth slice at nz/4 and the central inline and crossline sections
models = [("(a) $L_2$ workflow", "Stage 1: $L_2$, 17 iterations\nStage 2: $L_2$, 300 iterations", L2),
          ("(b) SeisLM workflow", "Stage 1: SeisLM, 100 iterations\nStage 2: $L_2$, 300 iterations", FE)]
nx, ny, nz = ld(L2 + "_refl2", "vp_current").shape
x, y, z = (np.arange(n) * 0.05 for n in (nx, ny, nz))
ix, iy, iz = nx // 2, ny // 2, nz // 4
Lx, Ly, Lz = x[-1], y[-1], z[-1]
fig = plt.figure(figsize=(2 * (Lx + Ly) * 0.95 + 1.6, (Ly + Lz) * 0.95 + 0.6))
outer = gridspec.GridSpec(1, 2, wspace=0.08)
plt.rcParams.update({"font.size": 20})
kw = dict(cmap="jet", vmin=1.5, vmax=6.0, aspect="auto", interpolation="nearest")
for k, (title, note, run) in enumerate(models):
    v = ld(run + "_refl2", "vp_current")
    err = ld(run + "_refl2", "vp_error_local_history")[-1]
    g = gridspec.GridSpecFromSubplotSpec(2, 2, outer[k], width_ratios=[Lx, Ly], height_ratios=[Ly, Lz],
                                         hspace=0.05, wspace=0.05)
    am, axz, acb = fig.add_subplot(g[0, 0]), fig.add_subplot(g[1, 0]), fig.add_subplot(g[0, 1])
    acb.axis("off")
    acb.text(0.06, 0.95, f"{note}\nError: {err:.3f}", transform=acb.transAxes, va="top", fontsize=22,
             linespacing=1.5)
    ayz = fig.add_subplot(g[1, 1], sharey=axz)
    am.imshow(v[:, :, iz].T, extent=[0, Lx, 0, Ly], origin="lower", **kw)
    am.axhline(y[iy], color="k", lw=1, ls="--")
    am.axvline(x[ix], color="m", lw=1, ls="--")
    am.set_ylabel("Crossline (km)")
    am.tick_params(labelbottom=False)
    am.set_title(title, fontsize=28, loc="left", pad=12)
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
cax = acb.inset_axes([0.08, 0.05, 0.08, 0.45])          # colorbar in the corner of panel (b)
cb = fig.colorbar(im, cax=cax)
cb.set_label("Velocity [km/s]", fontsize=20)
fig.savefig(os.path.join(OUT, "fig_3d_ovth_inv_lin.png"), dpi=150, bbox_inches="tight")
print("saved")
