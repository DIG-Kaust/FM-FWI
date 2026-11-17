# Paper Figs 16-17: 3D Overthrust, L2 and SeisLM workflows, both with the same stage-2 L2 refinement.
# Inputs: runs/3d/<name> of scripts/overthrust3d.py (vp_current.npy, vp_error_local_history.npy).
# L2 workflow: stage 1 masks both observed and modeled data (overthrust_l2mask_linear_init, 100 it; stage 2 _refl /
#   _refl2, 100 + 200 it). SeisLM workflow: time-averaged features, stage 1 overthrust_feat_linear_init (100 it)
#   and stage 2 overthrust_feat_linear_init_refl/2 (100 + 200 it).
# Fig 17 is drawn like Fig 15 (fm_fwi.plotting.plot_slices per model, colour scale 1.5-6 km/s, one panel per model).
# Usage: python fig16_17_overthrust3d.py <outdir>
import importlib.util, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(H, "../../runs"))   # driver outputs, see REPRODUCE.md
D, OUT = os.path.join(RUNS, "3d"), sys.argv[1]
spec = importlib.util.spec_from_file_location("plotting", os.path.join(H, "../../fm_fwi/plotting.py"))
plotting = importlib.util.module_from_spec(spec); spec.loader.exec_module(plotting)
L2_C, SEISLM_C = "#E8A317", "#1B9E9E"
ld = lambda run, f: np.load(os.path.join(D, run, f + ".npy")).astype(float)
L2, FE = "overthrust_l2mask_linear_init", "overthrust_feat_linear_init"

# Fig 16: relative velocity error, stage 2 continuing after the last stage-1 iterate
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
fig, ax = plt.subplots(figsize=(7.5, 4.6))
# Velocity error only: the 3D runs of the SeisLM workflow logged the last shot's L2 misfit, not the total, so
# their stage-2 loss cannot be shown.
e1 = ld(L2, "vp_error_local_history")
e2 = np.concatenate([ld(L2 + s, "vp_error_local_history")[1:] for s in ("_refl", "_refl2")])
ax.plot(np.arange(len(e1) + len(e2)), np.concatenate([e1, e2]), color=L2_C, lw=3)
ax.plot(len(e1) - 1, e1[-1], "o", color=L2_C, mfc="white", mew=2.2, ms=9, zorder=6)
f1 = ld(FE, "vp_error_local_history")
f2 = np.concatenate([ld(FE + s, "vp_error_local_history")[1:] for s in ("_refl", "_refl2")])
ax.plot(np.arange(len(f1) + len(f2)), np.concatenate([f1, f2]), color=SEISLM_C, lw=3)
ax.plot(len(f1) - 1, f1[-1], "o", color=SEISLM_C, mfc="white", mew=2.2, ms=9, zorder=6)
ax.set_xlabel("accepted L-BFGS iteration", fontsize=15)
ax.set_ylabel("RMS relative velocity error", fontsize=14)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
h = [mlines.Line2D([], [], color=L2_C, lw=3), mlines.Line2D([], [], color=SEISLM_C, lw=3),
     mlines.Line2D([], [], marker="o", color="#555555", mfc="white", mew=2.2, ms=9, ls="none")]
ax.legend(h, ["$L_2$ workflow", "SeisLM workflow", "stage 2 start"], frameon=False, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_16_ovth_3d_perf.png"), dpi=170, bbox_inches="tight")
plt.close(fig)

# Fig 17: one plot_slices panel per model (as Fig 15), then side by side under the panel titles
panels = [("(a) $L_2$ workflow", L2 + "_refl2"), ("(b) SeisLM workflow", FE + "_refl2")]
plt.rcParams.update(plt.rcParamsDefault)
imgs = []
for title, run in panels:
    v = ld(run, "vp_current")
    nx, ny, nz = v.shape
    x, y, z = (np.arange(n) * 0.05 for n in (nx, ny, nz))
    err = ld(run, "vp_error_local_history")[-1]
    p = os.path.join(OUT, f"_fig17_{run}.png")
    f, _ = plotting.plot_slices(v, x, y, z, vmin=1.5, vmax=6.0, title=f"Error: {err:.3f}", savepath=p, dpi=150)
    plt.close(f)
    imgs.append((title, plt.imread(p)))
    os.remove(p)
hgt = max(im.shape[0] for _, im in imgs)
fig, axs = plt.subplots(1, 2, figsize=(2 * imgs[0][1].shape[1] / 150, hgt / 150 + 0.9),
                        gridspec_kw=dict(wspace=0.03))
for a, (title, im) in zip(axs, imgs):
    a.imshow(im)
    a.axis("off")
    a.set_title(title, fontsize=30, pad=10)
fig.savefig(os.path.join(OUT, "fig_3d_ovth_inv_lin.png"), dpi=150, bbox_inches="tight")
print("saved")
