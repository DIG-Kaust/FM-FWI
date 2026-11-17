# SI Figure S6: established misfits from the true model smoothed over 1.25 km (sigma 50, 40 cells,
# error 0.109). Runs runs/{init_sm50_feature, init_sm50_l2, bl_stdl2_sm50, bl_envelope_p1_sm50, bl_w2_exact_sm50,
# bl_multiscale_pad_sm50, sdtw_g30_sm50} (overthrust2d.py, S1 100 it masked + S2 100 it L2).
# (a) relative velocity error, stage 2 continuing after the last accepted stage-1 iterate as in fig12_baselines.py;
# (b-g) starting model and stage-2 models, colour scale of Fig 11. Usage: python figS6_smooth_baselines.py <outdir>
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]
D = os.environ.get("FMFWI_RUNS", os.path.join(H, "../../runs"))   # driver outputs, see REPRODUCE.md
RUNS = [("$L_2$", "#E8A317", "init_sm50_l2"), ("SeisLM", "#1B9E9E", "init_sm50_feature"),
        ("$L_2$, standardized traces", "#7F7F7F", "bl_stdl2_sm50"),
        ("Envelope", "#D62728", "bl_envelope_p1_sm50"), ("W2", "#3B5BA5", "bl_w2_exact_sm50"),
        ("Frequency continuation", "#2CA02C", "bl_multiscale_pad_sm50"), ("Soft-DTW", "#8E44AD", "sdtw_g30_sm50")]
J = lambda r: json.load(open(os.path.join(D, r, "run.json")))
M = lambda r, f: np.load(os.path.join(D, r, f + ".npy"))
plt.rcParams.update({"axes.edgecolor": "#bdbdbd", "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
NC = 4   # 2 x 4 model panels: the start and the seven runs
fig = plt.figure(figsize=(26, 14))
gs = fig.add_gridspec(4, NC, height_ratios=[1.25, 1, 1, 0.08], hspace=0.45, wspace=0.12)
ax = fig.add_subplot(gs[0, :])
for lab, c, r in RUNS:
    j = J(r); e1, e2 = j["stage1"]["err"], j["stage2"]["err"]
    ax.plot(np.arange(len(e1)), e1, color=c, lw=2.5, label=f"{lab}: {e1[-1]:.3f} / {e2[-1]:.3f}")
    ax.plot(len(e1) - 1 + np.arange(len(e2)), e2, color=c, lw=2.5)
    ax.plot(len(e1) - 1, e2[0], "o", color=c, mfc="white", mew=2, ms=7, zorder=6)
ax.axhline(J(RUNS[0][2])["stage1"]["err"][0], color="#555555", lw=1, ls=(0, (2, 2)))
ax.set_title("(a) Relative velocity error", fontsize=18); ax.grid(color="#dddddd")
ax.set_xlabel("accepted L-BFGS iteration", fontsize=15); ax.set_ylabel("RMS relative velocity error", fontsize=14)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=13, title="error after stage 1 / stage 2", title_fontsize=13, ncol=1, loc="upper right")
x, z = np.arange(501) * 0.025, np.arange(M(RUNS[0][2], "vp_init").shape[1]) * 0.025
panels = [("start", M(RUNS[0][2], "vp_init"))] + [(lab, M(r, "vp_inv_refl")) for lab, _, r in RUNS]
for k, (lab, m) in enumerate(panels):
    a = fig.add_subplot(gs[1 + k // NC, k % NC])
    im = a.imshow(m.T, vmin=1.5, vmax=6.0, cmap="jet", extent=(x[0], x[-1], z[-1], z[0]), aspect="auto")
    t = "starting model" if k == 0 else f"{lab}, stage 2"
    a.set_title(f"({'bcdefghi'[k]}) {t}", fontsize=16); a.set_xlabel("X [km]", fontsize=13); a.set_ylabel("Z [km]", fontsize=13)
    a.tick_params(colors="black", labelsize=12)
    for s in a.spines.values(): s.set_edgecolor("black")
cax = fig.add_subplot(gs[3, 1:3])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[2, 3, 4, 5, 6])
cb.set_label("Velocity [km/s]", fontsize=16); cb.ax.tick_params(labelsize=14, colors="black")
fig.savefig(os.path.join(OUT, "figS6_smooth_baselines.png"), dpi=130, bbox_inches="tight")
print("saved")
