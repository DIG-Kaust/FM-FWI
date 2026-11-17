# Supporting Information Fig S4 (Text S3): sampling interval and record length, rigid-shift test on one masked
# 2D Overthrust shot, SeisLM full sequence (solid) and time average (dashed), last
# transformer block; every curve normalized to 0-1.
# Inputs: runs/diagnostics/{sampling,window}.npz (scripts/shift_diagnostics.py sampling / window).
# Usage: python figS4_sampling.py <outdir>
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(H, "../../runs"))   # driver outputs, see REPRODUCE.md
D, OUT = os.path.join(RUNS, "diagnostics"), sys.argv[1]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
s, w = np.load(os.path.join(D, "sampling.npz")), np.load(os.path.join(D, "window.npz"))
norm = lambda v: (v - v.min()) / (v.max() - v.min())

fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
for i, d in enumerate(s["dts_s"]):
    c = plt.cm.viridis(i / (len(s["dts_s"]) - 1))
    a.plot(s["shifts_s"] / s["period_s"], norm(s[f"full_{d * 1e3:g}ms"][-1]), lw=2.4, color=c, label=f"{d * 1e3:g} ms")
    a.plot(s["shifts_s"] / s["period_s"], norm(s[f"mean_{d * 1e3:g}ms"][-1]), lw=2.0, color=c, ls="--")
for i, T in enumerate(w["windows_s"]):
    c = plt.cm.viridis(i / (len(w["windows_s"]) - 1))
    b.plot(w["shifts_s"] / w["period_s"], norm(w[f"full_{T:g}s"][-1]), lw=2.4, color=c, label=f"{T:g} s")
    b.plot(w["shifts_s"] / w["period_s"], norm(w[f"mean_{T:g}s"][-1]), lw=2.0, color=c, ls="--")
for ax, ttl, yl, lt in ((a, "(a) Sampling interval (4-s record)", "Normalized feature distance", "sampling interval"),
                        (b, "(b) Record length (native 2.31-ms sampling)", "Normalized feature distance", "record length")):
    ax.set_title(ttl, fontsize=16, pad=10)
    ax.set_xlabel("time shift (dominant periods of the 5-Hz wavelet)", fontsize=13)
    ax.set_ylabel(yl, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_ylim(-0.03, 1.38); ax.set_yticks(np.arange(0, 1.01, 0.2))   # room for the legend above the curves
    h, l = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], color="#555555", lw=2.4), plt.Line2D([], [], color="#555555", lw=2.0, ls="--")]
    ax.legend(h, l + ["full sequence", "time average"], frameon=False, fontsize=10, title=lt, title_fontsize=11,
              loc="upper center", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figS4_sampling_window.png"), dpi=170, bbox_inches="tight")
print("saved")
