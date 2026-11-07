# Supporting Information Fig S4 (Text S3): sampling interval and record length, rigid-shift test on one masked
# 2D Overthrust shot, SeisLM full sequence, last transformer block.
# Inputs: runs/diagnostics/{sampling,window}.npz (scripts/shift_diagnostics.py sampling / window),
# written to runs/diagnostics. Usage: python figS4_sampling.py <outdir>
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md

H = os.path.dirname(os.path.abspath(__file__))
D, OUT = os.path.join(RUNS, "diagnostics"), sys.argv[1]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
s, w = np.load(os.path.join(D, "sampling.npz")), np.load(os.path.join(D, "window.npz"))
norm = lambda v: (v - v.min()) / (v.max() - v.min())

fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
for i, d in enumerate(s["dts_s"]):
    a.plot(s["shifts_s"] / s["period_s"], norm(s[f"full_{d * 1e3:g}ms"][-1]), lw=2.4,
           color=plt.cm.viridis(i / (len(s["dts_s"]) - 1)), label=f"{d * 1e3:g} ms")
for i, T in enumerate(w["windows_s"]):
    b.plot(w["shifts_s"] / w["period_s"], w[f"full_{T:g}s"][-1], lw=2.4,
           color=plt.cm.viridis(i / (len(w["windows_s"]) - 1)), label=f"{T:g} s")
for ax, ttl, yl, lt in ((a, "(a) Sampling interval (4-s record)", "Normalized feature distance", "sampling interval"),
                        (b, "(b) Record length (native 2.31-ms sampling)", "Feature distance", "record length")):
    ax.set_title(ttl, fontsize=16, pad=10)
    ax.set_xlabel("time shift (dominant periods of the 5-Hz wavelet)", fontsize=13)
    ax.set_ylabel(yl, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=11, title=lt, title_fontsize=11, loc="upper center")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figS4_sampling_window.png"), dpi=170, bbox_inches="tight")
print("saved")
