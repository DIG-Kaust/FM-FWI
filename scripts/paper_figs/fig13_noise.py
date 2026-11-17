# Paper figure: 2D Overthrust from the linear start with flat in-band noise (1-11 Hz) at 20, 10 and
# 5 dB, without and with the noise before the first arrival muted (oracle pick on the clean data).
# (a, b) Error after stage 1 and after stage 2 against SNR; (c) error against accepted iterations at 5 dB.
# Inputs: run.json of runs/{noise,nmute}_flat{20,10,5}_{feature,l2} and ref_{feature,l2} (noise-free).
# Usage: python fig13_noise.py <outdir>
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.environ.get("FMFWI_RUNS", os.path.join(REPO, "runs"))   # driver outputs, see REPRODUCE.md

H = os.path.dirname(os.path.abspath(__file__))
D, OUT = RUNS, sys.argv[1]
L2_C, SEISLM_C = "#E8A317", "#1B9E9E"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
run = lambda n: json.load(open(os.path.join(D, n, "run.json")))
SNR = [20, 10, 5]
fig, (a1, a, b) = plt.subplots(1, 3, figsize=(18, 5.0))
a1.sharey(a)
x = np.arange(len(SNR) + 1)                      # noise-free, then decreasing SNR
for loss, c in (("feature", SEISLM_C), ("l2", L2_C)):
    for pre, ls, mk in (("noise", "--", "s"), ("nmute", "-", "o")):
        for ax, st in ((a1, "stage1"), (a, "stage2")):
            e = [run(f"ref_{loss}")[st]["err"][-1]] + [run(f"{pre}_flat{s}_{loss}")[st]["err"][-1] for s in SNR]
            ax.plot(x, e, color=c, ls=ls, marker=mk, lw=2.6, ms=8, mfc="white" if pre == "noise" else c, mew=2)
        j = run(f"{pre}_flat5_{loss}")
        e1, e2 = j["stage1"]["err"], j["stage2"]["err"]
        b.plot(np.arange(len(e1) + len(e2) - 1), e1 + e2[1:], color=c, ls=ls, lw=2.6)
        b.plot(len(e1) - 1, e1[-1], "o", color=c, mfc="white", mew=2, ms=7, zorder=6)
for ax in (a1, a, b):
    ax.axhline(0.172, color="#555555", lw=1, ls=(0, (2, 2)))
for ax in (a1, a):
    ax.set_xticks(x, ["no noise"] + [f"{s} dB" for s in SNR])
for ax, ttl, xl in ((a1, "(a) Error after stage 1", "signal-to-noise ratio"),
                    (a, "(b) Error after stage 2", "signal-to-noise ratio"),
                    (b, "(c) Convergence at 5 dB", "accepted L-BFGS iteration")):
    ax.set_title(ttl, fontsize=16, pad=10)
    ax.set_xlabel(xl, fontsize=14)
    ax.set_ylabel("RMS relative velocity error", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
h = [mlines.Line2D([], [], color=SEISLM_C, lw=2.6), mlines.Line2D([], [], color=L2_C, lw=2.6),
     mlines.Line2D([], [], color="#555555", lw=2.6, ls="--"), mlines.Line2D([], [], color="#555555", lw=2.6),
     mlines.Line2D([], [], marker="o", color="#555555", mfc="white", mew=2, ms=7, ls="none")]
fig.legend(h, ["SeisLM", "$L_2$", "noise not muted", "noise before first arrival muted", "stage 2 start"],
           loc="lower center", ncol=5, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(os.path.join(OUT, "fig_noise.png"), dpi=170, bbox_inches="tight")
print("saved")
