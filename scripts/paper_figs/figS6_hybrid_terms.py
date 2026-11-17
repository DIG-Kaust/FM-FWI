# Supporting Information Fig S11 (Text S5, R1.1): the two normalized terms of the hybrid loss over stage 1, 2D Overthrust.
# Input: runs/hyb_trace_c (overthrust2d.py --loss hybrid --trace-terms 5): stage1.terms of run.json, or, when stage 1 did not
# finish (the paper run was stopped at stage-1 iteration 95), the "terms it" lines of a *.log file in that folder.
# Usage: python figS6_hybrid_terms.py <outdir>
import glob, json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.environ.get("FMFWI_RUNS", os.path.join(H, "../../runs")), "hyb_trace_c")
OUT = sys.argv[1]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbdbd",
                     "axes.linewidth": 1.0, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.9,
                     "xtick.color": "#555555", "ytick.color": "#555555", "font.size": 13})
L2_C, SEISLM_C, HYBRID_C = "#E8A317", "#1B9E9E", "#9B59B6"

pat = re.compile(r"terms it (\d+): alpha=([\d.]+) L2/S=([\d.e+-]+) F/S=([\d.e+-]+) \|gL2\|/\|gF\|=([\d.]+) "
                 r"L2 share=([\d.]+) cos=([\d.-]+)")
rec = json.load(open(os.path.join(RUN, "run.json"))) if os.path.exists(os.path.join(RUN, "run.json")) else {}
terms = rec.get("stage1", {}).get("terms")
if terms:
    it, alpha, jl2, jf, ratio, share, cos = np.array([[t[k] for k in ("iter", "alpha", "l2_norm", "feat_norm", "gnorm_ratio",
                                                                      "share_l2", "cos")] for t in terms]).T
else:
    log = "".join(open(f).read() for f in glob.glob(os.path.join(RUN, "*.log")))
    rows = sorted({tuple(map(float, m.groups())) for m in pat.finditer(log)})   # the log repeats lines
    it, alpha, jl2, jf, ratio, share, cos = np.array(rows).T

fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6))
a, b, c, d = axs.ravel()
a.semilogy(it, jl2, "o-", color=L2_C, lw=2.4, ms=5, label=r"$\bar{J}_{L_2}$")
a.semilogy(it, jf, "o-", color=SEISLM_C, lw=2.4, ms=5, label=r"$\bar{J}_{\mathrm{SeisLM}}$")
a.legend(frameon=False, fontsize=13)
b.semilogy(it, ratio, "o-", color=HYBRID_C, lw=2.4, ms=5)
b.axhline(1, color="#555555", lw=1, ls=(0, (2, 2)))
b.set_yscale("log", base=2)                        # powers of 2: evenly spaced ticks, each a doubling
b.set_yticks([0.25, 0.5, 1, 2, 4, 8])
b.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8"])
b.minorticks_off()
c.plot(it, share, "o-", color=HYBRID_C, lw=2.4, ms=5, label=r"$L_2$ share of the gradient")
c.plot(it, alpha, color="#555555", lw=1.6, ls=(0, (2, 2)), label=r"$\alpha_i$")
c.legend(frameon=False, fontsize=12, loc="upper left")
d.plot(it, cos, "o-", color=HYBRID_C, lw=2.4, ms=5)
d.axhline(0, color="#555555", lw=1, ls=(0, (2, 2)))
for ax, ttl, yl in ((a, "(a) Normalized terms", "normalized objective"),
                    (b, "(b) Gradient-norm ratio", r"$\|\nabla\bar{J}_{L_2}\| \,/\, \|\nabla\bar{J}_{\mathrm{SeisLM}}\|$"),
                    (c, r"(c) $L_2$ share of the gradient", "share"),
                    (d, "(d) Cosine between the two gradients", "cosine")):
    ax.set_title(ttl, fontsize=16, pad=10)
    ax.set_xlabel("accepted L-BFGS iteration", fontsize=14)
    ax.set_ylabel(yl, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figS6_hybrid_terms.png"), dpi=170, bbox_inches="tight")
for r in rows:
    if r[0] in (0, 5, 20, 25, 50, 75, 95):
        print(r)
