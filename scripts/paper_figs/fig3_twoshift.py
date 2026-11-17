# Paper Fig 3: two-shift test from runs/diagnostics/boa1d/boa1d.npz (scripts/boa1d.py).
# SeisLM full sequence, SeisLM time-averaged and a random encoder.
#   python fig3_twoshift.py <boa1d.npz> <out.png>   (maps show the distance, the square root of the stored values)
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(sys.argv[1])
out = sys.argv[2]
plt.rcParams.update({"font.size": 10})
ms = d["shifts_s"] * 1e3
n, dt = len(ms), float(d["dt"])
norm = lambda v: (lambda w: (w - w.min()) / (w.max() - w.min()))(np.sqrt(np.maximum(v, 0)))
last = lambda k, pool: d[f"{k}_std_grid"][-1, pool].reshape(n, n)
full = last("pretrained", 0)
# Standardized traces, as the loss feeds the encoder. The time-averaged distance (pool index 1) is ~1e-8 of the
# full-sequence one on this grid (shifts are whole multiples of the 4-sample stride), so it is not plotted.
panels = [(r"$L_2$", norm(d["bl_l2_grid"].reshape(n, n))),
          ("Pretrained SeisLM", norm(full)),
          ("Randomly initialized encoder", norm(last("random0", 0)))]


def basin(g):
    """Grid points from which steepest descent over the 8 neighbours ends at (0, 0)."""
    p = np.pad(g, 1, constant_values=np.inf)
    nb = np.stack([p[1 + a:1 + a + n, 1 + b:1 + b + n] for a in (-1, 0, 1) for b in (-1, 0, 1)])
    k = nb.argmin(0)                                          # index 4 = stay (local minimum)
    i, j = np.indices((n, n))
    nxt = ((i + k // 3 - 1) * n + (j + k % 3 - 1)).ravel()
    for _ in range(12):                                       # pointer doubling: 2^12 steps > any path
        nxt = nxt[nxt]
    c = n // 2
    return np.isin(nxt, [(c + a) * n + c + b for a in (-1, 0, 1) for b in (-1, 0, 1)]).reshape(n, n)


fig = plt.figure(figsize=(12, 7.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.8], width_ratios=[1, 1, 1, 0.05], hspace=0.5, wspace=0.2, bottom=0.14)
for i, (name, v) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(v.T, origin="lower", extent=[ms[0], ms[-1]] * 2, cmap="RdBu_r", vmin=0, vmax=1)
    bas = basin(v)
    ax.contour(ms, ms, bas.T.astype(float), levels=[0.5], colors="k", linewidths=1.6)
    print(f"{name}: basin {bas.mean() * 100:.1f}% of the grid")
    ax.set_title(f"({'abc'[i]}) {name}", fontsize=10.5)
    ax.set_xlabel(r"Time shift of event 1, $\Delta t_1$ (ms)")
    ax.set_xticks([-800, -400, 0, 400, 800])
    ax.set_yticks([-800, -400, 0, 400, 800])
    if i == 0:
        ax.set_ylabel(r"Time shift of event 2, $\Delta t_2$ (ms)")
    else:
        ax.set_yticklabels([])
fig.colorbar(im, cax=fig.add_subplot(gs[0, 3])).set_label("Normalized misfit")
# (d, e) cuts through the maps: one event shifted (dt2 = 0), and both shifted together (dt1 = dt2)
cut = gs[1, 0:3].subgridspec(1, 2, wspace=0.18)
c = n // 2
for k, (ttl, take) in enumerate(((r"(d) Cut at $\Delta t_2$ = 0", lambda v: v[:, c]),
                                 (r"(e) Cut at $\Delta t_1 = \Delta t_2$", lambda v: np.diag(v)))):
    ax = fig.add_subplot(cut[k])
    for (name, v), col, ls in zip(panels, ("#E8A317", "#1B9E9E", "#555555"), (":", "-", "--")):
        ax.plot(ms, take(v), color=col, ls=ls, lw=2.2, label=name)
    ax.set(xlim=(ms[0], ms[-1]), ylim=(0, 1.02), title=ttl, xlabel=r"Time shift $\Delta t_1$ (ms)")
    ax.set_xticks([-800, -400, 0, 400, 800])
    ax.grid(alpha=0.3)
    if k == 0:
        ax.set_ylabel("Normalized misfit")
h, lab = ax.get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.46, 0.035))
fig.savefig(out, dpi=300, bbox_inches="tight")
