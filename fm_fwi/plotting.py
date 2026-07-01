"""Shared plotting helpers for the SeisLM Marmousi / Overthrust FWI notebooks."""

import numpy as np
import matplotlib.pyplot as plt


def apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor":   "#bdbdbd", "axes.linewidth": 1.0,
        "axes.grid":        True,      "grid.color":     "#dddddd",
        "grid.linewidth":   0.9,       "xtick.color":    "#555555",
        "ytick.color":      "#555555", "font.size":      13,
    })


def plot_velmodel(vp, title, outpath, x, z, vmin=None, vmax=None,
                   cmap="jet", clbl="Velocity [km]/s", xtick_step=1):
    if vmin is None or vmax is None:
        pvmin, pvmax = np.percentile(vp, [2, 98])
        vmin = pvmin if vmin is None else vmin
        vmax = pvmax if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(vp.T, vmin=vmin, vmax=vmax, cmap=cmap,
                    extent=(x[0], x[-1], z[-1], z[0]))
    ax.set_xlabel("X [km]", fontsize=16)
    ax.set_ylabel("Z [km]", fontsize=16)
    ax.set_title(title, fontsize=24)
    ax.axis("tight")
    ax.set_xticks(np.arange(0, x[-1] + 0.1, xtick_step))
    ax.set_yticks(np.arange(0, z[-1] + 0.1, 1))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
    ax.tick_params(colors="black", labelsize=18)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(clbl, fontsize=16)
    cb.ax.tick_params(labelsize=14, colors="black")
    cb.outline.set_edgecolor("black")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    print("saved", outpath)


def plot_shotgathers(get_image, ishots, x_r, tmax, title, outpath=None,
                      vmin=None, vmax=None, cmap="gray",
                      directtime=None, toff=0.):
    extent = (x_r[0, 0], x_r[-1, 0], tmax, 0.)
    images = [get_image(ishot) for ishot in ishots]
    if vmin is None or vmax is None:
        pvmin, pvmax = np.percentile(
            np.concatenate([img.ravel() for img in images]), [2, 98])
        vmin = pvmin if vmin is None else vmin
        vmax = pvmax if vmax is None else vmax

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(14, 9))
    for ax, ishot, img in zip(axs, ishots, images):
        ax.imshow(img, aspect="auto", cmap=cmap, extent=extent,
                  vmin=vmin, vmax=vmax)
        if directtime is not None:
            ax.plot(x_r[:, 0], directtime[ishot] + toff, "r")
        ax.set_xlabel("Offset [km]", fontsize=14)
        ax.set_title(f"Shot {ishot}", fontsize=14)
        ax.set_ylim(tmax, 0)
        ax.grid(False)
        ax.tick_params(colors="black", labelsize=12)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
    axs[0].set_ylabel("Time [s]", fontsize=14)
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_curve(data, title, ylabel, outpath, data2=None, xlabel="iteration"):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    line, = ax.plot(np.arange(len(data)), data, lw=3, label="stage 1")
    color = line.get_color()
    if data2 is not None:
        n1 = len(data)
        x2 = np.arange(n1, n1 + len(data2))
        ax.plot(x2, data2, lw=3, ls="--", color=color, label="stage 2")
        ax.plot(n1, data2[0], "o", color=color, mfc="white", mew=2.2, ms=9, zorder=6)
        ax.legend(frameon=False, fontsize=12)
    ax.set_title(title, fontsize=17, pad=10)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(outpath, dpi=170, bbox_inches="tight")
    plt.show()
    print("saved", outpath)
