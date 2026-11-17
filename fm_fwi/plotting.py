"""Shared plotting helpers for the Marmousi and Overthrust FWI drivers."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor":   "#bdbdbd", "axes.linewidth": 1.0,
        "axes.grid":        True,      "grid.color":     "#dddddd",
        "grid.linewidth":   0.9,       "xtick.color":    "#555555",
        "ytick.color":      "#555555", "font.size":      13,
    })


def add_slice_location_cube(
    fig, bbox,
    Lx, Ly, Lz,
    fx, fy, fz,
    elev=20, azim=-60,
    face_alpha=0.5,
    edge_lw=1.2,
    colors=('magenta', '#000033', '0.4'),
    label_fontsize=9,
):
    """Add a small 3D wireframe cube inset showing slice locations.

    bbox   : (left, bottom, width, height) in figure coordinates (0–1)
    fx/fy/fz : fractional slice positions along each axis (0–1)
    colors : (x-plane, y-plane, z-plane) colours, matching the crosshair lines
    """
    ax3d = fig.add_axes(bbox, projection='3d')

    corners = np.array([[0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],
                        [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        ax3d.plot(*zip(corners[i], corners[j]), color='dimgray', lw=0.8)

    x_pos = fx * Lx
    poly = Poly3DCollection(
        [[[x_pos,0,0],[x_pos,Ly,0],[x_pos,Ly,Lz],[x_pos,0,Lz]]],
        facecolor=colors[0], alpha=face_alpha, edgecolor=colors[0], linewidth=edge_lw)
    ax3d.add_collection3d(poly)

    y_pos = fy * Ly
    poly = Poly3DCollection(
        [[[0,y_pos,0],[Lx,y_pos,0],[Lx,y_pos,Lz],[0,y_pos,Lz]]],
        facecolor=colors[1], alpha=face_alpha, edgecolor=colors[1], linewidth=edge_lw)
    ax3d.add_collection3d(poly)

    z_pos = fz * Lz
    poly = Poly3DCollection(
        [[[0,0,z_pos],[Lx,0,z_pos],[Lx,Ly,z_pos],[0,Ly,z_pos]]],
        facecolor=colors[2], alpha=face_alpha, edgecolor=colors[2], linewidth=edge_lw)
    ax3d.add_collection3d(poly)

    ax3d.set_xlim(0, Lx); ax3d.set_ylim(0, Ly); ax3d.set_zlim(0, Lz)
    ax3d.invert_zaxis()
    ax3d.set_box_aspect((Lx, Ly, Lz))
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_axis_off()

    ax3d.text(Lx/2,    0,      Lz*1.05, 'Inline',     fontsize=label_fontsize, ha='center', va='top')
    ax3d.text(Lx*1.05, Ly/2,   Lz*1.05, 'Crossline',  fontsize=label_fontsize, ha='left',   va='center')
    ax3d.text(0,       0,      Lz/2,    'Depth',       fontsize=label_fontsize, ha='right',  va='center', rotation=90)
    return ax3d


def plot_slices(
    vp, x, y, z,
    ix=None, iy=None, iz=None,
    vmin=None, vmax=None,
    cmap='jet',
    title=None,
    cbar_label='Velocity [km/s]',
    scale=1.2,
    savepath=None,
    dpi=300,
    tick_labelsize=22,
    cbar_labelsize=22,
):
    """Plot three orthogonal slices of a 3D model in a corner layout.

    Parameters
    ----------
    vp : np.ndarray, shape (nx, ny, nz)
    x, y, z : coordinate axes in km
    ix, iy, iz : slice indices (default: mid / quarter)
    vmin, vmax : colour limits (default: model min/max)
    tick_labelsize, cbar_labelsize : font sizes for ticks and colour bar
    """
    nx, ny, nz = vp.shape
    ix = ix if ix is not None else nx // 2
    iy = iy if iy is not None else ny // 2
    iz = iz if iz is not None else nz // 4
    vmin = vmin if vmin is not None else float(vp.min())
    vmax = vmax if vmax is not None else float(vp.max())

    Lx = x[-1] - x[0]; Ly = y[-1] - y[0]; Lz = z[-1] - z[0]
    fig_w = (Lx + Ly) * scale + 1.5
    fig_h = (Ly + Lz) * scale

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[Lx, Ly], height_ratios=[Ly, Lz],
                           hspace=0.04, wspace=0.04)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_cb  = fig.add_subplot(gs[0, 1])
    ax_xz  = fig.add_subplot(gs[1, 0], sharex=ax_map)
    ax_yz  = fig.add_subplot(gs[1, 1], sharey=ax_xz)

    imkw = dict(cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')

    ax_map.imshow(vp[:, :, iz].T, **imkw,
                  extent=[x[0], x[-1], y[0], y[-1]], origin='lower')
    ax_map.axhline(y[iy], color='#000033', lw=2, ls='--')
    ax_map.axvline(x[ix], color='magenta',  lw=2, ls='--')
    ax_map.set_ylabel('Crossline (km)', fontsize=22)
    ax_map.tick_params(labelbottom=False, labelsize=tick_labelsize)

    ax_xz.imshow(vp[:, iy, :].T, **imkw,
                 extent=[x[0], x[-1], z[-1], z[0]])
    ax_xz.axvline(x[ix], color='magenta', lw=2, ls='--')
    ax_xz.axhline(z[iz], color='0.4',     lw=2, ls='--')
    ax_xz.set_xlabel('Inline (km)', fontsize=22)
    ax_xz.set_ylabel('Depth (km)',  fontsize=22)
    ax_xz.tick_params(labelsize=tick_labelsize)

    im = ax_yz.imshow(vp[ix, :, :].T, **imkw,
                      extent=[y[0], y[-1], z[-1], z[0]])
    ax_yz.axvline(y[iy], color='#000033', lw=2, ls='--')
    ax_yz.axhline(z[iz], color='0.4',     lw=2, ls='--')
    ax_yz.set_xlabel('Crossline (km)', fontsize=22)
    ax_yz.tick_params(labelleft=False, labelsize=tick_labelsize)

    ax_cb.set_axis_off()
    cbar = fig.colorbar(im, ax=ax_cb, fraction=0.5, pad=0.05, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=cbar_labelsize)
    cbar.ax.tick_params(labelsize=cbar_labelsize)
    if title:
        ax_cb.text(0.05, 0.98, title, transform=ax_cb.transAxes,
                   fontsize=24, va='top', ha='left', fontweight='bold')

    pos = ax_cb.get_position()
    cube_bbox = [pos.x0 - 0.02, pos.y0, 0.18, 0.18]
    add_slice_location_cube(
        fig, cube_bbox, Lx, Ly, Lz,
        (x[ix]-x[0])/Lx, (y[iy]-y[0])/Ly, (z[iz]-z[0])/Lz,
        colors=('magenta', '#000033', '0.4'),
    )

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    return fig, (ax_map, ax_xz, ax_yz)
