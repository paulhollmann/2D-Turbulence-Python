#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter


def vorticity_to_movie(input_file, output_file="vorticity.mp4", fps=30, dpi=150, cmap="bwr"):
    """
    Make a movie of the 2D vorticity field from a fluid.nc file.
    Colormap is centered at zero (white = 0).
    """
    print(f"[Vorticity] Dataset: {input_file}")
    # ---- open the data ----
    data = Dataset(input_file, 'r')
    w = data.variables['w'][:]       # shape: (t, y, x)
    t = data.variables['t'][:]
    alpha = data.variables['alpha'][:] if 'alpha' in data.variables else None

    nt, ny, nx = w.shape

    # ---- max abs value for symmetric colormap ----
    wmax_abs = np.max(np.abs(w))
    print(f"[Vorticity] symmetric color range: {-wmax_abs:.5e} → {wmax_abs:.5e}")

    # ---- set up figure ----
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(w[0], vmin=-wmax_abs, vmax=wmax_abs, cmap=cmap, origin='lower', aspect='auto')
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("vorticity")

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12,
                        bbox=dict(facecolor='black', alpha=0.5))
    alpha_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='white', fontsize=12,
                         bbox=dict(facecolor='black', alpha=0.5))

    # ---- update function ----
    def update(frame):
        img.set_data(w[frame])
        time_text.set_text(f"t = {t[frame]:.4f}")
        if alpha is not None:
            alpha_text.set_text(f"alpha = {alpha[frame]:.4f}")
        progress = (frame + 1) / nt * 100
        sys.stdout.write(f"\r[Vorticity] Rendering frame {frame+1}/{nt} ({progress:.1f}%)")
        sys.stdout.flush()
        return img, time_text, alpha_text

    anim = animation.FuncAnimation(fig, update, frames=nt, blit=True)

    # ---- save movie ----
    mpl.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # adjust path if needed

    try:
        writer = FFMpegWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=dpi)
        print(f"\n[Vorticity] Movie saved: {output_file}")
    except Exception as e:
        print("\n[Vorticity] FFmpeg failed, fallback to GIF:", e)
        gif_file = output_file.replace(".mp4", ".gif")
        anim.save(gif_file, writer=PillowWriter(fps=fps))
        print(f"[Vorticity] Saved GIF: {gif_file}")

    data.close()
    plt.close(fig)


def get_vorticity_minmax(input_file, target_time):
    """
    Returns (t_closest, min, max) of vorticity field at closest timestep.
    """
    data = Dataset(input_file, 'r')
    
    t = data.variables['t'][:]

    # ---- find closest timestep ----
    idx = np.argmin(np.abs(t - target_time))
    t_closest = t[idx]

    w_slice = data.variables['w'][idx, :, :]
    w_min = np.min(w_slice)
    w_max = np.max(w_slice)

    print(f"[MinMax] requested t={target_time:.4f}, using t={t_closest:.4f} (idx={idx})")
    print(f"[MinMax] min={w_min:.5e}, max={w_max:.5e}")

    data.close()
    return t_closest, w_min, w_max

def save_vorticity_snapshot_pub(input_file, target_time, output_path,
                                vmin=None, vmax=None,
                                cmap="RdBu_r",
                                dpi=300):
    """
    Save a publication-quality vorticity snapshot.
    """

    # ---- matplotlib styling (publication standard) ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "ps.fonttype": 42,   
        "pdf.fonttype": 42,
    })

    data = Dataset(input_file, 'r')
    
    w = data.variables['w']
    t = data.variables['t'][:]
    alpha = data.variables['alpha'][:] if 'alpha' in data.variables else None

    # ---- closest timestep ----
    idx = np.argmin(np.abs(t - target_time))
    t_closest = t[idx]
    w_slice = w[idx, :, :]   # efficient slicing

    print(f"[Snapshot] using t={t_closest:.4f} (idx={idx})")

    # ---- color scaling ----
    if vmin is None or vmax is None:
        wmax_abs = np.max(np.abs(w_slice))
        vmin, vmax = -wmax_abs, wmax_abs

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(5.5, 4.5))  # good for papers

    img = ax.imshow(
        w_slice,
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        origin='lower',
        aspect='auto',
        interpolation='none'  # avoid smoothing artifacts
    )

    # ---- colorbar ----
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label(r"Vorticity $\omega$")

    # ---- labels ----
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    # minimal but informative title
    title = rf"$t = {np.round(t_closest, 1)}$"
    if alpha is not None:
        title += rf", $\alpha = {alpha[idx]:.1f}$"
    ax.set_title(title)

    # ---- clean layout ----
    ax.tick_params(direction='in', top=True, right=True)
    
    # optional: remove if domain is not meaningful
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # ---- save ----
    plt.savefig(
        output_path,
        format="eps",
        dpi=dpi,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"[Snapshot] saved (publication): {output_path}")

    data.close()
