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