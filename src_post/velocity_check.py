#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, PillowWriter

def vorticity_check_movie(input_file, output_file="vorticity_check.mp4", fps=30, dpi=150, cmap="bwr"):
    """
    Create a movie comparing:
    - given vorticity
    - computed vorticity from (u,v)
    - error field

    Also shows L2 and Linf errors over time.
    """
    print(f"[Check] Dataset: {input_file}")
    # ---- load data ----
    data = Dataset(input_file, 'r')

    u = data.variables['u'][:]   # (t, y, x)
    v = data.variables['v'][:]
    w = data.variables['w'][:]
    t = data.variables['t'][:] if 't' in data.variables else None

    nt, ny, nx = u.shape

    dx = 2*np.pi / nx
    dy = 2*np.pi / ny

    print(f"[Check] Data shape: nt={nt}, ny={ny}, nx={nx}")

    # ---- compute vorticity (finite difference) ----
    def compute_vorticity(u, v):
        dvdx = np.zeros_like(v)
        dudy = np.zeros_like(u)

        dvdx[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) / (2 * dx)
        dudy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy)

        dvdx[:, :, 0]  = (v[:, :, 1] - v[:, :, 0]) / dx
        dvdx[:, :, -1] = (v[:, :, -1] - v[:, :, -2]) / dx

        dudy[:, 0, :]  = (u[:, 1, :] - u[:, 0, :]) / dy
        dudy[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / dy

        return dvdx - dudy

    w_calc = compute_vorticity(u, v)

    # ---- error ----
    error = w_calc - w

    # ---- error metrics over time ----
    l2_t = np.sqrt(np.mean(error**2, axis=(1,2)))
    linf_t = np.max(np.abs(error), axis=(1,2))

    print(f"[Check] Global L2   = {np.mean(l2_t):.6e}")
    print(f"[Check] Global Linf = {np.max(linf_t):.6e}")

    # ---- color limits ----
    wmax = np.max(np.abs(w))
    wcalc_max = np.max(np.abs(w_calc))
    err_max = np.max(np.abs(error))

    # ---- figure ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axs[0].imshow(w[0], origin='lower', cmap=cmap, vmin=-wmax, vmax=wmax)
    axs[0].set_title("Given vorticity")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(w_calc[0], origin='lower', cmap=cmap, vmin=-wcalc_max, vmax=wcalc_max)
    axs[1].set_title("Computed vorticity")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(error[0], origin='lower', cmap=cmap, vmin=-err_max, vmax=err_max)
    axs[2].set_title("Error")
    plt.colorbar(im2, ax=axs[2])

    # ---- overlay text ----
    text = fig.text(0.02, 0.95, '', color='black', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))

    # ---- update ----
    def update(frame):
        im0.set_data(w[frame])
        im1.set_data(w_calc[frame])
        im2.set_data(error[frame])

        time_str = f"t = {t[frame]:.4f}" if t is not None else f"frame = {frame}"

        text.set_text(
            f"{time_str}\n"
            f"L2   = {l2_t[frame]:.3e}\n"
            f"Linf = {linf_t[frame]:.3e}"
        )

        progress = (frame + 1) / nt * 100
        sys.stdout.write(f"\r[Check] Rendering frame {frame+1}/{nt} ({progress:.1f}%)")
        sys.stdout.flush()

        return im0, im1, im2, text

    anim = animation.FuncAnimation(fig, update, frames=nt, blit=False)

    # ---- save ----
    mpl.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"

    try:
        writer = FFMpegWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=dpi)
        print(f"\n[Check] Movie saved: {output_file}")
    except Exception as e:
        print("\n[Check] FFmpeg failed, fallback to GIF:", e)
        gif_file = output_file.replace(".mp4", ".gif")
        anim.save(gif_file, writer=PillowWriter(fps=fps))
        print(f"[Check] Saved GIF: {gif_file}")

    data.close()
    plt.close(fig)