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



def compute_pdf2(input_file, bins=50, shift=(1, 0)):
    """
    Compute 2-point joint PDF of vorticity from a fluid.nc file.

    Stores BOTH:
    - bin centers (recommended for plotting)
    - bin edges (for reproducibility)
    
    Parameters
    ----------
    input_file : str
        Path to input netCDF file (fluid.nc)
    bins : int
        Number of bins along each axis
    shift : tuple
        Spatial shift (dx, dy) for 2-point PDF
    """

    print(f"[PDF2] opening {input_file} ...")
    data = Dataset(input_file, 'r')
    w = data.variables['w'][:]  # (t, y, x)
    t = data.variables['t'][:]
    nt = w.shape[0]

    # ---- global min/max over all time ----
    wmin = np.min(w)
    wmax = np.max(w)
    print(f"[PDF2] global min = {wmin:.5e}, max = {wmax:.5e}")

    # ---- bin edges + centers ----
    edges = np.linspace(wmin, wmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # ---- output file ----
    dirname = os.path.dirname(input_file)
    basename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(dirname, basename + ".pdf2.nc")

    out = Dataset(output_file, 'w', 'NETCDF4')

    # ---- dimensions ----
    out.createDimension('t', None)
    out.createDimension('b', bins)
    out.createDimension('b_edges', bins + 1)

    # ---- variables ----
    t_out = out.createVariable('t', 'float32', ('t',))
    pdf = out.createVariable('pdf', 'float32', ('t', 'b', 'b'))

    bins_centers_out = out.createVariable('bins', 'float32', ('b',))
    bins_edges_out   = out.createVariable('bin_edges', 'float32', ('b_edges',))

    # metadata
    pdf.setncattr('description', '2-point joint PDF of vorticity')
    bins_centers_out.setncattr('description', 'bin centers')
    bins_edges_out.setncattr('description', 'bin edges')

    # store bins once
    bins_centers_out[:] = centers
    bins_edges_out[:] = edges

    # ---- compute PDFs ----
    for i in range(nt):
        wi = w[i, :, :]
        w1 = wi
        w2 = np.roll(wi, shift=shift, axis=(0, 1))

        hist2d, _, _ = np.histogram2d(
            w1.flatten(),
            w2.flatten(),
            bins=(edges, edges),
            density=True
        )

        pdf[i, :, :] = hist2d
        t_out[i] = t[i]

        if i % 10 == 0:
            sys.stdout.write(f"\r[PDF2] Processing timestep {i}/{nt}    ")
            sys.stdout.flush()

    data.close()
    out.close()

    print(f"\n[PDF2] written to: {output_file}")
    return output_file

def pdf2_to_movie(nc_file, output_file="pdf2.mp4", fps=30, axis=0):
    """
    Make a movie of 2-point joint PDF slices along one axis.

    Parameters
    ----------
    nc_file : str
        Path to input pdf2 NetCDF file
    output_file : str
        Movie file to save (mp4 or gif)
    fps : int
        Frames per second
    axis : int
        Axis along which to slice for 2D plot (0: x-axis, 1: y-axis)
    """

    data = Dataset(nc_file, 'r')
    t = data.variables['t'][:]
    bins = data.variables['bins'][:]
    pdf = data.variables['pdf'][:]

    nframes = len(t)

    fig, ax = plt.subplots()
    line = ax.imshow(pdf[0, :, :], origin='lower',
                     extent=[bins[0], bins[-1], bins[0], bins[-1]],
                     aspect='auto', cmap='viridis')

    ax.set_xlabel("vorticity w1")
    ax.set_ylabel("vorticity w2")
    ax.set_title(f"t = {t[0]:.4f}")
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("PDF density")

    def update(i):
        line.set_data(pdf[i, :, :])
        ax.set_title(f"t = {t[i]:.4f}")
        progress = (i + 1) / nframes * 100
        sys.stdout.write(f"\r[PDF2] Rendering frame {i+1}/{nframes} ({progress:.1f}%)")
        sys.stdout.flush()
        return line,

    anim = animation.FuncAnimation(fig, update, frames=nframes, blit=False)

    # Try FFmpeg first
    try:
        from matplotlib.animation import FFMpegWriter
        mpl.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"
        writer = FFMpegWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=150)
        print(f"\n[PDF2] Movie saved: {output_file}")
    except Exception as e:
        print(f"\n[PDF2] FFmpeg failed, fallback to GIF: {e}")
        from matplotlib.animation import PillowWriter
        gif_file = output_file.replace(".mp4", ".gif")
        anim.save(gif_file, writer=PillowWriter(fps=fps), dpi=150)
        print(f"[PDF2] Saved GIF: {gif_file}")

    data.close()