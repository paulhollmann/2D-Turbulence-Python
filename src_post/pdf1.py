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



def compute_pdf1(input_file, bins=100):
    """
    Compute 1-point PDF of vorticity from a fluid.nc file.

    Stores BOTH:
    - bin centers (recommended for plotting)
    - bin edges (for reproducibility)
    """

    # ---- open input file ----
    print(f"[PDF1] opening {input_file} ...")
    data = Dataset(input_file, 'r')

    w = data.variables['w'][:]   # (t, y, x)
    t = data.variables['t'][:]

    nt = w.shape[0]

    # ---- global min/max ----
    wmin = np.min(w)
    wmax = np.max(w)

    print(f"[PDF1] global min = {wmin:.5e}, max = {wmax:.5e}")

    # ---- bin edges + centers ----
    edges = np.linspace(wmin, wmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])   

    # ---- output file ----
    dirname = os.path.dirname(input_file)
    basename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(dirname, basename + ".pdf1.nc")

    out = Dataset(output_file, 'w', 'NETCDF4')

    # dimensions
    out.createDimension('t', None)
    out.createDimension('b', bins)
    out.createDimension('b_edges', bins + 1)

    # variables
    t_out = out.createVariable('t', 'float32', ('t',))
    pdf = out.createVariable('pdf', 'float32', ('t', 'b'))

    bins_centers_out = out.createVariable('bins', 'float32', ('b',))      
    bins_edges_out   = out.createVariable('bin_edges', 'float32', ('b_edges',))  

    # metadata
    pdf.setncattr('description', '1-point PDF of vorticity')
    bins_centers_out.setncattr('description', 'bin centers')
    bins_edges_out.setncattr('description', 'bin edges')

    # store bins once
    bins_centers_out[:] = centers
    bins_edges_out[:] = edges

    # ---- compute PDFs ----
    for i in range(nt):
        wi = w[i, :, :].flatten()

        hist, _ = np.histogram(wi, bins=edges, density=True)

        pdf[i, :] = hist
        t_out[i] = t[i]

        if i % 10 == 0:
            sys.stdout.write(f"\r[PDF1] Processing timestep {i}/{nt}    ")
            sys.stdout.flush()


    # ---- close ----
    data.close()
    out.close()

    print(f"\n[PDF1] written to: {output_file}")
    return output_file



def pdf1_to_movie(nc_file, output_file="pdf1.mp4", fps=30):
    data = Dataset(nc_file, 'r')

    t = data.variables['t'][:]
    bins = data.variables['bins'][:]
    pdf = data.variables['pdf'][:]

    nframes = len(t)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    mag = np.max([bins.min(), bins.max()])
    ax.set_xlim(-mag, mag)
    ax.set_ylim(0, pdf.max() * 1.1)
    ax.set_xlabel("vorticity")
    ax.set_ylabel("PDF")

    def update(i):
        line.set_data(bins, pdf[i, :])
        ax.set_title(f"t = {t[i]:.4f}")

        # ✅ progress print
        progress = (i + 1) / nframes * 100
        sys.stdout.write(f"\r[PDF1] Rendering frame {i+1}/{nframes} ({progress:.1f}%)")
        sys.stdout.flush()

        return line,

    anim = animation.FuncAnimation(fig, update, frames=nframes, blit=True)

    mpl.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"

    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=150)
        print("\n[PDF1] Movie saved:", output_file)
    except Exception as e:
        print("\n[PDF1] FFmpeg failed, fallback to GIF:", e)
        from matplotlib.animation import PillowWriter
        gif_file = output_file.replace(".mp4", ".gif")
        anim.save(gif_file, writer=PillowWriter(fps=fps))
        print("\n[PDF1] Saved GIF:", gif_file)

    data.close()