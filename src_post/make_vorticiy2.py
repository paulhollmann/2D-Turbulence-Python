import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
import vorticity

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,          # keeps it fast but LaTeX-looking
    "mathtext.fontset": "cm",      # Computer Modern (MATLAB-like LaTeX feel)
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 11
})


times = [1.0, 5.0, 10.0, 15.0]

files = [
    "data_2d_turbulence_long/fluid.nc",
    "data_2026-04-03_11-01-28/fluid.nc",
    "data_2026-04-03_10-52-51/fluid.nc",
    "data_2026-04-03_11-00-34/fluid.nc",
]

names = ["$\mathrm{Re}=200$", "$\mathrm{Re}=360$", "$\mathrm{Re}=200$ (forced)", "$\mathrm{Re}=360$ (forced)"]

# ------------------------------------------------------------
# ✅ GRID WITH EXTRA COLUMN FOR COLORBARS
# ------------------------------------------------------------
fig = plt.figure(figsize=(11, 10))
gs = fig.add_gridspec(4, 5, width_ratios=[1, 1, 1, 1, 0.05])

axes = np.empty((4, 4), dtype=object)
caxes = np.empty(4, dtype=object)

for i in range(4):
    for j in range(4):
        axes[i, j] = fig.add_subplot(gs[i, j])
    caxes[i] = fig.add_subplot(gs[i, 4])  # colorbar axis per row

# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------
for i, (file, name) in enumerate(zip(files, names)):

    vmin_row = +np.inf
    vmax_row = -np.inf

    for t in times:
        _, vmin, vmax = vorticity.get_vorticity_minmax(file, t)
        vmin_row = min(vmin_row, vmin)
        vmax_row = max(vmax_row, vmax)

    vabs = max(abs(vmin_row), abs(vmax_row))

    if "f" in name or "F" in name:
        vabs = 0.3 * vabs
    else:
        vabs = 0.9 * vabs

    vmin_row, vmax_row = -vabs, vabs

    data = Dataset(file, 'r')
    w = data.variables['w']
    t_arr = data.variables['t'][:]

    for j, target_time in enumerate(times):

        ax = axes[i, j]

        idx = np.argmin(np.abs(t_arr - target_time))
        w_slice = w[idx, :, :]

        img = ax.imshow(
            w_slice,
            vmin=vmin_row,
            vmax=vmax_row,
            cmap="RdBu_r",
            origin='lower',
            aspect='auto',
            interpolation='none'
        )

        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.set_title(rf"$t = {t_arr[idx]:.1f}$")

    data.close()

    # ------------------------------------------------------------
    # ✅ ONE PERFECT COLORBAR PER ROW (NO DISTORTION)
    # ------------------------------------------------------------
    cbar = fig.colorbar(img, cax=caxes[i])
    cbar.set_label(rf"$\omega$ | {name}")

# ------------------------------------------------------------
# layout (IMPORTANT)
# ------------------------------------------------------------
fig.subplots_adjust(
    left=0.03,
    right=0.92,
    bottom=0.03,
    top=0.97,
    wspace=0.02,   # columns touch
    hspace=0.05    # rows separated
)

plt.savefig("vorticity_rows.eps", format="eps", dpi=600)
plt.savefig("vorticity_rows.pdf")

plt.close(fig)