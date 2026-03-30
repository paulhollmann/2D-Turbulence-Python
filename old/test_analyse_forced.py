from netCDF4 import Dataset
import xarray as xr
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter,FuncAnimation
import shutil
import os

src = 'P:\\git\\2D-Turbulence-Python\\data_2026-03-21_19-02-19\\fluid.nc'
out_dir = os.path.dirname(src)
gif_path = os.path.join(out_dir, "vorticity.gif")
dst_dir = 'C:\\tmp\\ph'
dst = os.path.join(dst_dir, 'fluid.nc')
os.makedirs(dst_dir, exist_ok=True)
shutil.copy2(src, dst)

ds = Dataset(dst, mode='r')
print(ds)
print("\nDimensions:")
for name, dim in ds.dimensions.items():
    print(f"  {name}: {len(dim)}")
# variables
print("\nVariables:")
for name, var in ds.variables.items():
    print(f"  {name}: shape={var.shape}, dtype={var.dtype}")
ds.close()

# open lazily
ds = xr.open_dataset(dst, chunks={"t": 1000})

# open your dataset (lazy loading with chunks)
ds = xr.open_dataset("C:\\tmp\\ph\\fluid.nc", chunks={"t": 100})

# vorticity
w = ds["w"]  # shape: (t, y, x)

# convert to numpy if dataset is small enough, otherwise slice
w_data = w[:].compute()  # (nt, ny, nx)
nt = w_data.shape[0]

# setup figure
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(w_data[0], cmap='RdBu_r', origin='lower')
ax.set_title("Vorticity")
plt.colorbar(im, ax=ax)

# update function for animation
def update(frame):
    im.set_data(w_data[frame])
    ax.set_xlabel(f"Time step: {frame}, t={ds['t'][frame].values:.3f}")
    return [im]

# create animation
anim = FuncAnimation(fig, update, frames=nt, blit=True)

# save as mp4
anim.save(gif_path, writer=PillowWriter(fps=30))
plt.close(fig)
"""

import xarray as xr
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

# Open dataset lazily
ds = xr.open_dataset("C:\\tmp\\ph\\fluid.nc", chunks={"t": 100})

w = ds["w"].data  # shape: (t, y, x)
nx, ny = w.shape[1], w.shape[2]

def compute_energy_spectrum(w_t):
    # Compute isotropic 2D energy spectrum for a single timestep.
    # Fourier transform
    wh = np.fft.fft2(w_t)
    wh = np.fft.fftshift(wh)  # shift zero freq to center
    
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=1./nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=1./ny))
    
    KX, KY = np.meshgrid(kx, ky)
    k_mag = np.sqrt(KX**2 + KY**2)
    
    # Isotropic binning
    k_bins = np.arange(0.5, np.max(k_mag)+1, 1.0)
    E_k = np.zeros_like(k_bins)
    
    for i, k in enumerate(k_bins):
        mask = (k_mag >= k-0.5) & (k_mag < k+0.5)
        E_k[i] = 0.5 * np.sum(np.abs(wh[mask])**2) / (nx*ny)**2  # normalize

    return E_k, k_bins

# Select a subset of timesteps to save computation
timesteps = range(0, w.shape[0], 50)

spectra = []
with ProgressBar():
    for t_idx in timesteps:
        w_t = w[t_idx, :, :].compute()
        E_k, k_bins = compute_energy_spectrum(w_t)
        spectra.append(E_k)

spectra = np.array(spectra)

# Plot as a waterfall / cascade plot
plt.figure(figsize=(8,6))
for i, E in enumerate(spectra):
    plt.loglog(k_bins, E, label=f"t={timesteps[i]}")

plt.xlabel("Wavenumber k")
plt.ylabel("Energy E(k)")
plt.title("Energy Spectrum Evolution (Inverse Cascade)")
plt.grid(True, which="both", ls="--")
plt.legend(ncol=2, fontsize=8)
plt.show()


plt.figure(figsize=(8,6))
plt.imshow(np.log10(spectra.T), origin='lower', 
           aspect='auto', extent=[0, w.shape[0], k_bins[0], k_bins[-1]],
           cmap='plasma')
plt.colorbar(label='log10(E(k))')
plt.xlabel('Time index')
plt.ylabel('Wavenumber k')
plt.title('Energy Spectrum Evolution (Inverse Cascade)')
plt.show()
"""
