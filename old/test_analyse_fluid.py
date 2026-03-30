from netCDF4 import Dataset
import xarray as xr
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt


path = 'P:\\git\\2D-Turbulence-Python\\data_2d_mcwilliams_Re360_384x384\\'

ds = Dataset(path + 'fluid.nc', mode='r')
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
ds = xr.open_dataset("C:\\tmp\\ph\\fluid.nc", chunks={"t": 1000})

# dask arrays for vorticity
w = ds["w"].data    # vorticity
w2 = w**2           # enstrophy = w^2
abs_w = da.abs(w)   # abs vorticity |w|
lapw = ds["lapw"].data # laplacian of vorticity



# compute global min/max
with ProgressBar():
    wmin = w.min().compute()
with ProgressBar():
    wmax = w.max().compute()

if True:
    nbins = 50
    bins = np.linspace(wmin, wmax, nbins + 1)

    # histogram of w (counts per bin)
    with ProgressBar():
        hist_w, _ = da.histogram(w, bins=bins, range=(wmin, wmax))
        hist_w = hist_w.compute()

    # histogram of w weighted by 
    with ProgressBar():
        hist_w2, _ = da.histogram(w, bins=bins, range=(wmin, wmax), weights=w2)
        hist_w2 = hist_w2.compute()

    # histogram of w weighted by 
    with ProgressBar():
        hist_abs_w, _ = da.histogram(w, bins=bins, range=(wmin, wmax), weights=abs_w)
        hist_abs_w = hist_abs_w.compute()

    # conditional expectation E[w^2 | w in bin_i]
    cond_vals_w2_w = hist_w2 / hist_w
    # conditional expectation E[sqrt(w^2) | w in bin_i]
    cond_vals_abs_w_w = hist_abs_w / hist_w


    ## PLOTTING
    # centers for plotting
    bin_centers = 0.5 * (bins[:-1] + bins[1:])


    plt.plot(bin_centers, cond_vals_w2_w, label=f"E[w^2 | w]")
    plt.plot(bin_centers, cond_vals_abs_w_w, label=f"E[|w| | w]")
    plt.plot(bin_centers, bin_centers**2, "--", label="w^2 reference")
    plt.xlabel("Vorticity w")
    plt.ylabel("Conditional quantity")
    plt.title("Conditional Turbulence Statistics")
    plt.legend()
    plt.grid()
    plt.show()

if True:
    # Bin setup
    nbins = 200
    bins = np.linspace(wmin, wmax, nbins + 1)
    
    # Choose a single time step
    time_index = 1000
    w_t0 = w[time_index, ...]  # select first time slice (all spatial points)

    # Compute histogram / PDF
    with ProgressBar():
        hist_t0, bin_edges = da.histogram(w_t0, bins=bins, range=(wmin, wmax), density=True)
        hist_t0 = hist_t0.compute()

    # Bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Plot PDF at one time step
    plt.figure()
    plt.plot(bin_centers, hist_t0, label=f"PDF of w at t_idx={time_index}")
    plt.xlabel("Vorticity w")
    plt.ylabel("PDF")
    plt.title("PDF of Vorticity at Single Time Step")
    plt.grid()
    plt.legend()
    plt.show()