import os
import sys
import time
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
from netCDF4 import Dataset
from numba import njit, prange, set_num_threads, get_num_threads


# ============================================================
# Plotting helper for saved conditional PDF moment files
# ============================================================

def plot_conditional_pdf_moments(output_dir):
    """
    Read all conditional_t*.nc files in output_dir and create plots for
    <|w1-w2|^k>(t,r) versus r for all existing timesteps.

    This function intentionally takes only one required argument, the path
    to the directory containing files such as conditional_t000000.nc.

    Output files are written to:
        output_dir/plots_moments/

    Created plots/files:
      - moments_vs_r_tXXXXXX.png for each timestep
      - moment_k<K>_all_timesteps.png for each moment order K
      - mean_moments_vs_r.png averaged over all available timesteps
      - scaling_exponents_vs_k.png from a simple log-log fit
      - scaling_exponents.csv with the fitted slopes

    Notes:
      - The scaling fit is only a quick diagnostic. It uses the middle 60%
        of finite positive r-values and should be adjusted manually for a
        physically meaningful inertial/scaling range.
      - Requires matplotlib and netCDF4.
    """
    import glob
    import csv

    # Use a non-interactive backend, which is safer on clusters/servers.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots_moments")
    os.makedirs(plot_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(output_dir, "conditional_t*.nc")))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No files matching conditional_t*.nc were found in {output_dir!r}."
        )

    all_times = []
    all_time_indices = []
    all_moments = []
    r_centers_ref = None
    moment_orders_ref = None

    print(f"Found {len(files)} conditional files in {output_dir}")
    print(f"Writing plots to {plot_dir}")

    for file_path in files:
        with Dataset(file_path, "r") as ds:
            if "abs_dw_moments" not in ds.variables:
                raise KeyError(
                    f"{file_path} does not contain 'abs_dw_moments'. "
                    "Re-run the conditional calculation with the moments-enabled file."
                )
            if "moment_orders" not in ds.variables:
                raise KeyError(f"{file_path} does not contain 'moment_orders'.")
            if "r_centers" not in ds.variables:
                raise KeyError(f"{file_path} does not contain 'r_centers'.")

            r_centers = np.asarray(ds.variables["r_centers"][:], dtype=np.float64)
            moment_orders = np.asarray(ds.variables["moment_orders"][:], dtype=np.int64)
            moments = np.asarray(ds.variables["abs_dw_moments"][:], dtype=np.float64)

            time_index = int(getattr(ds, "time_index", len(all_times)))
            time_value = float(getattr(ds, "time_value", time_index))

        if r_centers_ref is None:
            r_centers_ref = r_centers
            moment_orders_ref = moment_orders
        else:
            if not np.allclose(r_centers_ref, r_centers):
                raise ValueError(f"r_centers in {file_path} differ from previous files.")
            if not np.array_equal(moment_orders_ref, moment_orders):
                raise ValueError(f"moment_orders in {file_path} differ from previous files.")

        all_times.append(time_value)
        all_time_indices.append(time_index)
        all_moments.append(moments)

        # --------------------------------------------------------
        # Per-timestep plot: all k versus r
        # --------------------------------------------------------
        plt.figure(figsize=(7.0, 5.0))
        for m, k in enumerate(moment_orders):
            y = moments[m, :]
            mask = np.isfinite(y) & (y > 0.0) & np.isfinite(r_centers) & (r_centers > 0.0)
            if np.any(mask):
                plt.loglog(r_centers[mask], y[mask], marker="o", linewidth=1.5, markersize=3, label=f"k={int(k)}")

        plt.xlabel("r")
        plt.ylabel(r"$\langle |\omega_1-\omega_2|^k \rangle(t,r)$")
        plt.title(f"Two-point vorticity-difference moments, t={time_value:g}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        per_time_png = os.path.join(plot_dir, f"moments_vs_r_t{time_index:06d}.png")
        plt.savefig(per_time_png, dpi=200)
        plt.close()

    all_times = np.asarray(all_times, dtype=np.float64)
    all_time_indices = np.asarray(all_time_indices, dtype=np.int64)
    all_moments = np.asarray(all_moments, dtype=np.float64)  # shape: Nt_selected, Nk, Nr

    # ------------------------------------------------------------
    # For each k: overlay all timesteps versus r
    # ------------------------------------------------------------
    for m, k in enumerate(moment_orders_ref):
        plt.figure(figsize=(7.0, 5.0))
        for a, time_value in enumerate(all_times):
            y = all_moments[a, m, :]
            mask = np.isfinite(y) & (y > 0.0) & np.isfinite(r_centers_ref) & (r_centers_ref > 0.0)
            if np.any(mask):
                plt.loglog(r_centers_ref[mask], y[mask], linewidth=1.2, alpha=0.75, label=f"t={time_value:g}")

        plt.xlabel("r")
        plt.ylabel(rf"$\langle |\omega_1-\omega_2|^{int(k)} \rangle(t,r)$")
        plt.title(f"Moment order k={int(k)} over all timesteps")
        plt.grid(True, which="both", alpha=0.3)
        if len(all_times) <= 12:
            plt.legend(fontsize=8)
        plt.tight_layout()
        k_png = os.path.join(plot_dir, f"moment_k{int(k)}_all_timesteps.png")
        plt.savefig(k_png, dpi=200)
        plt.close()

    # ------------------------------------------------------------
    # Mean over selected timesteps
    # ------------------------------------------------------------
    mean_moments = np.nanmean(all_moments, axis=0)

    plt.figure(figsize=(7.0, 5.0))
    for m, k in enumerate(moment_orders_ref):
        y = mean_moments[m, :]
        mask = np.isfinite(y) & (y > 0.0) & np.isfinite(r_centers_ref) & (r_centers_ref > 0.0)
        if np.any(mask):
            plt.loglog(r_centers_ref[mask], y[mask], marker="o", linewidth=1.5, markersize=3, label=f"k={int(k)}")

    plt.xlabel("r")
    plt.ylabel(r"time mean of $\langle |\omega_1-\omega_2|^k \rangle(t,r)$")
    plt.title("Time-averaged two-point vorticity-difference moments")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    mean_png = os.path.join(plot_dir, "mean_moments_vs_r.png")
    plt.savefig(mean_png, dpi=200)
    plt.close()

# ------------------------------------------------------------
    # Quick scaling-exponent diagnostic: mean moment ~ r^zeta_k
    # Fixed fit range: 0.1 <= r <= 0.2
    # ------------------------------------------------------------
    zeta = np.full(moment_orders_ref.shape, np.nan, dtype=np.float64)
    fit_r_min = np.full(moment_orders_ref.shape, np.nan, dtype=np.float64)
    fit_r_max = np.full(moment_orders_ref.shape, np.nan, dtype=np.float64)

    fixed_fit_r_min = 0.1
    fixed_fit_r_max = 0.2

    for m, k in enumerate(moment_orders_ref):
        y = mean_moments[m, :]

        mask = (
            np.isfinite(y)
            & (y > 0.0)
            & np.isfinite(r_centers_ref)
            & (r_centers_ref >= fixed_fit_r_min)
            & (r_centers_ref <= fixed_fit_r_max)
        )

        fit_idx = np.where(mask)[0]

        if fit_idx.size >= 3:
            coeff = np.polyfit(
                np.log(r_centers_ref[fit_idx]),
                np.log(y[fit_idx]),
                deg=1,
            )

            zeta[m] = coeff[0]
            fit_r_min[m] = r_centers_ref[fit_idx[0]]
            fit_r_max[m] = r_centers_ref[fit_idx[-1]]

    csv_path = os.path.join(plot_dir, "scaling_exponents.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "zeta_k", "fit_r_min", "fit_r_max"])
        for k, zk, r0, r1 in zip(moment_orders_ref, zeta, fit_r_min, fit_r_max):
            writer.writerow([int(k), zk, r0, r1])

    plt.figure(figsize=(6.5, 4.5))
    mask = np.isfinite(zeta)
    if np.any(mask):
        plt.plot(moment_orders_ref[mask], zeta[mask], marker="o", linewidth=1.5)
    plt.xlabel("moment order k")
    plt.ylabel(r"estimated scaling exponent $\zeta_k$")
    plt.title(r"Quick fit: $\langle |\omega_1-\omega_2|^k \rangle \sim r^{\zeta_k}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    zeta_png = os.path.join(plot_dir, "scaling_exponents_vs_k.png")
    plt.savefig(zeta_png, dpi=200)
    plt.close()

    print("Created plots:")
    print(f"  {plot_dir}")
    print(f"  {csv_path}")

    return plot_dir
