import os
import sys
import time
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
from netCDF4 import Dataset
from numba import njit, prange, set_num_threads, get_num_threads


# ============================================================
# Logging helper: print to terminal and to file
# ============================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ============================================================
# Spectral operators from vorticity, periodic domain
# ============================================================

def make_spectral_operators(Ny, Nx, Lx=2*np.pi, Ly=2*np.pi):
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)

    KX, KY = np.meshgrid(kx, ky)
    k2 = KX**2 + KY**2

    op_u = np.zeros((Ny, Nx), dtype=np.complex128)
    op_v = np.zeros((Ny, Nx), dtype=np.complex128)
    op_lap = -k2.astype(np.complex128)

    mask = k2 > 0.0

    # For incompressible 2D periodic flow with w = dv/dx - du/dy:
    # psi_hat = -w_hat / k^2, u = d psi / dy, v = -d psi / dx.
    # This is equivalent to the operators below.
    op_u[mask] = 1j * KY[mask] / k2[mask]
    op_v[mask] = -1j * KX[mask] / k2[mask]

    op_u[0, 0] = 0.0
    op_v[0, 0] = 0.0
    op_lap[0, 0] = 0.0

    return op_u, op_v, op_lap


def make_velocity_operators(Ny, Nx, Lx=2*np.pi, Ly=2*np.pi):
    op_u, op_v, _ = make_spectral_operators(Ny, Nx, Lx=Lx, Ly=Ly)
    return op_u, op_v


def velocity_from_vorticity_periodic(w, op_u, op_v):
    w_hat = np.fft.fft2(w)

    u = np.fft.ifft2(op_u * w_hat).real
    v = np.fft.ifft2(op_v * w_hat).real

    return u, v


def laplacian_from_vorticity_periodic(w, op_lap):
    w_hat = np.fft.fft2(w)
    lapw = np.fft.ifft2(op_lap * w_hat).real
    return lapw


# ============================================================
# Dataset variable helpers
# ============================================================

def _find_variable_pair(data, candidate_pairs):
    for name_u, name_v in candidate_pairs:
        if name_u in data.variables and name_v in data.variables:
            return name_u, name_v
    return None, None


def _find_variable(data, candidate_names):
    for name in candidate_names:
        if name in data.variables:
            return name
    return None


# ============================================================
# Bin helpers
# ============================================================

def compute_w_bins(w, w_abs_max, n_bins):
    scale = n_bins / (2.0 * w_abs_max)

    wb = ((w + w_abs_max) * scale).astype(np.int64)
    wb = np.clip(wb, 0, n_bins - 1)

    return wb


def precompute_shifts_sorted_by_rbin(Ny, Nx, n_bins, L=2*np.pi):
    dx = L / Nx
    dy = L / Ny

    r_max = np.sqrt((L / 2.0)**2 + (L / 2.0)**2)

    r_edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    shifts = []

    for sy in range(Ny):
        ry = sy * dy
        if ry > L / 2.0:
            ry -= L

        for sx in range(Nx):
            rx = sx * dx
            if rx > L / 2.0:
                rx -= L

            r = np.sqrt(rx**2 + ry**2)

            if r == 0.0:
                continue

            r_bin = np.searchsorted(r_edges, r, side="right") - 1

            if r_bin < 0 or r_bin >= n_bins:
                continue

            shifts.append(
                (
                    r_bin,
                    sy,
                    sx,
                    rx / r,
                    ry / r,
                )
            )

    shifts.sort(key=lambda x: x[0])

    n_shifts = len(shifts)

    rbin_arr = np.empty(n_shifts, dtype=np.int64)
    sy_arr = np.empty(n_shifts, dtype=np.int64)
    sx_arr = np.empty(n_shifts, dtype=np.int64)
    rhx_arr = np.empty(n_shifts, dtype=np.float64)
    rhy_arr = np.empty(n_shifts, dtype=np.float64)

    for q, item in enumerate(shifts):
        rb, sy, sx, rhx, rhy = item

        rbin_arr[q] = rb
        sy_arr[q] = sy
        sx_arr[q] = sx
        rhx_arr[q] = rhx
        rhy_arr[q] = rhy

    rbin_start = np.zeros(n_bins, dtype=np.int64)
    rbin_end = np.zeros(n_bins, dtype=np.int64)

    q = 0
    for rb in range(n_bins):
        rbin_start[rb] = q

        while q < n_shifts and rbin_arr[q] == rb:
            q += 1

        rbin_end[rb] = q

    return (
        sy_arr,
        sx_arr,
        rbin_start,
        rbin_end,
        rhx_arr,
        rhy_arr,
        r_edges,
        r_centers,
    )


# ============================================================
# Parallel numba accumulator
# ============================================================

@njit(parallel=True)
def accumulate_conditional_parallel(
    u,
    v,
    w_bin,
    lapw,
    Re,
    sy_arr,
    sx_arr,
    rbin_start,
    rbin_end,
    rhx_arr,
    rhy_arr,
    sum_S,
    sum_A1,
    sum_A2,
    count,
):
    Ny, Nx = u.shape
    n_r_bins = rbin_start.shape[0]

    # Parallel over scalar r bins.
    # Safe because each rb writes to its own [rb,:,:] slice.
    for rb in prange(n_r_bins):
        q0 = rbin_start[rb]
        q1 = rbin_end[rb]

        for q in range(q0, q1):
            sy = sy_arr[q]
            sx = sx_arr[q]
            rhx = rhx_arr[q]
            rhy = rhy_arr[q]

            for j in range(Ny):
                j2 = j + sy
                if j2 >= Ny:
                    j2 -= Ny

                for i in range(Nx):
                    i2 = i + sx
                    if i2 >= Nx:
                        i2 -= Nx

                    b1 = w_bin[j, i]
                    b2 = w_bin[j2, i2]

                    du_long = (
                        (u[j2, i2] - u[j, i]) * rhx
                        + (v[j2, i2] - v[j, i]) * rhy
                    )

                    # A1 uses the Laplacian at point 1.
                    # A2 uses the Laplacian at point 2.
                    a1_val = lapw[j, i] / Re
                    a2_val = lapw[j2, i2] / Re

                    sum_S[rb, b1, b2] += du_long
                    sum_A1[rb, b1, b2] += a1_val
                    sum_A2[rb, b1, b2] += a2_val
                    count[rb, b1, b2] += 1


# ============================================================
# Main implementation
# ============================================================

def _compute_conditional_pdf_selected_timesteps_parallel_impl(
    input_file,
    output_dir,
    Re,
    timesteps=None,
    n_bins=100,
    L=2.0 * np.pi,
    n_threads=96,
    dtype_out=np.float32,
    compression_level=1,
    prefer_input_velocity=True,
    prefer_input_lapw=True,
    velocity_var_pairs=(
        ("u", "v"),
        ("ux", "uy"),
        ("vx", "vy"),
        ("velocity_x", "velocity_y"),
    ),
    lapw_var_names=("lapw", "lap_w", "laplacian_w", "omega_lap", "lap_omega"),
    log_file=None,
):
    data = Dataset(input_file, "r")

    try:
        if "w" not in data.variables:
            raise KeyError("Input file must contain vorticity variable 'w'.")

        w_all = data.variables["w"]
        t_all = data.variables["t"][:] if "t" in data.variables else None

        Nt, Ny, Nx = w_all.shape

        lapw_name = _find_variable(data, lapw_var_names)
        u_name, v_name = _find_variable_pair(data, velocity_var_pairs)

        use_input_lapw = bool(prefer_input_lapw and lapw_name is not None)
        use_input_velocity = bool(prefer_input_velocity and u_name is not None and v_name is not None)

        velocity_source = "input" if use_input_velocity else "constructed_from_w"
        lapw_source = "input" if use_input_lapw else "constructed_from_w"

        if use_input_velocity:
            velocity_construction = f"read directly from input variables '{u_name}' and '{v_name}'"
        else:
            velocity_construction = (
                "constructed spectrally from vorticity on a periodic domain: "
                "w_hat=fft2(w), u=ifft2((i*ky/k^2)*w_hat), "
                "v=ifft2((-i*kx/k^2)*w_hat), zero mode set to 0"
            )

        if use_input_lapw:
            lapw_construction = f"read directly from input variable '{lapw_name}'"
        else:
            lapw_construction = (
                "constructed spectrally from vorticity on a periodic domain: "
                "lapw=ifft2(-(kx^2+ky^2)*fft2(w)), zero mode set to 0"
            )

        print(f"Data shape: Nt={Nt}, Ny={Ny}, Nx={Nx}")
        print(f"Selected timesteps: {timesteps if timesteps is not None else 'all'}")
        print(f"Using n_bins = {n_bins}")
        print(f"Using Re = {Re}")
        print(f"Velocity source: {velocity_source}")
        print(f"Velocity construction: {velocity_construction}")
        print(f"Laplacian source: {lapw_source}")
        print(f"Laplacian construction: {lapw_construction}")
        if log_file is not None:
            print(f"Log file: {log_file}")

        if timesteps is None:
            timesteps = np.arange(Nt, dtype=np.int64)
        else:
            timesteps = np.asarray(timesteps, dtype=np.int64)

        for n in timesteps:
            if n < 0 or n >= Nt:
                raise ValueError(f"Timestep {n} is outside valid range [0, {Nt-1}]")

        # ------------------------------------------------------------
        # Global vorticity bounds over all time
        # ------------------------------------------------------------
        print("Scanning global vorticity bounds over all timesteps...")

        w_abs_max = 0.0

        for n in range(Nt):
            w_n = np.asarray(w_all[n, :, :])
            w_abs_max = max(w_abs_max, float(np.nanmax(np.abs(w_n))))

        if w_abs_max == 0.0:
            raise ValueError("Global vorticity maximum is zero.")

        w_edges = np.linspace(-w_abs_max, w_abs_max, n_bins + 1)
        w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])

        print(f"Using omega range: [{-w_abs_max}, {w_abs_max}]")

        # ------------------------------------------------------------
        # Precompute shifts and FFT operators
        # ------------------------------------------------------------
        print("Precomputing scalar-r displacement bins...")

        (
            sy_arr,
            sx_arr,
            rbin_start,
            rbin_end,
            rhx_arr,
            rhy_arr,
            r_edges,
            r_centers,
        ) = precompute_shifts_sorted_by_rbin(Ny, Nx, n_bins, L=L)

        print(f"Number of displacement vectors: {len(sy_arr)}")

        print("Precomputing spectral operators...")
        op_u, op_v, op_lap = make_spectral_operators(Ny, Nx, Lx=L, Ly=L)

        # ------------------------------------------------------------
        # Set numba threads
        # ------------------------------------------------------------
        set_num_threads(n_threads)
        print("Numba threads:", get_num_threads())

        # ------------------------------------------------------------
        # Compile numba kernel with a tiny dummy call
        # ------------------------------------------------------------
        print("Compiling numba kernel...")

        dummy_u = np.zeros((Ny, Nx), dtype=np.float64)
        dummy_v = np.zeros((Ny, Nx), dtype=np.float64)
        dummy_wb = np.zeros((Ny, Nx), dtype=np.int64)
        dummy_lapw = np.zeros((Ny, Nx), dtype=np.float64)

        dummy_sum = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
        dummy_sum_A1 = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
        dummy_sum_A2 = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
        dummy_count = np.zeros((n_bins, n_bins, n_bins), dtype=np.int64)

        accumulate_conditional_parallel(
            dummy_u,
            dummy_v,
            dummy_wb,
            dummy_lapw,
            float(Re),
            sy_arr,
            sx_arr,
            rbin_start,
            rbin_end,
            rhx_arr,
            rhy_arr,
            dummy_sum,
            dummy_sum_A1,
            dummy_sum_A2,
            dummy_count,
        )

        print("Compilation done.")

        # ------------------------------------------------------------
        # Process only selected timesteps
        # ------------------------------------------------------------
        total_start = time.perf_counter()

        for n in timesteps:
            step_start = time.perf_counter()

            print(f"\nProcessing timestep {n} / {Nt-1}")

            # -----------------------------
            # Read vorticity and possibly direct fields
            # -----------------------------
            t0 = time.perf_counter()
            w = np.asarray(w_all[n, :, :], dtype=np.float64)
            if use_input_lapw:
                lapw = np.asarray(data.variables[lapw_name][n, :, :], dtype=np.float64)
            else:
                lapw = None
            if use_input_velocity:
                u = np.asarray(data.variables[u_name][n, :, :], dtype=np.float64)
                v = np.asarray(data.variables[v_name][n, :, :], dtype=np.float64)
            else:
                u = None
                v = None
            t_read = time.perf_counter() - t0

            # -----------------------------
            # Recover missing velocity / Laplacian
            # -----------------------------
            t0 = time.perf_counter()
            if not use_input_velocity:
                u, v = velocity_from_vorticity_periodic(w, op_u, op_v)
            t_velocity = time.perf_counter() - t0

            t0 = time.perf_counter()
            if not use_input_lapw:
                lapw = laplacian_from_vorticity_periodic(w, op_lap)
            t_lapw = time.perf_counter() - t0

            # -----------------------------
            # Bin vorticity
            # -----------------------------
            t0 = time.perf_counter()
            w_bin = compute_w_bins(w, w_abs_max, n_bins)
            t_wbin = time.perf_counter() - t0

            # -----------------------------
            # Allocate accumulators
            # -----------------------------
            t0 = time.perf_counter()
            sum_S = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
            sum_A1 = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
            sum_A2 = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
            count = np.zeros((n_bins, n_bins, n_bins), dtype=np.int64)
            t_alloc = time.perf_counter() - t0

            # -----------------------------
            # Main conditional accumulation
            # -----------------------------
            t0 = time.perf_counter()

            accumulate_conditional_parallel(
                u,
                v,
                w_bin,
                lapw,
                float(Re),
                sy_arr,
                sx_arr,
                rbin_start,
                rbin_end,
                rhx_arr,
                rhy_arr,
                sum_S,
                sum_A1,
                sum_A2,
                count,
            )

            t_accum = time.perf_counter() - t0

            # -----------------------------
            # Normalize conditional average
            # -----------------------------
            t0 = time.perf_counter()

            S = np.full((n_bins, n_bins, n_bins), np.nan, dtype=dtype_out)
            A1 = np.full((n_bins, n_bins, n_bins), np.nan, dtype=dtype_out)
            A2 = np.full((n_bins, n_bins, n_bins), np.nan, dtype=dtype_out)

            mask = count > 0
            S[mask] = (sum_S[mask] / count[mask]).astype(dtype_out)
            A1[mask] = (sum_A1[mask] / count[mask]).astype(dtype_out)
            A2[mask] = (sum_A2[mask] / count[mask]).astype(dtype_out)

            t_normalize = time.perf_counter() - t0

            # -----------------------------
            # Write output
            # -----------------------------
            output_file = os.path.join(output_dir, f"conditional_t{int(n):06d}.nc")

            print(f"Writing {output_file}")

            t0 = time.perf_counter()

            with Dataset(output_file, "w") as out:
                out.createDimension("r_bin", n_bins)
                out.createDimension("w1_bin", n_bins)
                out.createDimension("w2_bin", n_bins)

                out.createDimension("r_edge", n_bins + 1)
                out.createDimension("w_edge", n_bins + 1)

                vr_edges = out.createVariable("r_edges", "f8", ("r_edge",))
                vw_edges = out.createVariable("w_edges", "f8", ("w_edge",))

                vr_centers = out.createVariable("r_centers", "f8", ("r_bin",))
                vw_centers = out.createVariable("w_centers", "f8", ("w1_bin",))

                vS = out.createVariable(
                    "S",
                    "f4",
                    ("r_bin", "w1_bin", "w2_bin"),
                    zlib=True,
                    complevel=compression_level,
                    fill_value=np.nan,
                )

                vA1 = out.createVariable(
                    "A1",
                    "f4",
                    ("r_bin", "w1_bin", "w2_bin"),
                    zlib=True,
                    complevel=compression_level,
                    fill_value=np.nan,
                )

                vA2 = out.createVariable(
                    "A2",
                    "f4",
                    ("r_bin", "w1_bin", "w2_bin"),
                    zlib=True,
                    complevel=compression_level,
                    fill_value=np.nan,
                )

                vcount = out.createVariable(
                    "count",
                    "i8",
                    ("r_bin", "w1_bin", "w2_bin"),
                    zlib=True,
                    complevel=compression_level,
                )

                vr_edges[:] = r_edges
                vw_edges[:] = w_edges
                vr_centers[:] = r_centers
                vw_centers[:] = w_centers

                vS[:, :, :] = S
                vA1[:, :, :] = A1
                vA2[:, :, :] = A2
                vcount[:, :, :] = count

                out.time_index = int(n)
                out.time_value = float(t_all[n]) if t_all is not None else float(n)
                out.box_length = float(L)
                out.w_abs_max = float(w_abs_max)
                out.n_bins = int(n_bins)
                out.Re = float(Re)
                out.numba_threads = int(get_num_threads())
                out.input_file = os.path.abspath(input_file)
                if log_file is not None:
                    out.log_file = os.path.abspath(log_file)

                out.velocity_source = velocity_source
                out.velocity_variable_u = u_name if use_input_velocity else ""
                out.velocity_variable_v = v_name if use_input_velocity else ""
                out.velocity_construction = velocity_construction

                out.lapw_source = lapw_source
                out.lapw_variable = lapw_name if use_input_lapw else ""
                out.lapw_construction = lapw_construction

                out.description = (
                    "Conditional averages: "
                    "S(t,r,w1,w2)=<(u2-u1) dot r_hat | scalar r,w1,w2>; "
                    "A1(t,r,w1,w2)=<(1/Re) lapw1 | scalar r,w1,w2>; "
                    "A2(t,r,w1,w2)=<(1/Re) lapw2 | scalar r,w1,w2>"
                )

            t_write = time.perf_counter() - t0

            # -----------------------------
            # File size
            # -----------------------------
            file_size_bytes = os.path.getsize(output_file)
            file_size_mb = file_size_bytes / 1024**2
            file_size_gb = file_size_bytes / 1024**3

            step_total = time.perf_counter() - step_start

            print("Timing summary:")
            print(f"  read input:    {t_read:10.3f} s")
            print(f"  velocity:      {t_velocity:10.3f} s  ({velocity_source})")
            print(f"  lapw:          {t_lapw:10.3f} s  ({lapw_source})")
            print(f"  bin omega:     {t_wbin:10.3f} s")
            print(f"  allocate:      {t_alloc:10.3f} s")
            print(f"  accumulate:    {t_accum:10.3f} s")
            print(f"  normalize:     {t_normalize:10.3f} s")
            print(f"  write file:    {t_write:10.3f} s")
            print(f"  total step:    {step_total:10.3f} s")
            print(f"  output size:   {file_size_mb:10.2f} MB  ({file_size_gb:.3f} GB)")

            print(f"Finished timestep {n}")

        total_time = time.perf_counter() - total_start
        print(f"\nTotal selected-timestep runtime: {total_time:.3f} s")

    finally:
        data.close()

    print("Done.")


# ============================================================
# Public function with file logging
# ============================================================

def compute_conditional_pdf_selected_timesteps_parallel(
    input_file,
    output_dir,
    Re,
    timesteps=None,
    n_bins=100,
    L=2.0 * np.pi,
    n_threads=96,
    dtype_out=np.float32,
    compression_level=1,
    prefer_input_velocity=True,
    prefer_input_lapw=True,
    velocity_var_pairs=(
        ("u", "v"),
        ("ux", "uy"),
        ("vx", "vy"),
        ("velocity_x", "velocity_y"),
    ),
    lapw_var_names=("lapw", "lap_w", "laplacian_w", "omega_lap", "lap_omega"),
    log_file=None,
):
    """
    Compute conditional averages for selected timesteps.

    Robust field handling:
      - If velocity variables are available and prefer_input_velocity=True,
        they are read from the input file.
      - Otherwise u,v are constructed spectrally from vorticity w.
      - If lapw is available and prefer_input_lapw=True, it is read from input.
      - Otherwise lapw is constructed spectrally as Delta w.

    All print output is written both to the terminal and to log_file.
    If log_file is None, output_dir/conditional_run.log is used.
    """
    os.makedirs(output_dir, exist_ok=True)

    if log_file is None:
        log_file = os.path.join(output_dir, "conditional_run.log")

    with open(log_file, "w", buffering=1) as log:
        tee_out = Tee(sys.__stdout__, log)
        tee_err = Tee(sys.__stderr__, log)

        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"Started conditional averaging at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            _compute_conditional_pdf_selected_timesteps_parallel_impl(
                input_file=input_file,
                output_dir=output_dir,
                Re=Re,
                timesteps=timesteps,
                n_bins=n_bins,
                L=L,
                n_threads=n_threads,
                dtype_out=dtype_out,
                compression_level=compression_level,
                prefer_input_velocity=prefer_input_velocity,
                prefer_input_lapw=prefer_input_lapw,
                velocity_var_pairs=velocity_var_pairs,
                lapw_var_names=lapw_var_names,
                log_file=log_file,
            )
            print(f"Finished conditional averaging at {time.strftime('%Y-%m-%d %H:%M:%S')}")
