# Review of the modified post-processing code

## Scope and review method

This review covers the modified files and newly added modules in `src_post`, plus the new driver in `test_post_fluid.py`. It is a static code review supported by syntax compilation and inspection against the solver's NetCDF writer.

A full runtime test was not possible in the current workspace:

- the system Python has neither `numpy` nor `netCDF4`;
- `.venv\\Scripts\\python.exe` cannot start because it points to a missing `C:\\Python314\\python.exe`.

The virtual environment must be recreated before running the new workflow.

## Priority findings

| Priority | Finding | Evidence / impact | Recommended action |
| --- | --- | --- | --- |
| P1 | A pytest-collected file starts production work during import. | `test_post_fluid.py:36` directly calls the moments calculation and line 48 directly plots results. Pytest imports files named `test_*.py`, so test collection can trigger an enormous calculation and overwrite outputs. | Move runnable code under `if __name__ == "__main__":`; rename the file to a driver name such as `run_post_fluid.py`; add small, real unit tests under `tests/`. |
| P1 | The pairwise estimator scales as O(Nx^2 Ny^2) per time. | The accumulator loops over almost every periodic displacement and then every grid point. A 384x384 snapshot implies approximately `(384^2-1)*384^2 = 21.7e9` innermost visits per time; the driver selects ten times. | Establish a computational budget before production use. Sample displacements/directions or points, reduce radial resolution, exploit symmetries, or redesign the estimator for the target statistic. |
| P1 | High-order vorticity-difference moments are biased by bin-centre discretisation. | `compute_two_point_pdf_and_moments` at line 135 calculates `abs(w_centers[i]-w_centers[j])**k`, not the actual pair difference. The error increases with moment order and bin width. | If moments are primary results, accumulate `abs(w2-w1)**k` in the Numba loop. Keep the binned-PDF moment only as a documented histogram estimate. |
| P2 | Snapshot export always writes EPS, irrespective of `output_path`. | `save_vorticity_snapshot_pub` calls `plt.savefig(..., format="eps")`. Passing a path ending in `.pdf` or `.png` creates EPS content with a misleading extension. | Remove the fixed `format` argument or expose it as a validated parameter. Note that `dpi` does not materially affect vector EPS output. |
| P2 | The stated scaling-fit rule does not match the code. | The plotter docstring says “middle 60%”; lines 172--173 use the fixed range `0.1 <= r <= 0.2`. | Make the documentation truthful or parameterise a physically justified fit interval and store it in the output. |
| P2 | “Time-averaged” moments are actually an equal-weight average of selected snapshots. | The driver uses strongly uneven time indices (1, 10, 100, ..., 30000), while `np.nanmean(all_moments, axis=0)` weights all snapshots identically. | Rename the quantity “mean over selected snapshots”, or use a justified time weighting / uniformly spaced statistically independent samples. |
| P2 | The maximum diagonal displacement is silently omitted for even-sized grids. | In `precompute_shifts_sorted_by_rbin`, the corner displacement has `r == r_edges[-1]`; `searchsorted(..., side="right") - 1` returns `n_bins`, then the code excludes it. | Include the rightmost edge in the final radial bin, or document the deliberate exclusion. It is only one shift but indicates an edge-bin definition issue. |
| P2 | Non-finite vorticity values are not safely handled. | The global scan uses `nanmax`, but `compute_w_bins` casts every value to `int64`. A NaN converts unpredictably and is then clipped into a valid bin, contaminating statistics. | Reject non-finite data with a clear error, or mask invalid values before binning and count only valid pairs. |
| P2 | The fixed request for 96 Numba threads is non-portable. | `set_num_threads(n_threads)` can fail if the requested number exceeds Numba's configured maximum; it can also oversubscribe a shared machine. | Default to the available thread count or cap/validate the requested value, and record the value actually used. |
| P3 | Nearly identical implementations are maintained in two files. | `cond_avg.py` and `cond_avg_with_pdf_moments.py` duplicate spectral operations, binning, accumulation, I/O, and logging. | Make the moments version an optional branch/flag in one implementation, or factor common code into a shared module. |
| P3 | Figure scripts are hard-coded and execution-location dependent. | `make_vorticiy.py`, `make_vorticiy2.py`, and `info.py` contain fixed input paths; the first two use `import vorticity`, which works from `src_post` but not reliably from the repository root/package. | Convert to argument-driven scripts, use package-safe imports, and guard top-level execution. |
| P3 | Repository hygiene issues remain. | `git diff --check` reports four trailing-whitespace lines in `src_post/vorticity.py`; `make_vorticiy.py` has mojibake comments; `make_vorticiy2.py` has invalid `\mathrm` escape warnings; an untracked VS Code setting embeds a machine-local interpreter path. | Clean whitespace and encoding; use raw strings for labels; ignore or omit the local editor setting and generated figure artifacts. |

## Correct design choices

- The fallback velocity reconstruction matches the solver convention: `u_hat = i*ky*w_hat/k^2` and `v_hat = -i*kx*w_hat/k^2`.
- The fallback Laplacian has the correct spectral sign.
- Parallelism over radial bins is race-free because each worker writes a distinct leading accumulator slice.
- The saved PDF is normalized at fixed radial bin by the vorticity-bin area.
- Input datasets are closed in `finally`; output directories and a line-buffered log are created automatically.
- The code makes useful source/provenance metadata available in the NetCDF output.

## Resource estimate at the current driver settings

With `n_bins=200`:

- One `(r, w1, w2)` array has 8,000,000 entries.
- Four float64/int64 accumulation arrays require about 244 MiB before outputs and temporary arrays.
- The primary result arrays add roughly another 150--250 MiB in memory, depending on which remain live.
- Each selected 384x384 time needs roughly 21.7 billion pair/displacement visits.

This is before NetCDF buffering, Numba runtime overhead, FFT arrays, and the ten selected timesteps. The operation should be benchmarked on one time first and only then scaled.

## Test plan after repairing the environment

1. Recreate the virtual environment and install the declared dependencies.
2. Add a 4x4 synthetic periodic field with known Fourier modes; verify reconstructed `u`, `v`, and `lapw`.
3. Add a tiny pair-count test verifying PDF normalization and direct moment values.
4. Run one small NetCDF input with one selected time and assert dimensions, metadata, finite-bin behaviour, and output plot creation.
5. Ensure `pytest --collect-only` does not run an analysis job.
6. Compare a small run with a refined-grid equivalent to identify the resolved `r`, `omega`, and moment ranges.

## Interpretation warning

Passing syntax checks is not evidence that the requested PDFs or conditional averages are physically converged. The results need both grid-refinement convergence and independent-time/block convergence. The accompanying `MODIFICATION_ANALYSIS.md` describes those scientific acceptance criteria.

