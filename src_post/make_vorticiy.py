import numpy as np
import vorticity

times = [ 1.0, 5.0, 10.0, 15.0 ]
files = [
    "data_2d_turbulence_long/fluid.nc",
    "data_2026-04-03_10-52-51/fluid.nc",
    "data_2026-04-03_11-00-34/fluid.nc",
    "data_2026-04-03_11-01-28/fluid.nc"
]   
names = [
    "Re200",
    "Re200F",
    "Re360F",
    "Re360"
]
# ------------------------------------------------------------
# ✅ choose dataset index
# ------------------------------------------------------------
i = 2  # <-- change this (0,1,2,3)

file = files[i]
name = names[i]

print(f"[Processing] {name}")

# ------------------------------------------------------------
# ✅ STEP 1: scaling using YOUR function (only this dataset)
# ------------------------------------------------------------
vmin_global = +np.inf
vmax_global = -np.inf

for t in times:
    _, vmin, vmax = vorticity.get_vorticity_minmax(file, t)
    vmin_global = min(vmin_global, vmin)
    vmax_global = max(vmax_global, vmax)

# symmetric scaling
vabs = max(abs(vmin_global), abs(vmax_global))

#if name has "F", use a more focused scaling (for better visualization)
if "F" in name:
    vabs_percent = 0.3 * vabs 
else:   
    vabs_percent = 0.9 * vabs  
vmin_global, vmax_global = -vabs_percent, vabs_percent

print(f"[Scaling vabs_percent] {vmin_global:.3e} → {vmax_global:.3e}")

# ------------------------------------------------------------
# ✅ STEP 2: save snapshots
# ------------------------------------------------------------
for t in times:
    output_name = f"{name}_t{t:.0f}.eps"

    vorticity.save_vorticity_snapshot_pub(
        input_file=file,
        target_time=t,
        output_path=output_name,
        vmin=vmin_global,
        vmax=vmax_global
    )