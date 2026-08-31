from netCDF4 import Dataset


path = 'P:\\git\\2D-Turbulence-Python\\data_2026-04-03_11-00-34\\'

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