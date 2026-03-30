import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

def visualize_nc(filename="fluid_000000.nc", save=True):
    # Load NetCDF
    data = Dataset(filename, mode="r")

    print("Variables in file:")
    for key in data.variables:
        print("  -", key)

    # Try common variable names from Lauber turbulence code
    # Look for vorticity
    if "w" in data.variables:
        field = data.variables["w"][:]       # vorticity
        title = "Vorticity ω"
    # Look for stream function
    elif "psi" in data.variables:
        field = data.variables["psi"][:]     # stream function
        title = "Stream function ψ"
    # Velocity components
    elif "u" in data.variables and "v" in data.variables:
        u = data.variables["u"][:]
        v = data.variables["v"][:]
        field = np.sqrt(u**2 + v**2)
        title = "Velocity magnitude |u|"
    else:
        raise ValueError("No known fields found in NetCDF file!")

    field = np.squeeze(field)  # remove extra dimensions if needed

    # Plotting
    plt.figure(figsize=(6, 5))
    plt.imshow(field, origin="lower", cmap="turbo")
    plt.colorbar(label=title)
    plt.title(f"{title} ({filename})")
    plt.tight_layout()

    if save:
        outname = filename.replace(".nc", ".png")
        plt.savefig(outname, dpi=200)
        print(f"Saved: {outname}")
    else:
        plt.show()

    data.close()


if __name__ == "__main__":
    visualize_nc("fluid_000000.nc")