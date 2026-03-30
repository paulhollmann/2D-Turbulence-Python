import time as t
import numpy as np
from src.fluid import Fluid
from src.field import *
import csv
import matplotlib.pyplot as plt
import matplotlib

import os
from datetime import datetime

matplotlib.use("Agg") # dont open any figures

# Create a folder with current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data_folder = f"data_{timestamp}"
os.makedirs(data_folder, exist_ok=True)

nx = 384
ny = 384
Re = 360.0  
# should be ok for 256x256. nu = 1/Re = 0.005
# Re_lambda ca 60–70 depending on initialization
# for a box of 2pi x 2pi


flow = Fluid(nx, ny, Re) # gridx , grid_y, ReynoldsNo 
flow.init_solver()
flow.init_field(McWilliams)
init_tke = flow.compute_initial_tke(McWilliams)*0.99
#flow.init_spectral_forcing(target_TKE=init_tke, injection_tau=0.6, kf_min=15, kf_max=40)
#flow.init_large_scale_damping(kd_max=2, drag_coeff=0.15)

flow.print_solver_params()
with open(f"{data_folder}/_params.txt", "w") as f:
    f.write(flow.get_solver_params())


#finish = # we finish iff TKE(t) < 0.15 * TKE(t=0) = 0.15 * 0.47991325459459433
start_time = t.time()

# --- CSV setup ---
csv_file = f"{data_folder}/log.csv"
csv_fields = ["iteration", "time", "TKE", "alpha"]
csv_buffer = []

# --- WRITE setup ---
write_steps = 200      # write to CSV + fluid file every XX steps
print_steps = 100      # print to screen every XX steps
display_steps = 600    # take a picture every XX steps; -1 = off


with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_fields)   # header
    tke = 1
    current_time = 0
    
    # main loop
    while current_time < 15:
        flow.update()
        iteration = flow.it
        current_time = flow.time
        tke = flow.tke()
        alpha = flow.alpha

        csv_buffer.append([iteration, current_time, tke, alpha])
        if iteration == 1 and display_steps > 0:
            fig = flow.display(show=False)
            fig.savefig(f"{data_folder}/vorticity{iteration:06d}.png", dpi=300)
            plt.close(fig)
            fig = flow.plot_spec(show=False)
            fig.savefig(f"{data_folder}/tke{iteration:06d}.png", dpi=300)
            plt.close(fig)

        # write csv every write_steps
        if iteration % write_steps == 0:
            # flush buffer to CSV
            writer.writerows(csv_buffer)
            csv_buffer = []  # reset buffer

        # print to screen every print_steps
        if iteration % print_steps == 0:
            tke_percentage = tke * 100.0 / init_tke
            print(f"Iteration {iteration:06d}, time {current_time:.4f}, TKE: {tke:.6f} ({tke_percentage:.4f}%), alpha: {alpha:.6f}")

        if display_steps != -1 and iteration % display_steps == 0:
            fig = flow.display(show=False)
            fig.savefig(f"{data_folder}/vorticity{iteration:06d}.png", dpi=300)
            plt.close(fig)
            fig = flow.plot_spec(show=False)
            fig.savefig(f"{data_folder}/tke{iteration:06d}.png", dpi=300)
            plt.close(fig)

        # write fluid file every write_steps
        if iteration % write_steps == 0:
            fluid_filename = f"{data_folder}/fluid"
            flow.write(file=fluid_filename)

end_time = t.time()
print("\nExecution time for %d iterations is %f seconds."
      % (flow.it, end_time - start_time))