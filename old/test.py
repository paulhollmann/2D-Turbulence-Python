import time as t
import numpy as np
from src.fluid import Fluid
from src.field import *
import csv

import os
from datetime import datetime

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
flow.init_field(DecayingTurbulence)

nu = 1.0 / Re
print(f"nu = {nu:.5f}")

#finish = # we finish iff TKE(t) < 0.15 * TKE(t=0) = 0.15 * 0.47991325459459433
start_time = t.time()

# --- CSV setup ---
csv_file = f"{data_folder}/log.csv"
csv_fields = ["iteration", "time", "TKE"]
csv_buffer = []

# --- WRITE setup ---
write_steps = 50      # write to CSV + fluid file every XX steps
print_steps = 100


with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_fields)   # header

    tke = 1
    # main loop
    while tke >= 0.47991325459459433 * 0.15:
        flow.update()

        iteration = flow.it
        current_time = flow.time
        #time_remaining = finish - flow.time
        tke = flow.tke()
        csv_buffer.append([iteration, current_time, tke])

        # write csv every write_steps
        if iteration % write_steps == 0:
            # flush buffer to CSV
            writer.writerows(csv_buffer)
            csv_buffer = []  # reset buffer

        # print to screen every print_steps
        if iteration % print_steps == 0:
            print(f"Iteration \t {iteration:06d}, time \t {current_time:.4f}, "
                  f"TKE: {tke:.6f}")

        # write fluid file every write_steps
        if iteration % write_steps == 0:
            fluid_filename = f"{data_folder}/fluid"
            flow.write(file=fluid_filename)

end_time = t.time()
print("\nExecution time for %d iterations is %f seconds."
      % (flow.it, end_time - start_time))