
from src_post.pdf1 import compute_pdf1,pdf1_to_movie
from src_post.pdf2 import compute_pdf2,pdf2_to_movie
from src_post.velocity_check import vorticity_check_movie
from src_post.vorticity import vorticity_to_movie
from src_post.tke import tke_pngs_to_movie, cleanup_png_files
from src_post.cond_avg import compute_conditional_pdf_selected_timesteps_parallel
from src_post.cond_avg_with_pdf_moments import compute_conditional_pdf_selected_timesteps_parallel as compute_conditional_pdf_selected_timesteps_parallel_with_moments
from src_post.cond_avg_with_pdf_moments_and_plots import plot_conditional_pdf_moments

datafolder = "data_2d_turbulence_long"
reynolds_number = 200


#vorticity_check_movie(f"{datafolder}/fluid.nc", output_file=f"{datafolder}/vorticity_check.mp4", fps=30 )

#tke_pngs_to_movie(f"{datafolder}", output_file=f"{datafolder}/tke.mp4", fps=30, dpi=150)

#vorticity_to_movie(f"{datafolder}/fluid.nc", output_file=f"{datafolder}/vorticity.mp4", fps=30)
#cleanup_png_files(f"{datafolder}", pattern=r"vorticity\d+\.png")

#compute_pdf1(f"{datafolder}/fluid.nc", bins=400)
#pdf1_to_movie(f"{datafolder}/fluid.pdf1.nc", output_file=f"{datafolder}/pdf1.mp4", fps=30)

#compute_pdf2(f"{datafolder}/fluid.nc", bins=200, shift=(1, 0))
#pdf2_to_movie(f"{datafolder}/fluid.pdf2.nc", output_file=f"{datafolder}/pdf2x.mp4", fps=30, axis=0)
#pdf2_to_movie(f"{datafolder}/fluid.pdf2.nc", output_file=f"{datafolder}/pdf2y.mp4", fps=30, axis=1)


#compute_conditional_pdf_fast(input_file=f"{datafolder}/fluid.nc", output_dir=f"{datafolder}/conditional_outputs", n_bins=200,)

#compute_conditional_pdf_selected_timesteps_parallel(input_file=f"{datafolder}/fluid.nc", output_dir=f"{datafolder}/conditional_outputs", Re=reynolds_number, n_bins=200, n_threads=96,)



compute_conditional_pdf_selected_timesteps_parallel_with_moments(
    input_file=f"{datafolder}/fluid.nc",
    output_dir=f"{datafolder}/conditional_outputs",
    Re=reynolds_number,
    n_bins=200,
    n_threads=96,
    timesteps=[1, 10, 100, 500, 1000, 2000, 5000, 10000, 20000, 30000],
    # New part: compute <|w1-w2|^k>(t,r)
    moment_orders=(1, 2, 3, 4, 5, 6),
)


plot_conditional_pdf_moments(f"{datafolder}/conditional_outputs")