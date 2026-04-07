from src_post.pdf1 import compute_pdf1,pdf1_to_movie
from src_post.pdf2 import compute_pdf2,pdf2_to_movie
from src_post.velocity_check import vorticity_check_movie
from src_post.vorticity import vorticity_to_movie
from src_post.tke import tke_pngs_to_movie, cleanup_png_files

datafolder = "data_2026-04-03_10-52-51"

#vorticity_check_movie(f"{datafolder}/fluid.nc", output_file=f"{datafolder}/vorticity_check.mp4", fps=30 )

tke_pngs_to_movie(f"{datafolder}", output_file=f"{datafolder}/tke.mp4", fps=30, dpi=150)

vorticity_to_movie(f"{datafolder}/fluid.nc", output_file=f"{datafolder}/vorticity.mp4", fps=30)
cleanup_png_files(f"{datafolder}", pattern=r"vorticity\d+\.png")

compute_pdf1(f"{datafolder}/fluid.nc", bins=400)
pdf1_to_movie(f"{datafolder}/fluid.pdf1.nc", output_file=f"{datafolder}/pdf1.mp4", fps=30)

#compute_pdf2(f"{datafolder}/fluid.nc", bins=200, shift=(1, 0))
#pdf2_to_movie(f"{datafolder}/fluid.pdf2.nc", output_file=f"{datafolder}/pdf2x.mp4", fps=30, axis=0)
#pdf2_to_movie(f"{datafolder}/fluid.pdf2.nc", output_file=f"{datafolder}/pdf2y.mp4", fps=30, axis=1)

