from src_post.pdf1 import compute_pdf1,pdf1_to_movie
from src_post.pdf2 import compute_pdf2,pdf2_to_movie
from src_post.vorticity import vorticity_to_movie

datafolder = "data_2026-03-24_21-57-02"

vorticity_to_movie(f"{datafolder}/fluid.nc", output_file=f"{datafolder}/vorticity.mp4", fps=30)

compute_pdf1(f"{datafolder}/fluid.nc", bins=400)
pdf1_to_movie(f"{datafolder}/fluid.pdf1.nc", output_file=f"{datafolder}/pdf1.mp4", fps=30)

compute_pdf2(f"{datafolder}/fluid.nc", bins=200, shift=(1, 0))
pdf2_to_movie(f"{datafolder}/fluid.pdf2.nc", output_file=f"{datafolder}/pdf2x.mp4", fps=30, axis=0)
pdf2_to_movie(f"{datafolder}/fluid.pdf2.nc", output_file=f"{datafolder}/pdf2y.mp4", fps=30, axis=1)

