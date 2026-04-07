import os
import re
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from PIL import Image


def cleanup_png_files(folder, pattern=r".*\.png"):
    """Delete PNG files in folder matching pattern."""
    print("[Cleanup] Removing PNG files...")
    for f in os.listdir(folder):
        if re.match(pattern, f):
            try:
                os.remove(os.path.join(folder, f))
            except Exception as e:
                print(f"[Cleanup] Failed to delete {f}: {e}")
    print("[Cleanup] Done.")


def tke_pngs_to_movie(folder, output_file="tke.mp4", fps=30, dpi=150):
    """
    Create a movie from tke*.png files in a folder.
    If successful, cleanup PNG files.
    """

    print(f"[TKE] Scanning folder: {folder}")

    # ---- collect files ----
    pattern = re.compile(r"tke(\d+)\.png")
    files = []

    for f in os.listdir(folder):
        match = pattern.match(f)
        if match:
            files.append((int(match.group(1)), f))

    if not files:
        print("[TKE] No matching files found.")
        return False

    # ---- sort numerically ----
    files.sort(key=lambda x: x[0])
    file_list = [os.path.join(folder, f[1]) for f in files]

    print(f"[TKE] Found {len(file_list)} frames.")

    # ---- load first image to get shape ----
    first_img = Image.open(file_list[0])
    fig, ax = plt.subplots()
    img = ax.imshow(first_img)
    ax.axis('off')

    # ---- update function ----
    def update(frame):
        im = Image.open(file_list[frame])
        img.set_data(im)
        progress = (frame + 1) / len(file_list) * 100
        sys.stdout.write(f"\r[TKE] Rendering frame {frame+1}/{len(file_list)} ({progress:.1f}%)")
        sys.stdout.flush()
        return [img]

    anim = animation.FuncAnimation(fig, update, frames=len(file_list), blit=True)

    success = False

    # ---- save movie ----
    try:
        mpl.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # adjust path if needed
        writer = FFMpegWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=dpi)
        print(f"\n[TKE] Movie saved: {output_file}")
        success = True
    except Exception as e:
        print("\n[TKE] FFmpeg failed, fallback to GIF:", e)
        try:
            gif_file = output_file.replace(".mp4", ".gif")
            anim.save(gif_file, writer=PillowWriter(fps=fps))
            print(f"[TKE] Saved GIF: {gif_file}")
            success = True
        except Exception as e2:
            print("[TKE] GIF creation also failed:", e2)

    plt.close(fig)

    # ---- cleanup only if successful ----
    if success:
        cleanup_png_files(folder, pattern=r"tke\d+\.png")
    else:
        print("[TKE] Skipping cleanup due to failure.")

    return success