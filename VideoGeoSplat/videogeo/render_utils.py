"""Rendering and demo-generation utilities."""

import os
import sys
import subprocess
import numpy as np
import cv2
from videogeo.io_utils import ensure_dir


def render_novel_views(
    threedgs_output: str,
    gs_data_dir: str,
    output_dir: str,
    trajectory: str = "circle",
    num_views: int = 60,
    resolution: tuple = (1024, 768),
) -> list:
    """
    Render novel views using the 3DGS render.py script.

    Returns list of rendered image paths.
    """
    threedgs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gaussian-splatting"))
    render_script = os.path.join(threedgs_root, "render.py")

    if not os.path.exists(render_script):
        print(f"render.py not found at {render_script}, skipping novel view rendering")
        return []

    demo_dir = ensure_dir(os.path.join(output_dir, "demo", "novel_views"))

    cmd = [
        sys.executable, render_script,
        "-m", os.path.abspath(threedgs_output),
        "--skip_train",
    ]

    print(f"Rendering novel views: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=threedgs_root)

    if result.returncode != 0:
        print(f"Rendering failed with return code {result.returncode}")
        return []

    # Find rendered images
    import glob
    renders = sorted(glob.glob(os.path.join(threedgs_output, "test", "*_*", "*.png")))
    return renders


def create_flythrough_video(
    render_paths: list,
    output_dir: str,
    fps: int = 24,
    resolution: tuple = (1024, 768),
) -> str:
    """Create a video from rendered novel views."""
    if not render_paths:
        return None

    demo_dir = ensure_dir(os.path.join(output_dir, "demo"))
    video_path = os.path.join(demo_dir, "novel_view.mp4")

    # Resize all renders to consistent size and write to video
    h, w = resolution[1], resolution[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for rp in render_paths:
        img = cv2.imread(rp)
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()
    print(f"Flythrough video saved to {video_path}")
    return video_path


def create_input_grid(frames_dir: str, output_dir: str, max_cols: int = 8) -> str:
    """Create a grid image from input frames."""
    import glob
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")))

    if not frames:
        return None

    demo_dir = ensure_dir(os.path.join(output_dir, "demo"))
    grid_path = os.path.join(demo_dir, "input_grid.png")

    # Select a subset of frames for the grid
    if len(frames) > 16:
        indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
        frames = [frames[i] for i in indices]

    images = [cv2.imread(f) for f in frames]
    images = [cv2.resize(img, (256, 256)) for img in images if img is not None]

    if not images:
        return None

    n = len(images)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols

    grid = np.zeros((rows * 256, cols * 256, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = i // cols, i % cols
        grid[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = img

    cv2.imwrite(grid_path, grid)
    print(f"Input grid saved to {grid_path}")
    return grid_path


def create_camera_trajectory_plot(cameras_json: str, output_dir: str) -> str:
    """Plot camera trajectory from cameras.json."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(cameras_json, "r") as f:
        cameras = json.load(f)

    positions = []
    for cam in cameras:
        c2w = np.array(cam["c2w"])
        positions.append(c2w[:3, 3])

    positions = np.array(positions)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-o", markersize=4)
    ax.scatter(*positions[0], color="green", s=50, label="Start")
    ax.scatter(*positions[-1], color="red", s=50, label="End")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Trajectory")
    ax.legend()

    demo_dir = ensure_dir(os.path.join(output_dir, "demo"))
    plot_path = os.path.join(demo_dir, "camera_trajectory.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Camera trajectory plot saved to {plot_path}")
    return plot_path
