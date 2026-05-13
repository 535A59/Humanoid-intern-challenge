"""3D Gaussian Splatting integration utilities."""

import os
import sys
import subprocess
import numpy as np
from videogeo.io_utils import ensure_dir


def prepare_3dgs_input(vggt_sparse_dir: str, frames_dir: str, output_dir: str) -> str:
    """
    Prepare 3DGS-compatible data directory from VGGT output and frames.

    The VGGT pipeline (run_vggt_to_3dgs.py) already outputs the correct
    COLMAP format. This function ensures the directory structure is complete
    and creates the gs_data directory with images/ and sparse/ subdirs.

    Returns path to gs_data directory.
    """
    gs_data_dir = ensure_dir(os.path.join(output_dir, "gs_data"))
    images_dir = ensure_dir(os.path.join(gs_data_dir, "images"))
    sparse_dir_dst = ensure_dir(os.path.join(gs_data_dir, "sparse", "0"))

    # Copy/symlink images from frames or VGGT output
    import glob
    import shutil

    # Try frames_dir first, then vggt images
    src_images = glob.glob(os.path.join(frames_dir, "*.png")) + \
                 glob.glob(os.path.join(frames_dir, "*.jpg"))
    if not src_images:
        src_images = glob.glob(os.path.join(vggt_sparse_dir, "..", "images", "*.png")) + \
                     glob.glob(os.path.join(vggt_sparse_dir, "..", "images", "*.jpg"))

    for src in src_images:
        dst = os.path.join(images_dir, os.path.basename(src))
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)

    # Copy COLMAP sparse files
    for fname in ["cameras.bin", "images.bin", "points3D.bin", "points3D.ply"]:
        src = os.path.join(vggt_sparse_dir, fname)
        dst = os.path.join(sparse_dir_dst, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)

    print(f"3DGS data prepared at {gs_data_dir}")
    return gs_data_dir


def run_3dgs_training(gs_data_dir: str, output_dir: str, cfg: dict) -> str:
    """
    Run 3D Gaussian Splatting training.

    Calls train.py from gaussian-splatting as a subprocess.

    Returns path to 3dgs output directory.
    """
    gs_cfg = cfg.get("gaussian", {})
    iterations = gs_cfg.get("iterations", 7000)

    threedgs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gaussian-splatting"))
    train_script = os.path.join(threedgs_root, "train.py")
    model_output_dir = ensure_dir(os.path.join(output_dir, "3dgs"))

    # Build command
    cmd = [
        sys.executable, train_script,
        "-s", os.path.abspath(gs_data_dir),
        "-m", os.path.abspath(model_output_dir),
        "--iterations", str(iterations),
        "--test_iterations", str(iterations),
        "--save_iterations", str(iterations),
        "--eval",
        "--quiet",
    ]

    if gs_cfg.get("white_background", False):
        cmd.append("-w")

    print(f"Running 3DGS training: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=threedgs_root)

    if result.returncode != 0:
        raise RuntimeError(f"3DGS training failed with return code {result.returncode}")

    # Find the latest point cloud
    pc_dir = os.path.join(model_output_dir, "point_cloud")
    if os.path.isdir(pc_dir):
        iterations_dirs = sorted([d for d in os.listdir(pc_dir) if d.startswith("iteration_")])
        if iterations_dirs:
            latest = iterations_dirs[-1]
            print(f"3DGS training complete. Latest iteration: {latest}")
            return model_output_dir

    raise RuntimeError("3DGS training completed but no point cloud found.")


def load_gaussian_ply(ply_path: str) -> tuple:
    """
    Load a 3DGS point cloud PLY file.

    Returns (xyz, features_dc, features_rest, opacity, scaling, rotation)
    """
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)

    vert = plydata["vertex"]
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=-1)
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = vert["f_dc_0"]
    features_dc[:, 1, 0] = vert["f_dc_1"]
    features_dc[:, 2, 0] = vert["f_dc_2"]

    features_extra = np.zeros((xyz.shape[0], 3, 15))
    for i in range(15):
        key = f"f_rest_{i}"
        if key in vert:
            features_extra[:, 0, i] = vert[key]
            features_extra[:, 1, i] = vert[key]
            features_extra[:, 2, i] = vert[key]

    opacity = np.array(vert["opacity"])[:, None]
    scale_names = [n for n in vert.data.dtype.names if n.startswith("scale_")]
    rotation_names = [n for n in vert.data.dtype.names if n.startswith("rot_")]
    scaling = np.stack([vert[s] for s in scale_names], axis=-1)
    rotation = np.stack([vert[r] for r in rotation_names], axis=-1)

    return xyz, features_dc, features_extra, opacity, scaling, rotation
