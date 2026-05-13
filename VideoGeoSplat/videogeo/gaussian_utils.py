"""3D Gaussian Splatting training utilities.

VGGT's run_vggt_to_3dgs.py already outputs COLMAP-compatible data
(images/ + sparse/0/{cameras.bin, images.bin, points3D.bin, points3D.ply}),
which 3DGS train.py can consume directly. This module wraps the training call.
"""

import os
import sys
import subprocess
import numpy as np
from videogeo.io_utils import ensure_dir


def verify_gs_data(data_dir: str) -> bool:
    """Verify that data_dir is valid 3DGS input (has images/ and sparse/0/)."""
    images_ok = os.path.isdir(os.path.join(data_dir, "images"))
    sparse_ok = os.path.isfile(os.path.join(data_dir, "sparse", "0", "cameras.bin"))
    if not images_ok:
        print(f"WARNING: {data_dir}/images/ not found")
    if not sparse_ok:
        print(f"WARNING: {data_dir}/sparse/0/cameras.bin not found")
    return images_ok and sparse_ok


def run_3dgs_training(gs_data_dir: str, output_dir: str, cfg: dict) -> str:
    """
    Run 3D Gaussian Splatting training.

    Calls train.py from gaussian-splatting as a subprocess.

    Args:
        gs_data_dir: Path to COLMAP-formatted data (with images/ and sparse/0/)
        output_dir: Parent output directory (3dgs/ will be created under it)
        cfg: Pipeline config dict

    Returns:
        Path to 3dgs output directory (output_dir/3dgs/)
    """
    gs_cfg = cfg.get("gaussian", {})
    iterations = gs_cfg.get("iterations", 7000)

    threedgs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gaussian-splatting"))
    train_script = os.path.join(threedgs_root, "train.py")
    model_output_dir = ensure_dir(os.path.join(output_dir, "3dgs"))

    # Build command matching train.py's argument interface
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

    print(f"Running 3DGS training (iterations={iterations})...")
    print(f"  Data:   {gs_data_dir}")
    print(f"  Output: {model_output_dir}")

    result = subprocess.run(cmd, cwd=threedgs_root)

    if result.returncode != 0:
        raise RuntimeError(f"3DGS training failed with return code {result.returncode}")

    # Verify output
    pc_dir = os.path.join(model_output_dir, "point_cloud")
    if os.path.isdir(pc_dir):
        iterations_dirs = sorted([d for d in os.listdir(pc_dir) if d.startswith("iteration_")])
        if iterations_dirs:
            latest = iterations_dirs[-1]
            print(f"3DGS training complete. Final iteration: {latest}")
            return model_output_dir

    raise RuntimeError("3DGS training completed but no point cloud output found.")


def load_gaussian_ply(ply_path: str) -> dict:
    """
    Load a 3DGS point cloud PLY file.

    Returns dict with keys: xyz, features_dc, opacity, scaling, rotation
    """
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    vert = plydata["vertex"]

    xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=-1)

    features_dc = np.zeros((xyz.shape[0], 3))
    for i, c in enumerate(["f_dc_0", "f_dc_1", "f_dc_2"]):
        if c in vert:
            features_dc[:, i] = vert[c]

    opacity = np.array(vert["opacity"]) if "opacity" in vert else np.zeros(xyz.shape[0])

    scale_names = sorted([n for n in vert.data.dtype.names if n.startswith("scale_")])
    rot_names = sorted([n for n in vert.data.dtype.names if n.startswith("rot_")])

    scaling = np.stack([vert[s] for s in scale_names], axis=-1) if scale_names else np.zeros((xyz.shape[0], 3))
    rotation = np.stack([vert[r] for r in rot_names], axis=-1) if rot_names else np.zeros((xyz.shape[0], 4))

    return {
        "xyz": xyz,
        "features_dc": features_dc,
        "opacity": opacity,
        "scaling": scaling,
        "rotation": rotation,
    }
