"""VGGT inference and data export utilities.

Design note: VGGT inference is done by calling run_vggt_to_3dgs.py as a
subprocess (to match the user's existing Docker workflow). This module provides
helper functions for post-processing VGGT output and quality checks.
"""

import os
import sys
import subprocess
import numpy as np
from videogeo.io_utils import ensure_dir, save_json, save_npy


def run_vggt_as_subprocess(image_dir: str, output_dir: str, cfg: dict) -> int:
    """
    Run VGGT inference by calling the existing run_vggt_to_3dgs.py as a subprocess.

    Returns the subprocess return code.
    """
    vggt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vggt"))
    vggt_script = os.path.join(vggt_root, "run_vggt_to_3dgs.py")

    vggt_cfg = cfg.get("vggt", {})

    cmd = [
        sys.executable, vggt_script,
        "--image_dir", os.path.abspath(image_dir),
        "--output_dir", os.path.abspath(output_dir),
        "--conf_thres_value", str(vggt_cfg.get("conf_thres_value", 5.0)),
        "--max_points", str(vggt_cfg.get("max_points", 100000)),
    ]

    if vggt_cfg.get("model_path"):
        cmd.extend(["--model_path", vggt_cfg["model_path"]])
    if vggt_cfg.get("use_ba", False):
        cmd.append("--use_ba")
        cmd.extend(["--max_reproj_error", str(vggt_cfg.get("max_reproj_error", 8.0))])
        cmd.extend(["--vis_thresh", str(vggt_cfg.get("vis_thresh", 0.2))])
        cmd.extend(["--query_frame_num", str(vggt_cfg.get("query_frame_num", 8))])
        cmd.extend(["--max_query_pts", str(vggt_cfg.get("max_query_pts", 4096))])
        if vggt_cfg.get("shared_camera", False):
            cmd.append("--shared_camera")
        cmd.extend(["--camera_type", vggt_cfg.get("camera_type", "SIMPLE_PINHOLE")])

    print(f"Running VGGT: {' '.join(cmd)}")

    # Ensure VGGT package is importable (add vggt_root to PYTHONPATH)
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    if vggt_root not in existing_path:
        env["PYTHONPATH"] = f"{vggt_root}:{existing_path}" if existing_path else vggt_root

    result = subprocess.run(cmd, cwd=vggt_root, env=env)
    return result.returncode


def export_vggt_cameras_json(sparse_dir: str, output_dir: str) -> str:
    """
    Read COLMAP binary output from VGGT and export as cameras.json,
    intrinsics.npy, extrinsics.npy for downstream use.
    """
    vggt_dir = ensure_dir(os.path.join(output_dir, "vggt"))
    images_bin = os.path.join(sparse_dir, "images.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")

    if not os.path.exists(images_bin) or not os.path.exists(cameras_bin):
        print(f"Warning: missing COLMAP binaries in {sparse_dir}")
        return vggt_dir

    # Add gaussian-splatting to path for colmap_loader
    threedgs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gaussian-splatting"))
    if threedgs_root not in sys.path:
        sys.path.insert(0, threedgs_root)

    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

    ext_data = read_extrinsics_binary(images_bin)
    int_data = read_intrinsics_binary(cameras_bin)

    cameras_list = []
    intrinsics_array = []
    extrinsics_array = []

    for i, (img_id, ext) in enumerate(sorted(ext_data.items())):
        cam = int_data[ext.camera_id]
        R = qvec2rotmat(ext.qvec)
        T = np.array(ext.tvec)

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)

        # Build intrinsics matrix from camera model
        if cam.model in ("SIMPLE_PINHOLE", "RADIAL"):
            f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        elif cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        else:
            print(f"Warning: unhandled camera model {cam.model}")
            f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

        cameras_list.append({
            "id": i,
            "image_name": ext.name,
            "width": cam.width,
            "height": cam.height,
            "model": cam.model,
            "params": cam.params.tolist(),
            "c2w": c2w[:3, :].tolist(),
            "K": K.tolist(),
        })
        intrinsics_array.append(K)
        extrinsics_array.append(c2w[:3, :])

    save_json(cameras_list, os.path.join(vggt_dir, "cameras.json"))
    save_npy(np.stack(intrinsics_array), os.path.join(vggt_dir, "intrinsics.npy"))
    save_npy(np.stack(extrinsics_array), os.path.join(vggt_dir, "extrinsics.npy"))

    print(f"Exported {len(cameras_list)} cameras to {vggt_dir}")
    return vggt_dir


def detect_degenerate_vggt(sparse_dir: str) -> list:
    """Check for degenerate VGGT output. Returns list of warning messages."""
    warnings = []
    threedgs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gaussian-splatting"))
    if threedgs_root not in sys.path:
        sys.path.insert(0, threedgs_root)

    points3d_bin = os.path.join(sparse_dir, "points3D.bin")
    if os.path.exists(points3d_bin):
        from scene.colmap_loader import read_points3D_binary
        xyz, _, _ = read_points3D_binary(points3d_bin)
        if len(xyz) < 100:
            warnings.append(f"Too few 3D points: {len(xyz)}. Expected >= 100.")
        nan_count = np.sum(np.isnan(xyz))
        if nan_count > 0:
            warnings.append(f"Found {nan_count} NaN values in 3D points.")

    images_bin = os.path.join(sparse_dir, "images.bin")
    if os.path.exists(images_bin):
        from scene.colmap_loader import read_extrinsics_binary
        ext = read_extrinsics_binary(images_bin)
        positions = np.array([e.tvec for e in ext.values()])
        if len(positions) > 1:
            max_dist = np.max(np.linalg.norm(positions - positions[0], axis=1))
            if max_dist < 0.01:
                warnings.append(f"Extremely small camera baseline: {max_dist:.4f}")

    if warnings:
        print("VGGT DEGENERACY WARNINGS:")
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("VGGT output looks healthy.")
    return warnings
