"""VGGT inference and data export utilities."""

import os
import sys
import glob
import copy
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import pycolmap


def run_vggt_inference(image_dir: str, output_dir: str, cfg: dict) -> None:
    """
    Run VGGT on a directory of images and produce COLMAP-style output.

    This function leverages the existing run_vggt_to_3dgs.py logic directly,
    calling it as a submodule rather than duplicating code.
    """
    # Add VGGT root to path to import run_vggt_to_3dgs
    vggt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vggt"))
    if vggt_root not in sys.path:
        sys.path.insert(0, vggt_root)

    # Import the VGGT pipeline module
    import run_vggt_to_3dgs as vggt_pipeline

    # Set up args namespace matching run_vggt_to_3dgs expectations
    class Args:
        pass

    args = Args()
    args.image_dir = image_dir
    args.output_dir = output_dir
    vggt_cfg = cfg.get("vggt", {})

    args.model_path = vggt_cfg.get("model_path", None)
    args.seed = 42
    args.use_ba = vggt_cfg.get("use_ba", False)
    args.max_reproj_error = vggt_cfg.get("max_reproj_error", 8.0)
    args.shared_camera = vggt_cfg.get("shared_camera", False)
    args.camera_type = vggt_cfg.get("camera_type", "SIMPLE_PINHOLE")
    args.vis_thresh = vggt_cfg.get("vis_thresh", 0.2)
    args.query_frame_num = vggt_cfg.get("query_frame_num", 8)
    args.max_query_pts = vggt_cfg.get("max_query_pts", 4096)
    args.fine_tracking = vggt_cfg.get("fine_tracking", True)
    args.conf_thres_value = vggt_cfg.get("conf_thres_value", 5.0)
    args.max_points = vggt_cfg.get("max_points", 100000)

    # Run the VGGT pipeline
    vggt_pipeline.main_with_args(args)


def export_vggt_cameras_json(sparse_dir: str, output_dir: str) -> str:
    """
    Export VGGT camera data as JSON and numpy arrays for downstream use.
    Reads from COLMAP sparse output and exports as cameras.json, intrinsics.npy, extrinsics.npy.
    """
    from videogeo.io_utils import ensure_dir, save_json, save_npy

    vggt_dir = ensure_dir(os.path.join(output_dir, "vggt"))

    # Try reading from COLMAP binary
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

    images_bin = os.path.join(sparse_dir, "images.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")

    if not os.path.exists(images_bin):
        print(f"Warning: no COLMAP binary found at {sparse_dir}")
        return vggt_dir

    extrinsics_data = read_extrinsics_binary(images_bin)
    intrinsics_data = read_intrinsics_binary(cameras_bin)

    cameras_list = []
    intrinsics_dict = {}
    extrinsics_list = []

    for i, (img_id, ext) in enumerate(sorted(extrinsics_data.items())):
        cam = intrinsics_data[ext.camera_id]
        R = qvec2rotmat(ext.qvec)
        T = np.array(ext.tvec)

        # Build world-to-camera 3x4 matrix
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        # Camera-to-world
        c2w = np.linalg.inv(w2c)

        # Intrinsics matrix
        if cam.model in ("SIMPLE_PINHOLE", "RADIAL"):
            f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        elif cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        else:
            print(f"Warning: unhandled camera model {cam.model}, using SIMPLE_PINHOLE")
            f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

        cameras_list.append({
            "id": i,
            "image_name": ext.name,
            "width": cam.width,
            "height": cam.height,
            "model": cam.model,
            "params": cam.params.tolist(),
            "c2w": c2w[:3, :].tolist(),
        })
        intrinsics_dict[ext.name] = K.tolist()
        extrinsics_list.append(c2w[:3, :].tolist())

    save_json(cameras_list, os.path.join(vggt_dir, "cameras.json"))
    save_npy(np.array([intrinsics_dict[n] for c in cameras_list for n in [c["image_name"]]]),
             os.path.join(vggt_dir, "intrinsics.npy"))
    save_npy(np.array(extrinsics_list), os.path.join(vggt_dir, "extrinsics.npy"))

    return vggt_dir


def detect_degenerate_vggt(sparse_dir: str) -> list:
    """Check for degenerate VGGT output. Returns list of warning messages."""
    warnings = []

    points3d_bin = os.path.join(sparse_dir, "points3D.bin")
    if os.path.exists(points3d_bin):
        from scene.colmap_loader import read_points3D_binary
        xyz, _, _ = read_points3D_binary(points3d_bin)
        if len(xyz) < 100:
            warnings.append(f"Too few 3D points: {len(xyz)}. Expected > 100.")

        nan_count = np.sum(np.isnan(xyz))
        if nan_count > 0:
            warnings.append(f"Found {nan_count} NaN values in 3D points.")

    # Check camera baseline
    images_bin = os.path.join(sparse_dir, "images.bin")
    if os.path.exists(images_bin):
        from scene.colmap_loader import read_extrinsics_binary
        ext_data = read_extrinsics_binary(images_bin)
        positions = np.array([e.tvec for e in ext_data.values()])
        if len(positions) > 1:
            max_dist = np.max(np.linalg.norm(positions - positions[0], axis=1))
            if max_dist < 0.01:
                warnings.append(f"Extremely small camera baseline: {max_dist:.4f}. "
                              "Trajectory may be collapsed.")

    if warnings:
        print("VGGT DEGENERACY WARNINGS:")
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("VGGT output looks healthy.")

    return warnings
