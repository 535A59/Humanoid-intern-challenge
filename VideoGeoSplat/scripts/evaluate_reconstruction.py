#!/usr/bin/env python3
"""Evaluate reconstruction quality and save metrics."""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogeo.io_utils import save_json, load_json


def count_gaussians(ply_path: str) -> int:
    """Count number of gaussians in a PLY file."""
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    return len(ply["vertex"])


def evaluate(output_dir: str) -> dict:
    """Compute all evaluation metrics."""
    metrics = {}

    # 1. Count input frames
    frames_dir = os.path.join(output_dir, "frames")
    if os.path.isdir(frames_dir):
        import glob
        frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        metrics["num_input_frames"] = len(frames)

    # 2. Count valid cameras
    cam_json = os.path.join(output_dir, "vggt", "cameras.json")
    if os.path.exists(cam_json):
        cameras = load_json(cam_json)
        metrics["num_valid_cameras"] = len(cameras)

    # 3. Point cloud sizes
    geo_dir = os.path.join(output_dir, "geometry")
    for name, fname in [
        ("pointcloud_vggt", "pointcloud_from_vggt.ply"),
        ("pointcloud_3dgs", "pointcloud_from_3dgs.ply"),
    ]:
        path = os.path.join(geo_dir, fname)
        if os.path.exists(path):
            try:
                from plyfile import PlyData
                ply = PlyData.read(path)
                metrics[f"{name}_points"] = len(ply["vertex"])
            except Exception:
                pass

    # 4. Gaussian count
    pc_dir = os.path.join(output_dir, "3dgs", "point_cloud")
    if os.path.isdir(pc_dir):
        iters = sorted([d for d in os.listdir(pc_dir) if d.startswith("iteration_")])
        if iters:
            ply_path = os.path.join(pc_dir, iters[-1], "point_cloud.ply")
            if os.path.exists(ply_path):
                metrics["num_gaussians"] = count_gaussians(ply_path)

    # 5. Mesh stats
    for mesh_name in ["mesh.obj", "mesh.ply"]:
        mesh_path = os.path.join(geo_dir, mesh_name)
        if os.path.exists(mesh_path):
            try:
                import trimesh
                m = trimesh.load(mesh_path)
                metrics["mesh_vertices"] = len(m.vertices)
                metrics["mesh_faces"] = len(m.faces)
                break
            except Exception:
                pass

    # 6. File sizes
    sizes = {}
    for subdir in ["frames", "vggt", "3dgs", "geometry", "demo"]:
        path = os.path.join(output_dir, subdir)
        if os.path.exists(path):
            from videogeo.io_utils import get_size_mb
            sizes[f"size_{subdir}_mb"] = round(get_size_mb(path), 1)
    metrics.update(sizes)

    # 7. Semantic coverage (if available)
    sem_labels = os.path.join(output_dir, "semantics", "semantic_labels.json")
    if os.path.exists(sem_labels):
        sem_data = load_json(sem_labels)
        # Count labeled vs total
        labeled = sum(1 for v in sem_data.values() if v.get("label") is not None)
        total = len(sem_data)
        metrics["semantic_coverage"] = round(labeled / total, 3) if total > 0 else 0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    metrics = evaluate(args.output)
    save_json(metrics, os.path.join(args.output, "metrics.json"))

    print("=" * 50)
    print("Reconstruction Metrics")
    print("=" * 50)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
