"""Geometry export: point cloud and mesh from 3DGS / VGGT output."""

import os
import sys
import numpy as np
import trimesh
from videogeo.io_utils import ensure_dir


def export_pointclouds(threedgs_output: str, vggt_sparse_dir: str, output_dir: str) -> dict:
    """
    Export point clouds from 3DGS and VGGT.

    Returns dict with paths to exported point clouds.
    """
    geo_dir = ensure_dir(os.path.join(output_dir, "geometry"))
    result = {}

    # 1. Export 3DGS gaussian centers as PLY
    pc_dir = os.path.join(threedgs_output, "point_cloud")
    if os.path.isdir(pc_dir):
        iters = sorted([d for d in os.listdir(pc_dir) if d.startswith("iteration_")])
        if iters:
            latest_pc = os.path.join(pc_dir, iters[-1], "point_cloud.ply")
            if os.path.exists(latest_pc):
                from plyfile import PlyData
                plydata = PlyData.read(latest_pc)
                vert = plydata["vertex"]
                xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=-1)
                colors = np.stack([
                    np.clip(vert["f_dc_0"], 0, 1),
                    np.clip(vert["f_dc_1"], 0, 1),
                    np.clip(vert["f_dc_2"], 0, 1),
                ], axis=-1)
                colors_uint8 = (colors * 255).astype(np.uint8)

                dst = os.path.join(geo_dir, "gaussians.ply")
                trimesh.PointCloud(xyz, colors=colors_uint8).export(dst)
                result["gaussians"] = dst
                print(f"Exported gaussians: {len(xyz)} points")

                # Simplified point cloud (centers only)
                dst_pc = os.path.join(geo_dir, "pointcloud_from_3dgs.ply")
                trimesh.PointCloud(xyz, colors=colors_uint8).export(dst_pc)
                result["pointcloud_3dgs"] = dst_pc

    # 2. Export VGGT sparse point cloud
    vggt_ply = os.path.join(vggt_sparse_dir, "points3D.ply")
    if os.path.exists(vggt_ply):
        dst = os.path.join(geo_dir, "pointcloud_from_vggt.ply")
        import shutil
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(vggt_ply), dst)
            except OSError:
                shutil.copy2(vggt_ply, dst)
        result["pointcloud_vggt"] = dst
        print(f"VGGT point cloud linked to {dst}")

    return result


def build_poisson_mesh(point_cloud_path: str, output_dir: str, depth: int = 9) -> str:
    """
    Build mesh from point cloud using Open3D Poisson surface reconstruction.

    Returns path to mesh file.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Install with: pip install open3d")
        return None

    pcd = o3d.io.read_point_cloud(point_cloud_path)
    if len(pcd.points) == 0:
        print("Empty point cloud, cannot build mesh.")
        return None

    # Estimate normals if needed
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Remove low-density vertices
    if len(densities) > 0:
        density_thresh = np.quantile(densities, 0.1)
        vertices_to_remove = densities < density_thresh
        mesh.remove_vertices_by_mask(vertices_to_remove)

    geo_dir = ensure_dir(os.path.join(output_dir, "geometry"))
    mesh_path = os.path.join(geo_dir, "mesh.obj")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Poisson mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # Also save as PLY
    mesh_ply = os.path.join(geo_dir, "mesh.ply")
    o3d.io.write_triangle_mesh(mesh_ply, mesh)

    return mesh_path


def build_tsdf_mesh(
    depth_dir: str,
    cameras_json: str,
    output_dir: str,
    voxel_size: float = 0.02,
) -> str:
    """
    Build mesh using TSDF fusion from VGGT depth maps and camera poses.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed.")
        return None
    import json

    with open(cameras_json, "r") as f:
        cameras = json.load(f)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4.0 * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    for cam in cameras:
        depth_path = os.path.join(depth_dir, f"{cam['image_name']}")
        color_path = None  # Use the frames

        if not os.path.exists(depth_path):
            continue

        depth = o3d.io.read_image(depth_path)
        # Build intrinsics
        K = cam.get("K", None)
        if K is None and "params" in cam:
            w, h = cam["width"], cam["height"]
            if cam["model"] == "SIMPLE_PINHOLE":
                f, cx, cy = cam["params"][0], cam["params"][1], cam["params"][2]
                intrinsic.set_intrinsics(w, h, f, f, cx, cy)
            elif cam["model"] == "PINHOLE":
                fx, fy, cx, cy = cam["params"][0], cam["params"][1], cam["params"][2], cam["params"][3]
                intrinsic.set_intrinsics(w, h, fx, fy, cx, cy)

        # Camera-to-world from JSON
        c2w = np.array(cam["c2w"])
        extrinsic = np.linalg.inv(c2w)  # world-to-camera for Open3D

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8)),
            depth,
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()
    geo_dir = ensure_dir(os.path.join(output_dir, "geometry"))
    mesh_path = os.path.join(geo_dir, "mesh_tsdf.obj")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"TSDF mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    return mesh_path
