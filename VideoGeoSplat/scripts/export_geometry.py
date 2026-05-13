#!/usr/bin/env python3
"""Export geometry: point clouds and optional mesh."""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogeo.io_utils import ensure_dir
from videogeo.geometry_utils import export_pointclouds, build_poisson_mesh


def main():
    parser = argparse.ArgumentParser(description="Export geometry from 3DGS/VGGT")
    parser.add_argument("--threedgs_output", type=str, required=True,
                        help="Path to 3DGS output directory")
    parser.add_argument("--vggt_sparse", type=str, required=True,
                        help="Path to VGGT sparse/0/ directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--mesh", action="store_true", help="Build Poisson mesh")
    parser.add_argument("--mesh_depth", type=int, default=9, help="Poisson depth")
    args = parser.parse_args()

    # Export point clouds
    result = export_pointclouds(args.threedgs_output, args.vggt_sparse, args.output)

    # Optional mesh
    if args.mesh:
        # Use 3DGS point cloud for mesh
        pc_path = result.get("pointcloud_3dgs")
        if pc_path and os.path.exists(pc_path):
            mesh_path = build_poisson_mesh(pc_path, args.output, depth=args.mesh_depth)
            if mesh_path:
                result["mesh"] = mesh_path
        else:
            print("No point cloud available for mesh reconstruction")

    print("Geometry export complete.")
    for k, v in result.items():
        if v and os.path.exists(v):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
