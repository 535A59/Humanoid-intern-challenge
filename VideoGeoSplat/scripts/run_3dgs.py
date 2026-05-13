#!/usr/bin/env python3
"""Run 3D Gaussian Splatting training.

The VGGT output (gs_data/) is already in the correct COLMAP format
for 3DGS to consume directly. This script wraps train.py.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogeo.io_utils import ensure_dir
from videogeo.gaussian_utils import run_3dgs_training


def main():
    parser = argparse.ArgumentParser(description="Run 3DGS training")
    parser.add_argument("--gs_data", type=str, required=True,
                        help="Path to 3DGS-compatible data (with images/ and sparse/0/)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--iterations", type=int, default=7000, help="Training iterations")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.gs_data, "sparse", "0")):
        print(f"ERROR: {args.gs_data} does not have sparse/0/ subdirectory.")
        print("Run VGGT first or ensure the data is in COLMAP format.")
        sys.exit(1)

    cfg = {"gaussian": {
        "iterations": args.iterations,
        "white_background": args.white_background,
        "eval": True,
    }}

    output_3dgs = run_3dgs_training(args.gs_data, args.output, cfg)

    # Find latest iteration
    import glob
    pc_dir = os.path.join(output_3dgs, "point_cloud")
    if os.path.isdir(pc_dir):
        iterations = sorted(glob.glob(os.path.join(pc_dir, "iteration_*")))
        if iterations:
            latest = os.path.basename(iterations[-1])
            print(f"\n3DGS training complete.")
            print(f"  Output:     {output_3dgs}")
            print(f"  Iteration:  {latest}")
            print(f"  Point cloud: {iterations[-1]}/point_cloud.ply")


if __name__ == "__main__":
    main()
