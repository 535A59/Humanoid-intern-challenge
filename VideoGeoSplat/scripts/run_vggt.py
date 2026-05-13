#!/usr/bin/env python3
"""Run VGGT geometry estimation on extracted frames."""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogeo.io_utils import ensure_dir
from videogeo.vggt_utils import run_vggt_as_subprocess, export_vggt_cameras_json, detect_degenerate_vggt


def main():
    parser = argparse.ArgumentParser(description="Run VGGT on frames")
    parser.add_argument("--frames", type=str, required=True, help="Directory of extracted frames")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--use_ba", action="store_true", help="Use bundle adjustment")
    parser.add_argument("--conf_thres", type=float, default=5.0, help="Confidence threshold")
    parser.add_argument("--max_points", type=int, default=100000, help="Max 3D points")
    args = parser.parse_args()

    # VGGT output goes into a temp location matching COLMAP format
    vggt_colmap_dir = ensure_dir(os.path.join(args.output, "vggt_colmap"))

    cfg = {"vggt": {
        "use_ba": args.use_ba,
        "conf_thres_value": args.conf_thres,
        "max_points": args.max_points,
    }}

    ret = run_vggt_as_subprocess(args.frames, vggt_colmap_dir, cfg)
    if ret != 0:
        print(f"ERROR: VGGT failed with return code {ret}")
        sys.exit(1)

    sparse_dir = os.path.join(vggt_colmap_dir, "sparse", "0")
    if not os.path.isdir(sparse_dir):
        print(f"ERROR: VGGT did not produce sparse/0/ at {sparse_dir}")
        sys.exit(1)

    # Export cameras for downstream use
    vggt_dir = export_vggt_cameras_json(sparse_dir, args.output)

    # Quality check
    warnings = detect_degenerate_vggt(sparse_dir)
    if warnings:
        print(f"{len(warnings)} degeneracy warning(s) detected — see above.")

    print(f"VGGT complete. COLMAP data at {vggt_colmap_dir}")
    print(f"Camera JSON at {vggt_dir}/cameras.json")


if __name__ == "__main__":
    main()
