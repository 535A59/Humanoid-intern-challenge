#!/usr/bin/env python3
"""Run 3D Gaussian Splatting training."""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogeo.io_utils import ensure_dir
from videogeo.gaussian_utils import prepare_3dgs_input, run_3dgs_training


def main():
    parser = argparse.ArgumentParser(description="Run 3DGS training")
    parser.add_argument("--vggt_sparse", type=str, required=True,
                        help="Path to VGGT sparse/0/ directory")
    parser.add_argument("--frames", type=str, required=True,
                        help="Directory of extracted frames")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--iterations", type=int, default=7000, help="Training iterations")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    args = parser.parse_args()

    # Prepare 3DGS-compatible data directory
    gs_data_dir = prepare_3dgs_input(args.vggt_sparse, args.frames, args.output)

    # Run training
    cfg = {"gaussian": {
        "iterations": args.iterations,
        "white_background": args.white_background,
        "eval": True,
    }}

    output_3dgs = run_3dgs_training(gs_data_dir, args.output, cfg)
    print(f"3DGS training complete. Output at {output_3dgs}")


if __name__ == "__main__":
    main()
