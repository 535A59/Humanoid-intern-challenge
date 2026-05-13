#!/usr/bin/env python3
"""Generate demo visualizations and report."""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogeo.io_utils import ensure_dir, save_json
from videogeo.render_utils import (
    render_novel_views,
    create_flythrough_video,
    create_input_grid,
    create_camera_trajectory_plot,
)


def main():
    parser = argparse.ArgumentParser(description="Generate demo outputs")
    parser.add_argument("--threedgs_output", type=str, required=True,
                        help="Path to 3DGS output directory")
    parser.add_argument("--gs_data", type=str, required=True,
                        help="Path to 3DGS data directory (gs_data)")
    parser.add_argument("--frames", type=str, required=True, help="Frames directory")
    parser.add_argument("--cameras_json", type=str, required=True, help="Path to cameras.json")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--trajectory", type=str, default="circle", help="Novel view trajectory")
    parser.add_argument("--num_views", type=int, default=60, help="Number of novel views")
    args = parser.parse_args()

    demo_dir = ensure_dir(os.path.join(args.output, "demo"))

    # 1. Input grid
    frames_dir = args.frames if os.path.isdir(args.frames) else os.path.join(args.output, "frames")
    input_grid = create_input_grid(frames_dir, args.output)

    # 2. Camera trajectory plot
    cam_json = args.cameras_json
    if not os.path.exists(cam_json):
        # Try alternate path
        cam_json = os.path.join(args.output, "vggt", "cameras.json")
    if os.path.exists(cam_json):
        traj_plot = create_camera_trajectory_plot(cam_json, args.output)
    else:
        print(f"Warning: cameras.json not found at {cam_json}")
        traj_plot = None

    # 3. Novel view rendering
    resolution = (1024, 768)
    gs_data = args.gs_data if os.path.isdir(args.gs_data) else os.path.join(args.output, "gs_data")
    renders = render_novel_views(
        args.threedgs_output,
        gs_data,
        args.output,
        trajectory=args.trajectory,
        num_views=args.num_views,
        resolution=resolution,
    )

    # 4. Flythrough video
    if renders:
        video_path = create_flythrough_video(renders, args.output, resolution=resolution)
    else:
        # Try to find renders in the default 3DGS output location
        import glob
        renders = sorted(glob.glob(os.path.join(args.threedgs_output, "test", "**", "*.png"), recursive=True))
        if renders:
            video_path = create_flythrough_video(renders, args.output, resolution=resolution)
        else:
            print("No rendered views found. Skipping flythrough video.")
            video_path = None

    # 5. Summary
    demo_files = {
        "input_grid": input_grid,
        "camera_trajectory": traj_plot,
        "novel_view_video": video_path,
    }
    save_json(demo_files, os.path.join(demo_dir, "demo_manifest.json"))

    print("Demo generation complete.")
    for k, v in demo_files.items():
        if v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
