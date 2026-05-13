#!/usr/bin/env python3
"""Extract frames from video with quality control."""

import os
import sys
import argparse

# Add parent to path so we can import videogeo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogeo.frame_utils import extract_frames


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second to extract")
    parser.add_argument("--max_frames", type=int, default=120, help="Maximum number of frames")
    parser.add_argument("--resize_long_edge", type=int, default=1024, help="Resize long edge to this size")
    parser.add_argument("--blur_threshold", type=float, default=100.0, help="Laplacian variance threshold")
    args = parser.parse_args()

    meta = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        resize_long_edge=args.resize_long_edge,
        blur_threshold=args.blur_threshold,
    )

    print(f"Extracted {meta['num_frames']} frames at {meta['image_size']}")
    print(f"Metadata saved to {args.output}/frames_meta.json")


if __name__ == "__main__":
    main()
