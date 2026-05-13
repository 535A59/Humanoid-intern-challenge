#!/usr/bin/env python3
"""
VideoGeoSplat — One-Command Video-to-3D Reconstruction Pipeline
================================================================
Takes a short phone video of an indoor scene and produces:
  - 3D Gaussian Splatting reconstruction
  - Point cloud / mesh geometry export
  - Rendered demo video
  - Evaluation metrics and report

Usage:
    python run_pipeline.py \
        --video examples/input_video.mp4 \
        --output outputs/example_scene \
        --fps 2 \
        --iterations 7000 \
        --mesh

Pipeline stages:
    video → frame extraction → VGGT → 3DGS → geometry export → demo → evaluation

Docker quickstart:
    docker run --gpus all -it --rm \
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      -v ~/workspace:/workspace -w /workspace/code/3DGS/SuGaR 3dgs:latest /bin/bash
    cd /workspace/code/3DGS/Humanoid-intern-challenge/VideoGeoSplat
    source ../../.venv/bin/activate
    uv run python run_pipeline.py --video examples/input_video.mp4 --output outputs/example_scene --mesh
"""

import os
import sys
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videogeo.config import load_config, merge_args_to_config
from videogeo.io_utils import ensure_dir, save_json, load_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="VideoGeoSplat: Video-to-3D Reconstruction Pipeline"
    )
    # Core I/O
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video (e.g., examples/input_video.mp4)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory (e.g., outputs/example_scene)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom YAML config file")
    # Pipeline overrides
    parser.add_argument("--fps", type=float, default=None, help="Frame extraction FPS")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to extract")
    parser.add_argument("--resize_long_edge", type=int, default=None, help="Resize long edge")
    parser.add_argument("--iterations", type=int, default=7000, help="3DGS training iterations")
    parser.add_argument("--use_ba", action="store_true", help="Use VGGT bundle adjustment")
    # Feature toggles
    parser.add_argument("--mesh", action="store_true", help="Build Poisson mesh")
    parser.add_argument("--no_mesh", action="store_true", help="Skip mesh")
    parser.add_argument("--semantic", action="store_true", help="Enable semantic labeling (Priority 3)")
    # Stage control
    parser.add_argument("--skip_vggt", action="store_true", help="Skip VGGT (use existing)")
    parser.add_argument("--skip_3dgs", action="store_true", help="Skip 3DGS training (use existing)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    return parser.parse_args()


def stage_extract_frames(cfg: dict, args, paths: dict) -> dict:
    """Stage 1: Extract frames from video."""
    from videogeo.frame_utils import extract_frames

    p = cfg.get("pipeline", {})
    meta = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps if args.fps is not None else p.get("fps", 2),
        max_frames=args.max_frames if args.max_frames is not None else p.get("max_frames", 120),
        resize_long_edge=args.resize_long_edge if args.resize_long_edge is not None else p.get("resize_long_edge", 1024),
        blur_threshold=p.get("blur_threshold", 100.0),
    )
    paths["frames_dir"] = os.path.join(args.output, "frames")
    paths["frames_meta"] = os.path.join(args.output, "frames_meta.json")
    return meta


def stage_run_vggt(cfg: dict, args, paths: dict) -> dict:
    """
    Stage 2: Run VGGT geometry estimation.

    VGGT outputs directly to gs_data/ in COLMAP format, which is also the
    3DGS input directory — no separate prepare step needed.
    """
    from videogeo.vggt_utils import run_vggt_as_subprocess, export_vggt_cameras_json, detect_degenerate_vggt

    # VGGT writes COLMAP output directly to gs_data/ (this IS the 3DGS input)
    gs_data_dir = ensure_dir(os.path.join(args.output, "gs_data"))

    if args.use_ba:
        cfg.setdefault("vggt", {})["use_ba"] = True

    ret = run_vggt_as_subprocess(paths["frames_dir"], gs_data_dir, cfg)
    if ret != 0:
        raise RuntimeError(f"VGGT failed with return code {ret}")

    sparse_dir = os.path.join(gs_data_dir, "sparse", "0")
    if not os.path.isdir(sparse_dir):
        raise RuntimeError(f"VGGT did not produce sparse/0/ at {sparse_dir}")

    paths["vggt_sparse"] = sparse_dir
    paths["gs_data"] = gs_data_dir

    # Export cameras as JSON for downstream use
    export_vggt_cameras_json(sparse_dir, args.output)
    paths["vggt_dir"] = os.path.join(args.output, "vggt")
    paths["cameras_json"] = os.path.join(args.output, "vggt", "cameras.json")

    # Quality check
    warnings = detect_degenerate_vggt(sparse_dir)
    return {"vggt_warnings": warnings}


def stage_run_3dgs(cfg: dict, args, paths: dict) -> dict:
    """Stage 3: Run 3DGS training."""
    from videogeo.gaussian_utils import run_3dgs_training

    if args.iterations is not None:
        cfg.setdefault("gaussian", {})["iterations"] = args.iterations

    output_3dgs = run_3dgs_training(paths["gs_data"], args.output, cfg)
    paths["threedgs_output"] = output_3dgs
    return {"threedgs_output": output_3dgs}


def stage_export_geometry(cfg: dict, args, paths: dict) -> dict:
    """Stage 4: Export point clouds and optional mesh."""
    from videogeo.geometry_utils import export_pointclouds, build_poisson_mesh

    result = export_pointclouds(paths["threedgs_output"], paths["vggt_sparse"], args.output)

    mesh_method = cfg.get("geometry", {}).get("mesh_method", "none")
    if args.mesh:
        mesh_method = "poisson"
    if args.no_mesh:
        mesh_method = "none"

    if mesh_method == "poisson":
        pc_path = result.get("pointcloud_3dgs")
        if pc_path and os.path.exists(pc_path):
            depth = cfg.get("geometry", {}).get("mesh_depth", 9)
            mesh_path = build_poisson_mesh(pc_path, args.output, depth=depth)
            if mesh_path:
                result["mesh"] = mesh_path

    return result


def stage_render_demo(cfg: dict, args, paths: dict) -> dict:
    """Stage 5: Generate demo outputs."""
    from videogeo.render_utils import (
        render_novel_views,
        create_flythrough_video,
        create_input_grid,
        create_camera_trajectory_plot,
    )
    import glob

    demo_cfg = cfg.get("demo", {})

    # 1. Input grid
    input_grid = create_input_grid(paths["frames_dir"], args.output)

    # 2. Camera trajectory plot
    cam_json = paths.get("cameras_json", os.path.join(args.output, "vggt", "cameras.json"))
    traj_plot = None
    if os.path.exists(cam_json):
        traj_plot = create_camera_trajectory_plot(cam_json, args.output)

    # 3. Novel view rendering (via 3DGS render.py)
    resolution = tuple(demo_cfg.get("resolution", [1024, 768]))
    renders = render_novel_views(
        paths["threedgs_output"],
        paths["gs_data"],
        args.output,
        num_views=demo_cfg.get("num_novel_views", 60),
        resolution=resolution,
    )

    # 4. Flythrough video
    if not renders:
        renders = sorted(glob.glob(
            os.path.join(paths["threedgs_output"], "test", "**", "*.png"), recursive=True
        ))
    video_path = None
    if renders:
        fps = demo_cfg.get("fps", 24)
        video_path = create_flythrough_video(renders, args.output, fps=fps, resolution=resolution)

    return {
        "input_grid": input_grid,
        "camera_trajectory": traj_plot,
        "novel_view_video": video_path,
    }


def stage_evaluate(cfg: dict, args, paths: dict) -> dict:
    """Stage 6: Compute evaluation metrics."""
    from scripts.evaluate_reconstruction import evaluate
    metrics = evaluate(args.output)
    return metrics


def generate_report(args, paths: dict, stage_results: dict, timings: dict) -> str:
    """Generate reconstruction_report.md."""
    lines = [
        "# VideoGeoSplat Reconstruction Report",
        "",
        "## Pipeline Summary",
        f"- **Input video:** `{args.video}`",
        f"- **Output directory:** `{args.output}`",
        f"- **Pipeline stages completed:** {len(stage_results)}",
        "",
        "## Timings",
    ]
    for stage, elapsed in timings.items():
        lines.append(f"- **{stage}:** {elapsed:.1f}s")

    lines.append("")
    lines.append("## Key Metrics")
    metrics = stage_results.get("evaluate", {})
    for k, v in sorted(metrics.items()):
        lines.append(f"- **{k}:** {v}")

    lines.append("")
    lines.append("## Output Structure")
    for subdir in ["frames", "vggt", "3dgs", "geometry", "demo", "gs_data"]:
        path = os.path.join(args.output, subdir)
        if os.path.exists(path):
            lines.append(f"- `{subdir}/`")

    lines.append("")
    lines.append("## Design Notes")
    lines.append(
        "This system uses VGGT as a feed-forward geometry front-end to estimate "
        "camera poses, depths, and point maps from video frames. "
        "These predictions are then used to drive a 3D Gaussian Splatting "
        "reconstruction, which provides a photorealistic and interactive scene "
        "representation. Since 3DGS is primarily optimized for rendering rather "
        "than strict surface geometry, the system also exports explicit point "
        "clouds and optional meshes for geometric inspection."
    )

    report_path = os.path.join(args.output, "reconstruction_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def get_stages_to_run(args) -> list:
    """Determine which stages to run based on CLI flags."""
    stages = []

    if not args.skip_vggt:
        stages.append("extract_frames")
        stages.append("run_vggt")
    else:
        print("Skipping VGGT stages (--skip_vggt)")

    if not args.skip_3dgs:
        stages.append("run_3dgs")
    else:
        print("Skipping 3DGS training (--skip_3dgs)")

    stages.append("export_geometry")
    stages.append("render_demo")
    stages.append("evaluate")

    return stages


STAGE_FUNCTIONS = {
    "extract_frames": stage_extract_frames,
    "run_vggt": stage_run_vggt,
    "run_3dgs": stage_run_3dgs,
    "export_geometry": stage_export_geometry,
    "render_demo": stage_render_demo,
    "evaluate": stage_evaluate,
}


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    cfg = merge_args_to_config(cfg, args)

    # Setup output directory
    ensure_dir(args.output)

    # Load checkpoint if resuming
    checkpoint = {}
    if args.resume:
        print(f"Resuming from {args.output}")
        ckpt_path = os.path.join(args.output, "pipeline_checkpoint.json")
        if os.path.exists(ckpt_path):
            checkpoint = load_json(ckpt_path)

    # Track paths through the pipeline
    paths = {
        "frames_dir": os.path.join(args.output, "frames"),
        "vggt_dir": os.path.join(args.output, "vggt"),
        "gs_data": checkpoint.get("paths", {}).get("gs_data", os.path.join(args.output, "gs_data")),
        "threedgs_output": checkpoint.get("paths", {}).get("threedgs_output", os.path.join(args.output, "3dgs")),
    }

    stages = get_stages_to_run(args)
    stage_results = {}
    timings = {}

    print("=" * 60)
    print("VideoGeoSplat Pipeline")
    print("=" * 60)
    print(f"  Input:  {args.video}")
    print(f"  Output: {args.output}")
    print(f"  Stages: {' → '.join(stages)}")
    print(f"  Config: iterations={cfg.get('gaussian', {}).get('iterations', 7000)}, "
          f"mesh={'yes' if args.mesh else 'no'}, semantic={args.semantic}")
    print("=" * 60)

    for i, stage_name in enumerate(stages, 1):
        if stage_name not in STAGE_FUNCTIONS:
            print(f"\n[{i}/{len(stages)}] {stage_name} — SKIP (not implemented)")
            continue

        print(f"\n[{i}/{len(stages)}] Running: {stage_name}")
        print("-" * 40)

        try:
            t0 = time.time()
            result = STAGE_FUNCTIONS[stage_name](cfg, args, paths)
            elapsed = time.time() - t0
            timings[stage_name] = elapsed
            stage_results[stage_name] = result
            print(f"  ✓ {stage_name} completed in {elapsed:.1f}s")

            # Save checkpoint after each stage
            save_json({
                "completed_stages": list(stage_results.keys()),
                "paths": {k: v for k, v in paths.items() if isinstance(v, str)},
            }, os.path.join(args.output, "pipeline_checkpoint.json"))

        except Exception as e:
            print(f"  ✗ {stage_name} FAILED: {e}")
            traceback.print_exc()
            print(f"\nPartial results saved to {args.output}")
            print(f"Resume with: --resume --skip_vggt --skip_3dgs")
            sys.exit(1)

    # Generate report
    print("\n" + "=" * 60)
    print("Generating report...")
    report_path = generate_report(args, paths, stage_results, timings)
    print(f"Report: {report_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Output:  {args.output}")
    print(f"Time:    {sum(timings.values()):.1f}s total")

    full_metrics = stage_results.get("evaluate", {})
    if full_metrics:
        print("\nMetrics:")
        for k, v in sorted(full_metrics.items()):
            print(f"  {k}: {v}")

    print(f"\nResults:")
    print(f"  Point cloud:  {args.output}/geometry/pointcloud_from_3dgs.ply")
    print(f"  Gaussians:    {args.output}/3dgs/point_cloud/iteration_*/point_cloud.ply")
    if args.mesh:
        print(f"  Mesh:         {args.output}/geometry/mesh.obj")
    print(f"  Demo video:   {args.output}/demo/novel_view.mp4")
    print(f"  Report:       {args.output}/reconstruction_report.md")
    print(f"  Metrics:      {args.output}/metrics.json")
    print(f"  Cameras:      {args.output}/vggt/cameras.json")


if __name__ == "__main__":
    main()
