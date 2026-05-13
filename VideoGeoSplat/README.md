# VideoGeoSplat

**One-command video-to-3D reconstruction pipeline** — from a phone video to a 3D Gaussian Splatting scene, point cloud, mesh, and rendered demo.

## Overview

VideoGeoSplat takes a short phone video of an indoor scene and produces:
- **3D Gaussian Splatting (3DGS) reconstruction** — photorealistic novel-view synthesis
- **Point cloud geometry** — explicit 3D points from both VGGT and 3DGS
- **Optional Poisson mesh** — surface reconstruction from the point cloud
- **Rendered demo video** — flythrough of the reconstructed scene
- **Evaluation metrics and report** — summary of reconstruction quality

## Why VGGT + 3DGS

**VGGT** is a feed-forward vision transformer that estimates camera poses, depth maps, and point maps directly from video frames in a single forward pass — no per-scene optimization needed. This replaces the traditional COLMAP pipeline (SfM + MVS) with a single GPU-friendly model, making the geometry front-end fast and robust.

**3D Gaussian Splatting (3DGS)** takes the camera poses and sparse point cloud from VGGT and optimizes a dense, photo-realistic representation suitable for real-time rendering from novel viewpoints.

This combination gives us both speed (no multi-hour SfM) and quality (state-of-the-art novel-view synthesis).

## Quick Start (Docker)

```bash
# Launch the 3DGS Docker container
docker run --gpus all -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ~/workspace:/workspace \
  -w /workspace/code/3DGS/SuGaR \
  3dgs:latest /bin/bash

# Inside the container:
cd /workspace/code/3DGS/Humanoid-intern-challenge/VideoGeoSplat
source ../../.venv/bin/activate

# One-command pipeline
uv run python run_pipeline.py \
  --video examples/input_video.mp4 \
  --output outputs/example_scene \
  --fps 2 \
  --iterations 7000 \
  --mesh
```

## Installation (without Docker)

```bash
cd VideoGeoSplat
pip install -r requirements.txt
# Or with conda:
conda env create -f environment.yml
conda activate videogeo
```

## Pipeline

```
video → frame extraction → VGGT → 3DGS → geometry export → demo → evaluation
```

### Individual stages

Each stage can be run independently:

```bash
# Extract frames
python scripts/extract_frames.py --video input.mp4 --output outputs/scene

# Run VGGT (also creates 3DGS-compatible COLMAP data)
python scripts/run_vggt.py --frames outputs/scene/frames --output outputs/scene

# Run 3DGS training
python scripts/run_3dgs.py --gs_data outputs/scene/gs_data --output outputs/scene --iterations 7000

# Export geometry (point clouds + optional mesh)
python scripts/export_geometry.py \
  --threedgs_output outputs/scene/3dgs \
  --vggt_sparse outputs/scene/gs_data/sparse/0 \
  --output outputs/scene \
  --mesh

# Render demo
python scripts/render_demo.py \
  --threedgs_output outputs/scene/3dgs \
  --gs_data outputs/scene/gs_data \
  --frames outputs/scene/frames \
  --cameras_json outputs/scene/vggt/cameras.json \
  --output outputs/scene

# Evaluate
python scripts/evaluate_reconstruction.py --output outputs/scene
```

## Stage control

```bash
# Resume from a failed run
python run_pipeline.py ... --resume

# Skip VGGT (use existing camera poses)
python run_pipeline.py ... --skip_vggt

# Skip 3DGS training (use existing checkpoints)
python run_pipeline.py ... --skip_3dgs

# Fast demo mode (3000 iterations)
python run_pipeline.py ... --iterations 3000

# With mesh and semantic labels
python run_pipeline.py ... --mesh --semantic
```

## Expected Input

- **Format:** MP4, MOV, AVI (anything ffmpeg/OpenCV can read)
- **Content:** Short video of a small indoor scene (room, desk, corner, etc.)
- **Duration:** 10–60 seconds
- **Motion:** Smooth camera movement (no rapid shaking or abrupt cuts)
- **Coverage:** Enough viewpoints for reasonable 3D coverage

## Output Structure

```
outputs/example_scene/
  frames/               # Extracted & filtered frames
    frames_meta.json
  gs_data/              # 3DGS-compatible COLMAP data (from VGGT)
    images/
    sparse/0/
  vggt/                 # VGGT camera exports
    cameras.json
    intrinsics.npy
    extrinsics.npy
  3dgs/                 # 3DGS training output
    point_cloud/
      iteration_7000/
        point_cloud.ply
    cameras.json
  geometry/             # Exported point clouds & mesh
    gaussians.ply
    pointcloud_from_vggt.ply
    pointcloud_from_3dgs.ply
    mesh.obj            # optional
  demo/                 # Rendered visualizations
    input_grid.png
    camera_trajectory.png
    novel_view.mp4
  metrics.json          # Reconstruction quality metrics
  reconstruction_report.md
```

## Design Choices

This system uses **VGGT** as a feed-forward geometry front-end to estimate camera poses, depths, and point maps from video frames. These predictions are then used to drive a **3D Gaussian Splatting** reconstruction, which provides a photorealistic and interactive scene representation. Since 3DGS is primarily optimized for rendering rather than strict surface geometry, the system also exports explicit point clouds and optional meshes for geometric inspection.

Key tradeoffs:
- **Feed-forward vs bundle adjustment:** VGGT feed-forward mode is fast (~seconds) but less accurate than VGGT with BA or traditional COLMAP. Use `--use_ba` for higher quality at the cost of speed.
- **3DGS iterations:** 7000 iterations gives good quality; 3000 is acceptable for quick demos.
- **Mesh quality:** Poisson surface reconstruction works best on dense, uniformly sampled point clouds. Indoor scenes with thin structures may produce imperfect meshes.

## Known Limitations

- **Camera path:** Degenerate camera motion (pure rotation, no translation) will produce poor results
- **Transparent/reflective surfaces:** Violate the Lambertian assumption of both VGGT and 3DGS
- **Dynamic objects:** Moving people/objects in the video will create artifacts
- **Large outdoor scenes:** VGGT is optimized for indoor-scale scenes
- **Mesh quality:** Poisson reconstruction may produce blobby meshes for sparse point clouds

## License

This project uses:
- [VGGT](https://github.com/facebookresearch/vggt) — Meta Research license
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — Inria research license
