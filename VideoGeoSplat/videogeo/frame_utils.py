"""Frame extraction and quality control utilities."""

import os
import cv2
import json
import subprocess
import numpy as np
from videogeo.io_utils import ensure_dir, save_json


def extract_frames_ffmpeg(video_path: str, output_dir: str, fps: float = 2):
    """Extract frames from video using ffmpeg at given fps."""
    ensure_dir(output_dir)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        f"{output_dir}/%06d.png"
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    # Get list of extracted frames
    import glob
    frames = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    return frames


def extract_frames_opencv(video_path: str, output_dir: str, fps: float = 2):
    """Extract frames using OpenCV for more control."""
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(video_fps / fps))
    frame_indices = list(range(0, total_frames, interval))

    saved = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"{len(saved):06d}.png")
        cv2.imwrite(out_path, frame)
        saved.append(out_path)

    cap.release()
    return saved


def laplacian_variance(image: np.ndarray) -> float:
    """Compute Laplacian variance (blur detection). Higher = sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def filter_blurry_frames(frame_paths: list, threshold: float = 100.0) -> list:
    """Remove blurry frames based on Laplacian variance."""
    sharp = []
    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        var = laplacian_variance(img)
        if var >= threshold:
            sharp.append(fp)
        else:
            os.remove(fp)  # Delete blurry frame
    return sharp


def resize_frames(frame_paths: list, long_edge: int = 1024) -> tuple:
    """Resize frames so the long edge equals `long_edge`. Returns (paths, new_size)."""
    new_size = None
    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) == long_edge:
            new_size = (w, h)
            continue
        scale = long_edge / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fp, img_resized)
        new_size = (new_w, new_h)
    return frame_paths, new_size


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 2,
    max_frames: int = 120,
    resize_long_edge: int = 1024,
    blur_threshold: float = 100.0,
) -> dict:
    """
    Full frame extraction pipeline with quality control.

    Returns metadata dict.
    """
    frames_dir = os.path.join(output_dir, "frames")
    ensure_dir(frames_dir)

    # Clear existing frames
    import glob
    for old in glob.glob(os.path.join(frames_dir, "*.png")):
        os.remove(old)

    # Extract frames
    try:
        frame_paths = extract_frames_ffmpeg(video_path, frames_dir, fps)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg failed or not found, falling back to OpenCV")
        frame_paths = extract_frames_opencv(video_path, frames_dir, fps)

    if not frame_paths:
        raise ValueError(f"No frames could be extracted from {video_path}")

    print(f"Extracted {len(frame_paths)} frames")

    # Filter blurry
    frame_paths = filter_blurry_frames(frame_paths, blur_threshold)
    print(f"After blur filtering: {len(frame_paths)} frames")

    # Resize
    frame_paths, new_size = resize_frames(frame_paths, resize_long_edge)
    if new_size is None:
        # Read size from first frame
        img = cv2.imread(frame_paths[0])
        new_size = (img.shape[1], img.shape[0])

    print(f"Frame size: {new_size}")

    # Limit to max_frames by uniform sampling
    if len(frame_paths) > max_frames:
        indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
        kept_paths = []
        for i, fp in enumerate(frame_paths):
            if i in indices:
                kept_paths.append(fp)
            else:
                os.remove(fp)
        frame_paths = kept_paths
        print(f"Uniformly sampled down to {len(frame_paths)} frames")

    # Save metadata
    meta = {
        "video_path": os.path.abspath(video_path),
        "fps": fps,
        "num_frames": len(frame_paths),
        "image_size": list(new_size),
    }
    save_json(meta, os.path.join(output_dir, "frames_meta.json"))

    return meta
