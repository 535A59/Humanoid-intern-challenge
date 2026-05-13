"""Configuration loading and management for VideoGeoSplat."""

import os
import yaml
from argparse import Namespace


def load_config(config_path=None) -> dict:
    """Load YAML config file, with optional override path."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_args_to_config(cfg: dict, args: Namespace) -> dict:
    """Override config values from CLI args (non-None values only)."""
    # Pipeline overrides
    if hasattr(args, "fps") and args.fps is not None:
        cfg.setdefault("pipeline", {})["fps"] = args.fps
    if hasattr(args, "max_frames") and args.max_frames is not None:
        cfg.setdefault("pipeline", {})["max_frames"] = args.max_frames
    if hasattr(args, "resize_long_edge") and args.resize_long_edge is not None:
        cfg.setdefault("pipeline", {})["resize_long_edge"] = args.resize_long_edge
    # Gaussian overrides
    if hasattr(args, "iterations") and args.iterations is not None:
        cfg.setdefault("gaussian", {})["iterations"] = args.iterations
    # Semantic override
    if hasattr(args, "semantic") and args.semantic:
        cfg.setdefault("semantic", {})["enabled"] = True
    # Mesh override
    if hasattr(args, "mesh") and args.mesh:
        cfg.setdefault("geometry", {})["mesh_method"] = "poisson"
    if hasattr(args, "no_mesh") and args.no_mesh:
        cfg.setdefault("geometry", {})["mesh_method"] = "none"
    # Device override
    if hasattr(args, "device") and args.device is not None:
        cfg.setdefault("device", args.device)
    return cfg


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VGGT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "vggt"))
THREEDGS_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "gaussian-splatting"))
