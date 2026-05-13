"""I/O utilities for VideoGeoSplat."""

import os
import json
import shutil
import numpy as np


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: dict, path: str, indent: int = 2):
    """Save dict as JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str) -> dict:
    """Load JSON file as dict."""
    with open(path, "r") as f:
        return json.load(f)


def save_npy(data: np.ndarray, path: str):
    """Save numpy array to .npy file."""
    ensure_dir(os.path.dirname(path))
    np.save(path, data)


def load_npy(path: str) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(path)


def copy_files(src_dir: str, dst_dir: str, pattern: str = "*"):
    """Copy files matching pattern from src_dir to dst_dir."""
    import glob
    ensure_dir(dst_dir)
    for src in glob.glob(os.path.join(src_dir, pattern)):
        dst = os.path.join(dst_dir, os.path.basename(src))
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)


def check_file_exists(path: str, description: str = "") -> bool:
    """Return True if file exists, print warning if not."""
    if not os.path.exists(path):
        msg = f"WARNING: {description} not found at {path}"
        print(msg)
        return False
    return True


def get_size_mb(path: str) -> float:
    """Get file/directory size in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)
