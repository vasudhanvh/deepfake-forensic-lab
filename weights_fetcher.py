"""
Robust Model Weights Fetcher & Git LFS Verification
Ensures that the 132.7 MB detector_weights.pth file is present, genuine,
and not an un-hydrated Git LFS pointer text file.
"""

import os
import sys
from pathlib import Path
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weights_fetcher")

# Primary and Fallback URLs for remote weights
REMOTE_WEIGHTS_URLS = [
    # Hugging Face Model Hub direct download link (resolve endpoint)
    "https://huggingface.co/antigravity-deepfake/deepfake-detector-ema/resolve/main/detector_weights.pth",
    # GitHub Release asset direct download fallback
    "https://github.com/deepfake-forensics/multi-branch-detector/releases/download/v1.0.0/detector_weights.pth",
]

EXPECTED_MIN_BYTES = 100 * 1024 * 1024  # > 100 MB required (actual is ~132.7 MB)


def is_lfs_pointer(file_path: Path) -> bool:
    """Checks whether the file is a Git LFS pointer text file (~130 bytes)."""
    if not file_path.is_file():
        return False
    size = file_path.stat().st_size
    if size > 1024:
        return False
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.read(100)
            return "version https://git-lfs.github.com/spec/v1" in header
    except Exception:
        return False


def verify_weights_file(weights_path: Path) -> bool:
    """Returns True if weights file exists and is a valid binary model file."""
    if not weights_path.exists():
        return False
    size = weights_path.stat().st_size
    if is_lfs_pointer(weights_path):
        logger.warning(f"File {weights_path} is an unhydrated Git LFS pointer ({size} bytes).")
        return False
    if size < EXPECTED_MIN_BYTES:
        logger.warning(f"File {weights_path} size ({size / 1e6:.1f} MB) is below minimum expected {EXPECTED_MIN_BYTES / 1e6:.1f} MB.")
        return False
    return True


def download_weights(target_path: Path, progress_callback=None) -> bool:
    """Downloads detector_weights.pth from fallback remote repositories."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(".pth.tmp")

    for url in REMOTE_WEIGHTS_URLS:
        logger.info(f"Attempting download from: {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "DeepfakeForensics/1.0 (StreamlitCloud/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_target, "wb") as out_file:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size > 0:
                        progress_callback(min(1.0, downloaded / total_size))

            if temp_target.stat().st_size >= EXPECTED_MIN_BYTES:
                if target_path.exists():
                    target_path.unlink()
                temp_target.rename(target_path)
                logger.info(f"Successfully downloaded weights ({target_path.stat().st_size / 1e6:.2f} MB)")
                return True
            else:
                logger.warning(f"Downloaded file from {url} was too small ({temp_target.stat().st_size} bytes).")
                if temp_target.exists():
                    temp_target.unlink()
        except Exception as e:
            logger.warning(f"Download from {url} failed: {e}")
            if temp_target.exists():
                temp_target.unlink()

    return False


def ensure_weights(bundle_dir: str or Path = ".", progress_callback=None) -> Path:
    """
    Validates presence of detector_weights.pth.
    If missing or an unhydrated LFS pointer, automatically fetches from cloud storage.
    """
    bundle_path = Path(bundle_dir)
    weights_path = bundle_path / "detector_weights.pth"

    if verify_weights_file(weights_path):
        return weights_path

    logger.info("Weights not found or invalid. Triggering automated cloud fetcher...")
    success = download_weights(weights_path, progress_callback=progress_callback)
    if not success and not verify_weights_file(weights_path):
        raise FileNotFoundError(
            f"Could not locate or download genuine weights to {weights_path}. "
            "If using Git LFS, execute 'git lfs pull' or provide detector_weights.pth manually."
        )
    return weights_path


if __name__ == "__main__":
    path = ensure_weights(".")
    print(f"Weights verification complete: {path} ({path.stat().st_size / 1e6:.2f} MB)")
