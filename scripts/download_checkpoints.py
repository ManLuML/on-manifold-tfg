#!/usr/bin/env python
"""Download the pretrained diffusion-model checkpoints used by the paper.

*Not All Prediction Targets Keep Training-Free Diffusion Guidance on the
Manifold* (ECCV 2026) evaluates four pretrained Diffusion Transformers spanning
the three prediction targets (x / v / epsilon):

    JiT-H/16    x   pixel    (Back to Basics, Li & He 2025)
    DiT-XL/2    eps latent   (Peebles & Xie 2023)
    SiT-XL/2    v   latent   (Ma et al. 2024)
    PixelFlow   v   pixel    (Chen et al. 2025)

This script downloads them into ``checkpoints/`` (gitignored). Weights are
*never* committed to the repository.

Examples
--------
    # Download everything that has a known source
    uv run python scripts/download_checkpoints.py --all

    # Download a subset
    uv run python scripts/download_checkpoints.py dit sit

    # Just print what would happen
    uv run python scripts/download_checkpoints.py --all --dry-run

Notes
-----
The JiT-H/16 weights are published by the original authors (Li & He, "Back to
Basics: Let Denoising Generative Models Denoise", arXiv:2511.13720) at
https://github.com/LTH14/JiT and distributed via Dropbox, so they cannot be
auto-downloaded from a single stable URL. ``download_jit`` prints the manual
step; place the file at ``checkpoints/jit/jit-h-16.pth`` (see ``CHECKPOINTS.md``).
DiT / SiT / PixelFlow download automatically from their official Hub sources.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Project root = parent of this scripts/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Sentinel marking an unset placeholder constant.
TODO_PREFIX = "TODO"

# --- JiT-H/16 (x-prediction, pixel space) ------------------------------------
# Published by the original authors at https://github.com/LTH14/JiT (weights via
# Dropbox). Manual download — see download_jit(). Keep the filename
# 'jit-h-16.pth'; the loader infers the variant ('JiT-H/16') from it.
JIT_REPO_URL = "https://github.com/LTH14/JiT"
JIT_WEIGHTS_DROPBOX = (
    "https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/"
    "AK218ZNvXKSv74igVvht4PQ?rlkey=14gjrblmljewpl6ygxzlr3njm&dl=0"
)
JIT_H16_TARGET = CHECKPOINTS_DIR / "jit" / "jit-h-16.pth"

# --- DiT-XL/2 (epsilon-prediction, latent space) -----------------------------
# Provenance: official Meta DiT release (Peebles & Xie). The HuggingFace repo
# facebook/DiT-XL-2-256 hosts only the diffusers-format weights (no .pt file),
# so the original .pt checkpoint is fetched from dl.fbaipublicfiles.com.
DIT_DIRECT_URL = "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt"
# SHA256 of the official checkpoint — same pinned constant as
# src/jit_tfg/models/dit/denoiser.py.
DIT_SHA256 = "9ec1876e4c03471bca126663a30e2d1b20610b6d2f87850a39a36f25cc685521"
DIT_TARGET = CHECKPOINTS_DIR / "dit" / "DiT-XL-2-256x256.pt"

# --- SiT-XL/2 (v-prediction, latent space) -----------------------------------
SIT_HF_REPO = "nyu-visionx/SiT-collections"
SIT_TARGET_DIR = CHECKPOINTS_DIR / "sit"

# --- PixelFlow (v-prediction, pixel space) -----------------------------------
PIXELFLOW_HF_REPO = "ShoufaChen/PixelFlow-Class2Image"
PIXELFLOW_TARGET_DIR = CHECKPOINTS_DIR / "pixelflow"

MODELS = ("jit", "dit", "sit", "pixelflow")


def _is_placeholder(value: str) -> bool:
    """Return True if a constant is still an unset TODO placeholder."""
    return value.startswith(TODO_PREFIX)


def _require_hub():
    """Import huggingface_hub or exit with an actionable message."""
    try:
        import huggingface_hub
    except ImportError:
        sys.exit(
            "huggingface_hub is required for this download.\n"
            "Install it with:  uv add huggingface_hub   (or: pip install huggingface_hub)"
        )
    return huggingface_hub


def _download_url(url: str, target: Path, *, dry_run: bool) -> None:
    """Download a single file from a direct URL to ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] would download {url} -> {target}")
        return
    print(f"Downloading {url}\n        -> {target}")
    urllib.request.urlretrieve(url, target)  # noqa: S310 (trusted official URL)
    print(f"  done ({target.stat().st_size / 1e9:.2f} GB)")


def _verify_sha256(target: Path, expected: str) -> None:
    """Verify ``target`` against a pinned SHA256, exiting on mismatch."""
    print(f"  verifying SHA256 of {target.name} ...")
    digest = hashlib.sha256()
    with target.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        sys.exit(
            f"SHA256 mismatch for {target}:\n"
            f"  expected {expected}\n"
            f"  got      {actual}\n"
            "The file may be corrupted — delete it and re-run."
        )
    print("  SHA256 OK")


def download_jit(*, dry_run: bool) -> None:
    """JiT-H/16 (x-prediction) — manual download from the original authors."""
    already = JIT_H16_TARGET.exists()
    print(
        "JiT-H/16 weights are distributed by the original authors and cannot be\n"
        "auto-downloaded. To obtain them:\n"
        f"  1. Open the official repo: {JIT_REPO_URL}\n"
        f"  2. Download the pretrained weights (Dropbox): {JIT_WEIGHTS_DROPBOX}\n"
        f"  3. Place the JiT-H/16 file at: {JIT_H16_TARGET}\n"
        "     (keep the filename 'jit-h-16.pth' — the loader parses the variant from it).\n"
        f"  Status: {'FOUND' if already else 'NOT FOUND yet'} at {JIT_H16_TARGET}"
    )
    if not dry_run and not already:
        JIT_H16_TARGET.parent.mkdir(parents=True, exist_ok=True)


def download_dit(*, dry_run: bool) -> None:
    """Download DiT-XL/2 (epsilon-prediction) from the official Meta URL."""
    if dry_run:
        print(f"[dry-run] would download {DIT_DIRECT_URL} -> {DIT_TARGET} (then verify SHA256)")
        return
    if DIT_TARGET.exists():
        print(f"Found existing {DIT_TARGET}; skipping download.")
    else:
        _download_url(DIT_DIRECT_URL, DIT_TARGET, dry_run=False)
    _verify_sha256(DIT_TARGET, DIT_SHA256)


def download_sit(*, dry_run: bool) -> None:
    """Download the SiT-XL/2 collection (v-prediction)."""
    if dry_run:
        print(f"[dry-run] would snapshot {SIT_HF_REPO} -> {SIT_TARGET_DIR}")
        return
    hub = _require_hub()
    SIT_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SIT_HF_REPO} (HuggingFace snapshot)")
    path = hub.snapshot_download(repo_id=SIT_HF_REPO, local_dir=str(SIT_TARGET_DIR))
    print(f"  done -> {path}")


def download_pixelflow(*, dry_run: bool) -> None:
    """Download the PixelFlow class-to-image weights (v-prediction)."""
    if dry_run:
        print(f"[dry-run] would snapshot {PIXELFLOW_HF_REPO} -> {PIXELFLOW_TARGET_DIR}")
        return
    hub = _require_hub()
    PIXELFLOW_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {PIXELFLOW_HF_REPO} (HuggingFace snapshot)")
    path = hub.snapshot_download(
        repo_id=PIXELFLOW_HF_REPO,
        local_dir=str(PIXELFLOW_TARGET_DIR),
    )
    print(f"  done -> {path}")


DOWNLOADERS = {
    "jit": download_jit,
    "dit": download_dit,
    "sit": download_sit,
    "pixelflow": download_pixelflow,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download pretrained diffusion checkpoints into checkpoints/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help=f"Models to download (any of: {', '.join(MODELS)}).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all four models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without downloading.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)

    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        print(f"Unknown model(s): {', '.join(unknown)}. Choose from: {', '.join(MODELS)}")
        return 2

    selected = MODELS if args.all else tuple(dict.fromkeys(args.models))
    if not selected:
        print("Nothing to do. Pass --all or one or more of:", ", ".join(MODELS))
        print("Run with -h for usage.")
        return 1

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {CHECKPOINTS_DIR}")
    print(f"Selected: {', '.join(selected)}\n")

    for name in selected:
        DOWNLOADERS[name](dry_run=args.dry_run)
        print()

    print("All requested downloads handled. Weights are gitignored (never committed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
