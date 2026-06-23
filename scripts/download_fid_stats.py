#!/usr/bin/env python
"""Download the FID reference statistics used for evaluation.

The paper evaluates with three Inception-V3 FID reference statistics
(``mu`` / ``sigma`` pairs stored as ``.npz``):

    imagenet/in256_stats.npz          ImageNet 256 -> Parent-FID, CFG-only FID
    finegrained/parent_fid_stats.npz  bird parent  -> Parent-FID (P-FID)
    finegrained/child_fid_stats.npz   bird child   -> Child-FID  (C-FID, headline)

They are downloaded into ``src/jit_tfg/evaluation/generation/fid_stats/`` — the
exact paths the evaluation code resolves. The ``.npz`` files there are
gitignored and never committed to the repository.

HOSTING IS DEFERRED: the HuggingFace dataset that will host these files is not
published yet. ``HF_FID_STATS_REPO`` is a placeholder; until it is set to the
real dataset ID, this script prints an explanatory message and exits without
downloading.

Examples
--------
    uv run python scripts/download_fid_stats.py
    uv run python scripts/download_fid_stats.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root = parent of this scripts/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FID_STATS_DIR = PROJECT_ROOT / "src" / "jit_tfg" / "evaluation" / "generation" / "fid_stats"

# Sentinel marking an unset placeholder constant.
TODO_PREFIX = "TODO"

# FIXME: set after the FID-stats dataset is uploaded to the HuggingFace Hub.
HF_FID_STATS_REPO = "TODO-ManLuML/<dataset>"

# Files to fetch: (path-within-HF-repo, path-relative-to-FID_STATS_DIR).
# Both use the same relative layout, matching what the evaluation code expects.
FID_STATS_FILES = (
    ("imagenet/in256_stats.npz", "imagenet/in256_stats.npz"),
    ("finegrained/parent_fid_stats.npz", "finegrained/parent_fid_stats.npz"),
    ("finegrained/child_fid_stats.npz", "finegrained/child_fid_stats.npz"),
)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download FID reference statistics into the evaluation package.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    print(f"Target directory: {FID_STATS_DIR}")

    if _is_placeholder(HF_FID_STATS_REPO):
        print(
            "\n[deferred] The FID-stats HuggingFace dataset is not published yet.\n"
            f"           HF_FID_STATS_REPO is still a placeholder: {HF_FID_STATS_REPO!r}\n"
            "\n"
            "To enable this download:\n"
            "  1. Upload the .npz files to a HuggingFace dataset, preserving layout:\n"
            "       imagenet/in256_stats.npz\n"
            "       finegrained/parent_fid_stats.npz\n"
            "       finegrained/child_fid_stats.npz\n"
            "  2. Set HF_FID_STATS_REPO in this script to that dataset ID.\n"
            "\n"
            "See CHECKPOINTS.md for details.",
        )
        return 1

    if args.dry_run:
        for remote, local in FID_STATS_FILES:
            print(f"[dry-run] would download {HF_FID_STATS_REPO}:{remote} -> {FID_STATS_DIR / local}")
        return 0

    hub = _require_hub()
    FID_STATS_DIR.mkdir(parents=True, exist_ok=True)

    for remote, local in FID_STATS_FILES:
        target = FID_STATS_DIR / local
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {HF_FID_STATS_REPO}:{remote}")
        path = hub.hf_hub_download(
            repo_id=HF_FID_STATS_REPO,
            repo_type="dataset",
            filename=remote,
            local_dir=str(FID_STATS_DIR),
        )
        print(f"  done -> {path}")

    print("\nAll FID statistics downloaded. Files are gitignored (never committed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
