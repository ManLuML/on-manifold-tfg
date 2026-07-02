"""Generative model evaluation metrics.

This module provides utilities for calculating Inception Score (IS)
and unified evaluation functions.

Uses torch-fidelity for IS calculation to ensure consistency with
published benchmarks.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch_fidelity
from tqdm import tqdm

from jit_tfg.evaluation.generation.fid import calculate_fid
from jit_tfg.evaluation.generation.inception import InceptionFeatureExtractor


def calculate_inception_score(
    folder_path: str | Path,
) -> tuple[float, float]:
    """Calculate Inception Score using torch-fidelity.

    IS measures how well a generative model produces images that:
    1. Are clearly classifiable (low entropy of p(y|x))
    2. Cover many classes (high entropy of p(y))

    Uses torch-fidelity for exact compatibility with published benchmarks.

    Args:
        folder_path: Path to folder containing generated images.

    Returns:
        Tuple of (mean IS, std IS).
    """
    metrics = torch_fidelity.calculate_metrics(
        input1=str(folder_path),
        cuda=True,
        isc=True,
        fid=False,
        kid=False,
        prc=False,
        verbose=False,
    )

    return (
        float(metrics["inception_score_mean"]),
        float(metrics["inception_score_std"]),
    )


def evaluate_generation(
    generated_folder: str | Path,
    reference_stats_path: str | Path,
    device: str = "cuda",
    batch_size: int = 64,
    show_progress: bool = True,
) -> dict[str, float]:
    """Evaluate generated images with FID and IS metrics.

    Uses torch-fidelity's Inception-v3 for feature extraction, then computes:
    - FID from extracted features + pre-computed reference stats
    - IS from extracted logits (avoids re-processing images)

    Args:
        generated_folder: Path to folder containing generated images.
        reference_stats_path: Path to NPZ file with pre-computed reference statistics.
        device: Device for Inception model ('cuda', 'cpu', 'mps').
        batch_size: Batch size for feature extraction.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary with evaluation metrics:
            - 'fid': Frechet Inception Distance
            - 'is_mean': Inception Score (mean)
            - 'is_std': Inception Score (std)
            - 'num_images': Number of images evaluated
    """
    generated_folder = Path(generated_folder)

    extractor = InceptionFeatureExtractor(device=device, batch_size=batch_size)

    result = extractor.extract_from_folder(
        generated_folder,
        show_progress=show_progress,
    )

    fid = calculate_fid(
        generated_features=result["features"],
        reference_stats_path=reference_stats_path,
    )

    is_mean, is_std = calculate_inception_score_from_logits(result["logits"])

    return {
        "fid": fid,
        "is_mean": is_mean,
        "is_std": is_std,
        "num_images": len(result["features"]),
    }


def _get_default_stats_dir() -> Path:
    """Get the default FID stats directory (bundled with the package)."""
    return Path(__file__).parent / "fid_stats"


def get_reference_stats_path(img_size: int, stats_dir: str | Path | None = None) -> Path:
    """Get the path to ImageNet reference statistics file based on image size.

    Args:
        img_size: Image resolution (256 or 512).
        stats_dir: Directory containing stats files. Defaults to package's fid_stats/.

    Returns:
        Path to the appropriate statistics file.

    Raises:
        ValueError: If img_size is not 256 or 512.
        FileNotFoundError: If statistics file doesn't exist.
    """
    stats_dir = _get_default_stats_dir() if stats_dir is None else Path(stats_dir)

    if img_size == 256:
        stats_file = stats_dir / "imagenet" / "in256_stats.npz"
    elif img_size == 512:
        stats_file = stats_dir / "imagenet" / "in512_stats.npz"
    else:
        raise ValueError(f"Unsupported image size: {img_size}. Must be 256 or 512.")

    if not stats_file.exists():
        raise FileNotFoundError(f"Reference statistics not found: {stats_file}")

    return stats_file


def get_finegrained_stats_path(
    stats_type: str,
    stats_dir: str | Path | None = None,
) -> Path:
    """Get the path to fine-grained reference statistics file.

    Args:
        stats_type: Type of stats: "child" or "parent" for bird classification.
        stats_dir: Directory containing stats files. Defaults to package's fid_stats/.

    Returns:
        Path to the appropriate statistics file.

    Raises:
        ValueError: If stats_type is not "child" or "parent".
        FileNotFoundError: If statistics file doesn't exist.
    """
    stats_dir = _get_default_stats_dir() if stats_dir is None else Path(stats_dir)

    if stats_type == "child":
        stats_file = stats_dir / "finegrained" / "child_fid_stats.npz"
    elif stats_type == "parent":
        stats_file = stats_dir / "finegrained" / "parent_fid_stats.npz"
    else:
        raise ValueError(f"Unsupported stats_type: {stats_type}. Must be 'child' or 'parent'.")

    if not stats_file.exists():
        raise FileNotFoundError(f"Reference statistics not found: {stats_file}")

    return stats_file


def get_class_stats_path(
    class_idx: int,
    stats_dir: str | Path | None = None,
) -> Path:
    """Get the path to per-class FID statistics file.

    .. deprecated::
        Per-class FID stats have been removed. Use get_reference_stats_path()
        for global FID or get_finegrained_stats_path() for fine-grained evaluation.

    Args:
        class_idx: ImageNet class index (0-999).
        stats_dir: Directory containing per-class stats.

    Returns:
        Path to the class-specific statistics file.

    Raises:
        FileNotFoundError: If statistics file doesn't exist.
    """
    import warnings

    warnings.warn(
        "Per-class FID stats have been removed. Use get_reference_stats_path() "
        "for global FID or get_finegrained_stats_path() for fine-grained evaluation.",
        DeprecationWarning,
        stacklevel=2,
    )

    if stats_dir is None:
        raise FileNotFoundError(
            "Per-class FID stats have been removed from the package. "
            "Provide stats_dir if you have pre-computed class stats."
        )

    stats_dir = Path(stats_dir)
    stats_file = stats_dir / f"class_{class_idx:04d}.npz"

    if not stats_file.exists():
        raise FileNotFoundError(f"Class statistics not found: {stats_file}")

    return stats_file


def calculate_class_fid(
    generated_folder: str | Path,
    target_classes: list[int],
    images_per_class: int,
    device: str = "cuda",
    batch_size: int = 64,
    stats_dir: str | Path | None = None,
    show_progress: bool = True,
    precomputed_features: np.ndarray | None = None,
    precomputed_paths: list[Path] | None = None,
) -> dict[str, float]:
    """Calculate per-class FID for guided generation evaluation.

    Computes FID for each target class separately using class-specific
    reference statistics, then averages. This is the proper way to evaluate
    class-conditional generation quality (as done in TFG paper).

    Args:
        generated_folder: Path to folder containing generated images.
            Images should be named with class info (e.g., img_class0111_0001.png).
        target_classes: List of target class indices (e.g., [111, 222, 333, 444]).
        images_per_class: Number of images generated per class.
        device: Device for Inception model.
        batch_size: Batch size for feature extraction.
        stats_dir: Directory containing per-class stats files.
        show_progress: Whether to show progress bar.
        precomputed_features: Optional pre-extracted Inception features (N, 2048).
            If provided, skips feature extraction.
        precomputed_paths: Optional list of image paths corresponding to features.
            Required if precomputed_features is provided.

    Returns:
        Dictionary with:
            - 'fid_mean': Average FID across all classes
            - 'fid_per_class': Dict mapping class index to FID
            - 'num_images': Total number of images
            - 'features': Extracted features (for reuse)
            - 'image_paths': Image paths (for reuse)
    """
    import re

    generated_folder = Path(generated_folder)

    # Extract features (or use precomputed)
    if precomputed_features is not None and precomputed_paths is not None:
        all_features = precomputed_features
        image_paths = precomputed_paths
    else:
        extractor = InceptionFeatureExtractor(device=device, batch_size=batch_size)
        result = extractor.extract_from_folder(generated_folder, show_progress=show_progress)
        all_features = result["features"]

        from jit_tfg.evaluation.generation.inception import ImageFolderDataset

        dataset = ImageFolderDataset(generated_folder)
        image_paths = dataset.image_paths

    # Parse class from filename (e.g., img_class0111_0001.png -> 111)
    class_pattern = re.compile(r"class(\d{4})")

    # Group features by class
    class_features: dict[int, list[np.ndarray]] = {cls: [] for cls in target_classes}

    for i, path in enumerate(image_paths):
        match = class_pattern.search(path.name)
        if match:
            cls = int(match.group(1))
            if cls in class_features:
                class_features[cls].append(all_features[i])

    # Calculate FID for each class
    fid_per_class = {}

    class_iter = target_classes
    if show_progress:
        class_iter = tqdm(target_classes, desc="Computing per-class FID")

    for cls in class_iter:
        features = class_features.get(cls, [])
        if len(features) == 0:
            print(f"Warning: No images found for class {cls}")
            continue

        features_array = np.stack(features, axis=0)

        try:
            stats_path = get_class_stats_path(cls, stats_dir)
            fid = calculate_fid(features_array, stats_path)
            fid_per_class[cls] = fid
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    if not fid_per_class:
        raise ValueError("Could not compute FID for any class. Check stats files.")

    fid_mean = np.mean(list(fid_per_class.values()))

    return {
        "fid_mean": float(fid_mean),
        "fid_per_class": fid_per_class,
        "num_images": len(all_features),
        "features": all_features,
        "image_paths": image_paths,
    }


def calculate_inception_score_from_logits(
    logits: np.ndarray,
    num_splits: int = 10,
) -> tuple[float, float]:
    """Calculate Inception Score from pre-computed logits.

    Uses the standard IS formula without re-extracting features.

    Args:
        logits: Pre-computed Inception logits of shape (N, 1000).
        num_splits: Number of splits for computing mean/std.

    Returns:
        Tuple of (mean IS, std IS).
    """
    from scipy.special import softmax

    # Apply softmax to get probabilities
    probs = softmax(logits, axis=1)

    # Split for computing mean/std
    scores = []
    split_size = len(probs) // num_splits

    for i in range(num_splits):
        start = i * split_size
        end = start + split_size if i < num_splits - 1 else len(probs)
        part = probs[start:end]

        # p(y|x) for each image, p(y) marginal
        py = np.mean(part, axis=0, keepdims=True)

        # KL divergence: sum over classes of p(y|x) * log(p(y|x) / p(y))
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.sum(kl, axis=1)

        scores.append(np.exp(np.mean(kl)))

    return float(np.mean(scores)), float(np.std(scores))


def calculate_fid_from_features(
    features: np.ndarray,
    reference_stats_path: str | Path,
) -> float:
    """Calculate FID from pre-computed features.

    Args:
        features: Pre-computed Inception features of shape (N, 2048).
        reference_stats_path: Path to NPZ file with reference mu and sigma.

    Returns:
        FID score.
    """
    return calculate_fid(features, reference_stats_path)
