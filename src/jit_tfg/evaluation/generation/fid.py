"""FID (Frechet Inception Distance) calculation.

This module provides FID calculation using pre-computed reference statistics,
following the methodology from the JiT paper.

FID measures the distance between two Gaussian distributions fitted to
Inception-v3 features of real and generated images:

    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r Σ_g))

References:
    - "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
      (Heusel et al., NeurIPS 2017)
"""

from pathlib import Path
from typing import Union

import numpy as np
from scipy import linalg

from jit_tfg.evaluation.generation.inception import InceptionFeatureExtractor


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calculate Frechet Distance between two multivariate Gaussians.

    Args:
        mu1: Mean of first distribution (D,).
        sigma1: Covariance of first distribution (D, D).
        mu2: Mean of second distribution (D,).
        sigma2: Covariance of second distribution (D, D).
        eps: Small value for numerical stability.

    Returns:
        Frechet distance between the two distributions.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return float(fid)


def calculate_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate mean and covariance of feature vectors.

    Args:
        features: Feature array of shape (N, D).

    Returns:
        Tuple of (mean, covariance) where mean is (D,) and covariance is (D, D).
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def load_reference_statistics(
    stats_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-computed reference statistics from NPZ file.

    Args:
        stats_path: Path to the NPZ file containing 'mu' and 'sigma'.

    Returns:
        Tuple of (mu, sigma) loaded from the file.
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")

    data = np.load(stats_path)

    if "mu" not in data or "sigma" not in data:
        raise KeyError(f"Statistics file must contain 'mu' and 'sigma' keys. Found: {list(data.keys())}")

    return data["mu"], data["sigma"]


def calculate_fid(
    generated_features: np.ndarray,
    reference_stats_path: str | Path,
) -> float:
    """Calculate FID between generated features and reference statistics.

    Args:
        generated_features: Feature array of shape (N, 2048) from generated images.
        reference_stats_path: Path to NPZ file with pre-computed reference statistics.

    Returns:
        FID score (lower is better).
    """
    mu_gen, sigma_gen = calculate_statistics(generated_features)

    mu_ref, sigma_ref = load_reference_statistics(reference_stats_path)

    fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)

    return fid


def calculate_fid_from_folder(
    generated_folder: str | Path,
    reference_stats_path: str | Path,
    device: str = "cuda",
    batch_size: int = 64,
    show_progress: bool = True,
) -> float:
    """Calculate FID for images in a folder against reference statistics.

    Args:
        generated_folder: Path to folder containing generated images.
        reference_stats_path: Path to NPZ file with pre-computed reference statistics.
        device: Device for Inception model ('cuda', 'cpu', 'mps').
        batch_size: Batch size for feature extraction.
        show_progress: Whether to show progress bar.

    Returns:
        FID score (lower is better).
    """
    extractor = InceptionFeatureExtractor(device=device, batch_size=batch_size)

    result = extractor.extract_from_folder(
        generated_folder,
        show_progress=show_progress,
    )

    fid = calculate_fid(
        generated_features=result["features"],
        reference_stats_path=reference_stats_path,
    )

    return fid
