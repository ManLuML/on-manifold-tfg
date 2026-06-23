#!/usr/bin/env python3
"""Inference script for JiT + TFG Spiral Test.

This script loads trained models from an output folder and performs:
1. Sampling with DSP (classifier) guidance
2. Visualization with classification landscape background

Results are saved to <output_folder>/results/.

Usage:
    cd on-manifold-tfg

    # Run with default settings
    uv run python spiral_test/inference.py --exp doublespiral_20260113_123456

    # Run with specific guidance scale and steps
    uv run python spiral_test/inference.py --exp doublespiral_20260113_123456 -s 2.0 -n 50

    # Run with multiple scales and steps
    uv run python spiral_test/inference.py --exp doublespiral_20260113_123456 -s 0.0 2.0 5.0 -n 50 100
"""

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    # Sampling
    num_samples: int = 10000
    t_eps: float = 5e-2

    # Visualization
    figsize: tuple[float, float] | None = None  # Auto-computed if None
    dpi: int = 300
    point_size: int = 1
    background_resolution: int = 100

    # Device & Seed
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Validate and adjust device."""
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"


# =============================================================================
# Models (must match train.py definitions)
# =============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SpiralClassifier(nn.Module):
    """MLP classifier for spiral data."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (128, 128, 128),
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim

        if num_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for each class."""
        logits = self.forward(x)

        if self.num_classes == 2:
            prob_1 = torch.sigmoid(logits)
            prob_0 = 1 - prob_1
            probs = torch.cat([prob_0, prob_1], dim=-1)
            return torch.log(probs + 1e-8)
        else:
            return F.log_softmax(logits, dim=-1)


class ResMLPBlock(nn.Module):
    """Residual MLP block with time conditioning."""

    def __init__(self, dim: int, time_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.relu(self.fc1(h))
        h = h + self.time_proj(t_emb)
        h = F.relu(self.fc2(h))
        return x + h


class SpiralDiffusionModel(nn.Module):
    """ResMLP diffusion model for spiral data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 5,
        pred_target: Literal["x", "e", "v"] = "x",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.pred_target = pred_target

        time_dim = hidden_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([ResMLPBlock(hidden_dim, time_dim) for _ in range(num_blocks)])

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.input_proj(z)

        for block in self.blocks:
            h = block(h, t_emb)

        return self.output_proj(h)

    def predict_x0(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_eps: float = 5e-2,
    ) -> torch.Tensor:
        """Predict clean data x0 from noisy input."""
        t = t.view(-1, 1)
        net_out = self(z, t.squeeze(-1))

        if self.pred_target == "x":
            x0_pred = net_out
        elif self.pred_target == "e":
            x0_pred = (z - (1 - t) * net_out) / t.clamp_min(t_eps)
        elif self.pred_target == "v":
            x0_pred = (1 - t) * net_out + z
        else:
            raise ValueError(f"Unknown pred_target: {self.pred_target}")

        return x0_pred

    def predict_v(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_eps: float = 5e-2,
    ) -> torch.Tensor:
        """Predict velocity v = x - e from noisy input."""
        t = t.view(-1, 1)
        net_out = self(z, t.squeeze(-1))

        if self.pred_target == "x":
            v_pred = (net_out - z) / (1 - t).clamp_min(t_eps)
        elif self.pred_target == "e":
            v_pred = (z - net_out) / t.clamp_min(t_eps)
        elif self.pred_target == "v":
            v_pred = net_out
        else:
            raise ValueError(f"Unknown pred_target: {self.pred_target}")

        return v_pred


# =============================================================================
# Path Utilities
# =============================================================================


def get_output_dir() -> Path:
    """Get the output directory path."""
    return Path(__file__).parent / "output"


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent / "data"


def parse_data_name_from_exp(exp_name: str) -> str:
    """Parse data name from experiment folder name.

    Experiment folders are named: <data_name>_<timestamp>
    e.g., doublespiral_20260113_123456 -> doublespiral

    Args:
        exp_name: Experiment folder name.

    Returns:
        Data name.
    """
    # Split by underscore and take everything before the timestamp
    # Timestamp format: YYYYMMDD_HHMMSS (15 chars with underscore)
    parts = exp_name.rsplit("_", 2)
    if len(parts) >= 3:
        # Check if last two parts look like timestamp
        try:
            int(parts[-2])  # YYYYMMDD
            int(parts[-1])  # HHMMSS
            return "_".join(parts[:-2])
        except ValueError:
            pass
    return exp_name


# =============================================================================
# Loading Functions
# =============================================================================


def load_train_config(exp_dir: Path) -> dict:
    """Load training configuration."""
    with open(exp_dir / "train_config.json") as f:
        return json.load(f)


def load_data(data_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load original data from data directory."""
    data_dir = get_data_dir() / data_name

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data = np.load(data_dir / "data.npz")
    points_2d = data["points_2d"]
    labels = data["labels"]

    with open(data_dir / "class_info.json") as f:
        class_info = json.load(f)

    return points_2d, labels, class_info


def load_data_config(data_name: str) -> dict:
    """Load data configuration."""
    data_dir = get_data_dir() / data_name
    with open(data_dir / "config.json") as f:
        return json.load(f)


def load_projection_matrix(D: int, exp_dir: Path) -> np.ndarray:
    """Load projection matrix for dimension D."""
    return np.load(exp_dir / f"D{D}" / "proj_matrix.npy")


def load_classifier(D: int, exp_dir: Path, device: str) -> SpiralClassifier:
    """Load classifier for dimension D."""
    save_dict = torch.load(exp_dir / f"D{D}" / "classifier.pt", map_location=device, weights_only=True)

    classifier = SpiralClassifier(
        input_dim=save_dict["input_dim"],
        num_classes=save_dict["num_classes"],
        hidden_dims=tuple(save_dict["hidden_dims"]),
    )
    classifier.load_state_dict(save_dict["state_dict"])
    classifier = classifier.to(device).eval()

    return classifier


def load_diffusion_model(
    D: int,
    pred_target: str,
    exp_dir: Path,
    device: str,
) -> SpiralDiffusionModel:
    """Load diffusion model for dimension D and prediction target."""
    save_dict = torch.load(
        exp_dir / f"D{D}" / f"diffusion_{pred_target}.pt",
        map_location=device,
        weights_only=True,
    )

    model = SpiralDiffusionModel(
        input_dim=save_dict["input_dim"],
        hidden_dim=save_dict["hidden_dim"],
        num_blocks=save_dict["num_blocks"],
        pred_target=save_dict["pred_target"],
    )
    model.load_state_dict(save_dict["state_dict"])
    model = model.to(device).eval()

    return model


# =============================================================================
# Sampling with DSP Guidance
# =============================================================================


def project_to_2d(points_high_dim: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    """Project high-dimensional points back to 2D for visualization."""
    return points_high_dim @ proj_matrix.T


@torch.enable_grad()
def sample_with_dsp(
    model: SpiralDiffusionModel,
    classifier: SpiralClassifier,
    num_samples: int,
    num_steps: int,
    guidance_scale: float,
    target_class: int,
    config: InferenceConfig,
) -> torch.Tensor:
    """Sample from diffusion model with DSP (classifier) guidance."""
    device = config.device

    z = torch.randn(num_samples, model.input_dim, device=device)
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    for i in tqdm(range(num_steps), desc=f"Sampling (s={guidance_scale})", leave=False):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t

        t_batch = t.expand(num_samples)

        with torch.no_grad():
            v_pred = model.predict_v(z, t_batch, config.t_eps)

        if guidance_scale > 0.0:
            z_grad = z.clone().detach().requires_grad_(True)
            x0_pred = model.predict_x0(z_grad, t_batch, config.t_eps)

            # Use get_log_probs for proper binary/multi-class handling
            log_probs = classifier.get_log_probs(x0_pred)
            target_log_prob = log_probs[:, target_class].sum()

            grad = torch.autograd.grad(target_log_prob, z_grad)[0]
            v_pred = v_pred + guidance_scale * grad

        z = z + dt * v_pred

    return z.detach()


# =============================================================================
# Metrics
# =============================================================================


def compute_mmd(x: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None) -> float:
    """Compute MMD (Maximum Mean Discrepancy) with Gaussian kernel.

    Bandwidth is set via median heuristic: sigma^2 = median(||xi - xj||^2).

    Args:
        x: Samples from first distribution, shape (N, D).
        y: Samples from second distribution, shape (M, D).
        rng: Random number generator for reproducible subsampling.

    Returns:
        MMD^2 value (unbiased estimate).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Subsample for efficiency if too many points
    max_n = 2000
    if len(x) > max_n:
        idx = rng.choice(len(x), max_n, replace=False)
        x = x[idx]
    if len(y) > max_n:
        idx = rng.choice(len(y), max_n, replace=False)
        y = y[idx]

    n, m = len(x), len(y)
    if n <= 1 or m <= 1:
        return 0.0

    # Compute pairwise distances for bandwidth selection (memory-efficient)
    xy = np.vstack([x, y])
    dists = cdist(xy, xy, metric="sqeuclidean")
    # Median heuristic (exclude zero diagonal)
    median_dist_sq = np.median(dists[dists > 0])
    sigma_sq = median_dist_sq if median_dist_sq > 0 else 1.0

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = cdist(a, b, metric="sqeuclidean")
        return np.exp(-d / (2 * sigma_sq))

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)

    # Unbiased MMD^2 estimator
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)

    mmd_sq = kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.sum() / (n * m)
    return float(max(mmd_sq, 0.0))


def compute_kl_divergence(generated_2d: np.ndarray, target_gt_2d: np.ndarray) -> float | None:
    """Compute KL divergence KL(p_gen || p_target) via dual KDE.

    Args:
        generated_2d: Generated samples in 2D, shape (N, 2).
        target_gt_2d: Target class ground truth in 2D, shape (M, 2).

    Returns:
        KL divergence estimate.
    """
    try:
        kde_target = gaussian_kde(target_gt_2d.T)
        kde_gen = gaussian_kde(generated_2d.T)

        # Evaluate log densities on generated samples
        log_p_gen = kde_gen.logpdf(generated_2d.T)
        log_p_target = kde_target.logpdf(generated_2d.T)

        # KL(p_gen || p_target) = E_p_gen[log p_gen - log p_target]
        kl = float(np.mean(log_p_gen - log_p_target))
        return kl
    except (np.linalg.LinAlgError, ValueError):
        return None


def compute_on_manifold_rate_crossed_lines(
    generated_2d: np.ndarray,
    gt_2d: np.ndarray,
    gt_labels: np.ndarray,
    target_class: int,
) -> float:
    """Compute on-manifold rate for crossed_lines dataset.

    Measures perpendicular distance from each generated point to the target line.
    - Class 0 (y=x): distance = |x - y| / sqrt(2)
    - Class 1 (y=-x): distance = |x + y| / sqrt(2)

    Threshold delta is calibrated from 95th percentile of GT distances.

    Args:
        generated_2d: Generated samples in 2D, shape (N, 2).
        gt_2d: Ground truth points in 2D, shape (M, 2).
        gt_labels: Ground truth labels, shape (M,).
        target_class: Target class index.

    Returns:
        Fraction of generated samples within delta of the manifold.
    """
    # Compute GT distances for calibration
    gt_target = gt_2d[gt_labels == target_class]
    if len(gt_target) == 0:
        return 0.0
    if target_class == 0:
        gt_dists = np.abs(gt_target[:, 0] - gt_target[:, 1]) / np.sqrt(2)
    else:
        gt_dists = np.abs(gt_target[:, 0] + gt_target[:, 1]) / np.sqrt(2)

    delta = np.percentile(gt_dists, 95)

    # Compute generated distances
    if target_class == 0:
        gen_dists = np.abs(generated_2d[:, 0] - generated_2d[:, 1]) / np.sqrt(2)
    else:
        gen_dists = np.abs(generated_2d[:, 0] + generated_2d[:, 1]) / np.sqrt(2)

    return float(np.mean(gen_dists < delta))


def compute_on_manifold_rate_half_arcs(
    generated_2d: np.ndarray,
    gt_2d: np.ndarray,
    gt_labels: np.ndarray,
    target_class: int,
) -> float:
    """Compute on-manifold rate for half_arcs dataset.

    Measures radial deviation from the semicircle arc.
    The arc radius in normalized space is estimated from GT data.

    Args:
        generated_2d: Generated samples in 2D, shape (N, 2).
        gt_2d: Ground truth points in 2D, shape (M, 2).
        gt_labels: Ground truth labels, shape (M,).
        target_class: Target class index.

    Returns:
        Fraction of generated samples within delta of the manifold.
    """
    gt_target = gt_2d[gt_labels == target_class]
    if len(gt_target) == 0:
        return 0.0

    # Estimate arc radius from GT (median distance from origin)
    gt_radii = np.sqrt(gt_target[:, 0] ** 2 + gt_target[:, 1] ** 2)
    arc_radius = np.median(gt_radii)

    # Calibrate threshold from GT radial deviations
    gt_radial_dev = np.abs(gt_radii - arc_radius)
    delta = np.percentile(gt_radial_dev, 95)

    # Also check angular range (class 0: upper, class 1: lower)
    gen_radii = np.sqrt(generated_2d[:, 0] ** 2 + generated_2d[:, 1] ** 2)
    gen_radial_dev = np.abs(gen_radii - arc_radius)

    # Check radial closeness
    on_manifold = gen_radial_dev < delta

    # Also check if in correct half (upper or lower semicircle)
    if target_class == 0:
        correct_half = generated_2d[:, 1] >= 0  # Upper half
    else:
        correct_half = generated_2d[:, 1] <= 0  # Lower half

    return float(np.mean(on_manifold & correct_half))


def compute_on_manifold_rate(
    generated_2d: np.ndarray,
    gt_2d: np.ndarray,
    gt_labels: np.ndarray,
    target_class: int,
    data_name: str,
) -> float | None:
    """Compute on-manifold rate for a given dataset type.

    Args:
        generated_2d: Generated samples in 2D, shape (N, 2).
        gt_2d: Ground truth points in 2D, shape (M, 2).
        gt_labels: Ground truth labels, shape (M,).
        target_class: Target class index.
        data_name: Name of the dataset (e.g., "crossed_lines", "half_arcs").

    Returns:
        On-manifold rate (float), or None if dataset type not supported.
    """
    if data_name == "crossed_lines":
        return compute_on_manifold_rate_crossed_lines(
            generated_2d, gt_2d, gt_labels, target_class,
        )
    elif data_name == "half_arcs":
        return compute_on_manifold_rate_half_arcs(
            generated_2d, gt_2d, gt_labels, target_class,
        )
    else:
        return None


def compute_target_class_accuracy(
    samples_high_dim: torch.Tensor,
    classifier: "SpiralClassifier",
    target_class: int,
    device: str,
) -> float:
    """Compute fraction of generated samples classified as target class.

    Args:
        samples_high_dim: Generated samples in high-D space, shape (N, D).
        classifier: Trained classifier.
        target_class: Target class index.
        device: Device string.

    Returns:
        Fraction of samples predicted as target class.
    """
    with torch.no_grad():
        if not isinstance(samples_high_dim, torch.Tensor):
            samples_high_dim = torch.from_numpy(samples_high_dim).to(device)
        log_probs = classifier.get_log_probs(samples_high_dim)
        preds = log_probs.argmax(dim=1)
        accuracy = float((preds == target_class).float().mean().item())
    return accuracy


def compute_all_metrics(
    generated_2d: np.ndarray,
    samples_high_dim: torch.Tensor,
    gt_2d: np.ndarray,
    gt_labels: np.ndarray,
    classifier: "SpiralClassifier",
    target_class: int,
    data_name: str,
    device: str,
) -> dict:
    """Compute all metrics for a single (D, pred_target) configuration.

    All spatial metrics are computed in 2D (after back-projection).

    Args:
        generated_2d: Generated samples projected to 2D, shape (N, 2).
        samples_high_dim: Generated samples in high-D space, shape (N, D).
        gt_2d: Ground truth 2D points, shape (M, 2).
        gt_labels: Ground truth labels, shape (M,).
        classifier: Trained classifier.
        target_class: Target class index.
        data_name: Dataset name.
        device: Device string.

    Returns:
        Dict with metric names and values.
    """
    gt_target_2d = gt_2d[gt_labels == target_class]

    rng = np.random.default_rng(42)
    metrics = {}

    # On-manifold rate
    metrics["on_manifold_rate"] = compute_on_manifold_rate(
        generated_2d, gt_2d, gt_labels, target_class, data_name,
    )

    # Source MMD: MMD(generated, all GT)
    metrics["source_mmd"] = compute_mmd(generated_2d, gt_2d, rng=rng)

    # Target MMD: MMD(generated, target class GT)
    metrics["target_mmd"] = compute_mmd(generated_2d, gt_target_2d, rng=rng)

    # KL Divergence
    metrics["kl_div"] = compute_kl_divergence(generated_2d, gt_target_2d)

    # Target class accuracy
    metrics["class_accuracy"] = compute_target_class_accuracy(
        samples_high_dim, classifier, target_class, device,
    )

    # Convert any remaining NaN values to None for JSON serialization
    for k, v in metrics.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            metrics[k] = None

    return metrics


# =============================================================================
# Visualization
# =============================================================================


def create_classification_background(
    classifier: SpiralClassifier,
    proj_matrix: np.ndarray,
    D: int,
    resolution: int,
    device: str,
    data_range: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create classification landscape for background."""
    margin = data_range + 0.5
    x_range = np.linspace(-margin, margin, resolution)
    y_range = np.linspace(-margin, margin, resolution)
    xx, yy = np.meshgrid(x_range, y_range)

    grid_2d = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    if D == 2:
        grid_high = grid_2d
    else:
        grid_high = grid_2d @ proj_matrix

    with torch.no_grad():
        grid_tensor = torch.from_numpy(grid_high).to(device)
        log_probs = classifier.get_log_probs(grid_tensor)
        preds = log_probs.argmax(dim=1).cpu().numpy()

    class_map = preds.reshape(xx.shape)

    return xx, yy, class_map


def visualize_results(
    gt_data_2d: np.ndarray,
    gt_labels: np.ndarray,
    generated_samples: dict[tuple[int, str], np.ndarray],
    classifiers: dict[int, SpiralClassifier],
    proj_matrices: dict[int, np.ndarray],
    d_values: list[int],
    class_info: dict,
    guidance_scale: float,
    num_steps: int,
    target_class: int,
    config: InferenceConfig,
    results_dir: Path,
    data_range: float = 3.0,
) -> None:
    """Create and save paper-quality visualization figure."""
    pred_targets = ["x", "e", "v"]
    num_rows = len(d_values)
    num_cols = 4  # GT + 3 pred targets

    # Dynamic figsize: 3.5 inches per cell (readable at ~2x shrink to paper column)
    cell_size = 3.5
    figsize = config.figsize or (cell_size * num_cols, cell_size * num_rows)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=figsize,
        squeeze=False,
    )

    # Get colors from class_info
    class_colors = {int(k): v["color"] for k, v in class_info["classes"].items()}
    num_classes = class_info["num_classes"]

    # Light background colors (same as class colors, alpha applied in contourf)
    bg_colors = {i: class_colors[i] for i in range(num_classes)}

    # Use target class color for generated samples
    color_gen = class_colors[target_class]

    margin = data_range + 0.5

    # Paper-quality formatting constants (readable at ~2x shrink to 7-inch column)
    title_fontsize = 12
    ylabel_fontsize = 11
    tick_fontsize = 8
    point_size = max(config.point_size, 3)

    for row_idx, D in enumerate(d_values):
        classifier = classifiers[D]
        proj_matrix = proj_matrices[D]

        xx, yy, class_map = create_classification_background(
            classifier,
            proj_matrix,
            D,
            config.background_resolution,
            config.device,
            data_range,
        )

        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]

            # Draw classification background
            levels = [-0.5 + i for i in range(num_classes + 1)]
            colors = [bg_colors[i] for i in range(num_classes)]
            ax.contourf(xx, yy, class_map, levels=levels, colors=colors, alpha=0.3)

            if col_idx == 0:
                # Ground Truth
                for label in range(num_classes):
                    mask = gt_labels == label
                    ax.scatter(
                        gt_data_2d[mask, 0],
                        gt_data_2d[mask, 1],
                        c=class_colors[label],
                        s=point_size,
                        alpha=0.6,
                        label=f"Class {label}",
                    )
                ax.set_ylabel(f"D={D}", fontsize=ylabel_fontsize, fontweight="bold")
            else:
                pred_target = pred_targets[col_idx - 1]
                key = (D, pred_target)
                samples_2d = generated_samples.get(key)

                if samples_2d is not None:
                    ax.scatter(
                        samples_2d[:, 0],
                        samples_2d[:, 1],
                        c=color_gen,
                        s=point_size,
                        alpha=0.6,
                    )

            ax.set_xlim(-margin, margin)
            ax.set_ylim(-margin, margin)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=tick_fontsize)

            # Remove inner tick labels for cleaner look
            if col_idx > 0:
                ax.tick_params(labelleft=False)
            if row_idx < num_rows - 1:
                ax.tick_params(labelbottom=False)

            # Consistent axis spines
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("gray")

            if row_idx == 0:
                if col_idx == 0:
                    ax.set_title("Ground Truth", fontsize=title_fontsize, fontweight="bold")
                else:
                    ax.set_title(
                        f"{pred_targets[col_idx - 1]}-prediction",
                        fontsize=title_fontsize,
                        fontweight="bold",
                    )

    plt.suptitle(
        f"Spiral Test: Guidance Scale={guidance_scale}, Steps={num_steps}",
        fontsize=title_fontsize + 2,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    filename = f"s{guidance_scale}_steps{num_steps}.png"
    filepath = results_dir / filename
    plt.savefig(filepath, dpi=config.dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filepath}")


# =============================================================================
# Main Inference Pipeline
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference for JiT + TFG Spiral Test")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment folder name in spiral_test/output/",
    )
    parser.add_argument(
        "--guidance_scale",
        "-s",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 5.0],
        help="Guidance scale(s) to use (default: 0.0 2.0 5.0)",
    )
    parser.add_argument(
        "--num_steps",
        "-n",
        type=int,
        nargs="+",
        default=[50, 100],
        help="Number of sampling steps (default: 50 100)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to generate per configuration (default: 10000)",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=1,
        help="Target class for guidance (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the inference pipeline."""
    args = parse_args()

    config = InferenceConfig(
        num_samples=args.num_samples,
        seed=args.seed,
        device=args.device,
    )

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.device == "cuda":
        torch.cuda.manual_seed(config.seed)

    # Get experiment directory
    exp_dir = get_output_dir() / args.exp
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}\nPlease run train.py first.")

    # Parse data name from experiment folder
    data_name = parse_data_name_from_exp(args.exp)

    # Create results directory inside experiment folder
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")
    print(f"Data name: {data_name}")
    print(f"Results directory: {results_dir}")
    print(f"Device: {config.device}")
    print(f"Guidance scales: {args.guidance_scale}")
    print(f"Num steps: {args.num_steps}")
    print(f"Target class: {args.target_class}")
    print("=" * 60)

    # Load training config
    train_config = load_train_config(exp_dir)
    d_values = train_config["d_values"]

    # Load data from data directory
    print("\n[Step 1] Loading data...")
    points_2d, labels, class_info = load_data(data_name)
    data_config = load_data_config(data_name)
    data_range = data_config.get("data_range", 3.0)

    # Load models
    print("\n[Step 2] Loading models...")
    proj_matrices: dict[int, np.ndarray] = {}
    classifiers: dict[int, SpiralClassifier] = {}
    diffusion_models: dict[tuple[int, str], SpiralDiffusionModel] = {}

    for D in d_values:
        print(f"  Loading models for D={D}...")
        proj_matrices[D] = load_projection_matrix(D, exp_dir)
        classifiers[D] = load_classifier(D, exp_dir, config.device)

        for pred_target in ["x", "e", "v"]:
            diffusion_models[(D, pred_target)] = load_diffusion_model(D, pred_target, exp_dir, config.device)

    # Sample and visualize
    print("\n" + "=" * 60)
    print("Sampling and Visualization")
    print("=" * 60)

    def _fmt(v, fmt=".3f"):
        return f"{v:{fmt}}" if v is not None else "N/A"

    for num_steps in args.num_steps:
        for guidance_scale in args.guidance_scale:
            print(f"\n[Step 3] Sampling with s={guidance_scale}, steps={num_steps}...")

            generated_samples: dict[tuple[int, str], np.ndarray] = {}
            # Store high-dim samples for classifier accuracy
            generated_samples_highd: dict[tuple[int, str], torch.Tensor] = {}

            for D in d_values:
                for pred_target in ["x", "e", "v"]:
                    model = diffusion_models[(D, pred_target)]
                    classifier = classifiers[D]

                    samples_D = sample_with_dsp(
                        model=model,
                        classifier=classifier,
                        num_samples=config.num_samples,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                        target_class=args.target_class,
                        config=config,
                    )

                    generated_samples_highd[(D, pred_target)] = samples_D

                    samples_2d = project_to_2d(
                        samples_D.cpu().numpy(),
                        proj_matrices[D],
                    )
                    generated_samples[(D, pred_target)] = samples_2d

            # Compute metrics
            print(f"\n[Step 4] Computing metrics...")
            all_metrics: dict[str, dict[str, dict]] = {}

            for D in d_values:
                d_key = f"D{D}"
                all_metrics[d_key] = {}
                for pred_target in ["x", "e", "v"]:
                    key = (D, pred_target)
                    samples_2d = generated_samples[key]
                    samples_hd = generated_samples_highd[key]

                    metrics = compute_all_metrics(
                        generated_2d=samples_2d,
                        samples_high_dim=samples_hd,
                        gt_2d=points_2d,
                        gt_labels=labels,
                        classifier=classifiers[D],
                        target_class=args.target_class,
                        data_name=data_name,
                        device=config.device,
                    )
                    all_metrics[d_key][pred_target] = metrics

                    print(
                        f"  D={D} {pred_target}-pred: "
                        f"on_manifold={_fmt(metrics['on_manifold_rate'])} "
                        f"src_mmd={_fmt(metrics['source_mmd'], '.4f')} "
                        f"tgt_mmd={_fmt(metrics['target_mmd'], '.4f')} "
                        f"kl={_fmt(metrics['kl_div'])} "
                        f"acc={_fmt(metrics['class_accuracy'])}"
                    )

            # Save metrics JSON
            metrics_output = {
                "guidance_scale": guidance_scale,
                "num_steps": num_steps,
                "target_class": args.target_class,
                "data_name": data_name,
                "num_samples": config.num_samples,
                "metrics": all_metrics,
            }
            metrics_filename = f"metrics_s{guidance_scale}_steps{num_steps}.json"
            metrics_filepath = results_dir / metrics_filename
            with open(metrics_filepath, "w") as f:
                json.dump(metrics_output, f, indent=2)
            print(f"Saved metrics: {metrics_filepath}")

            # Free GPU tensors after metrics are saved
            del generated_samples_highd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n[Step 5] Creating visualization...")
            visualize_results(
                gt_data_2d=points_2d,
                gt_labels=labels,
                generated_samples=generated_samples,
                classifiers=classifiers,
                proj_matrices=proj_matrices,
                d_values=d_values,
                class_info=class_info,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                target_class=args.target_class,
                config=config,
                results_dir=results_dir,
                data_range=data_range,
            )

    # Save inference config
    inference_config = {
        "experiment": args.exp,
        "data_name": data_name,
        "guidance_scales": args.guidance_scale,
        "num_steps": args.num_steps,
        "num_samples": config.num_samples,
        "target_class": args.target_class,
        "seed": config.seed,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(results_dir / "inference_config.json", "w") as f:
        json.dump(inference_config, f, indent=2)

    print("\n" + "=" * 60)
    print("Inference completed!")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
