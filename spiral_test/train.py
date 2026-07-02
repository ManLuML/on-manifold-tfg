#!/usr/bin/env python3
"""Training script for JiT + TFG Spiral Test.

This script trains all components for the spiral test experiment:
1. Load data from spiral_test/data/<dataset_name>/
2. Create projection matrices for each D
3. Train classifiers for each D
4. Train diffusion models for each (D, pred_target)

All artifacts are saved to spiral_test/output/<dataset_name>_<timestamp>/.

Usage:
    cd on-manifold-tfg
    uv run python spiral_test/train.py --data doublespiral
    uv run python spiral_test/train.py --data doublespiral --d_values 2 8 16
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainConfig:
    """Configuration for training the spiral test models."""

    # Data
    data_name: str = "doublespiral"
    d_values: tuple[int, ...] = (2, 8, 32, 128, 512)

    # Classifier
    classifier_hidden_dims: tuple[int, ...] = (128, 128, 128)
    classifier_epochs: int = 100
    classifier_lr: float = 1e-3
    classifier_batch_size: int = 256

    # Diffusion Model
    diffusion_hidden_dim: int = 256
    diffusion_num_blocks: int = 5
    diffusion_epochs: int = 500
    diffusion_lr: float = 1e-3
    diffusion_batch_size: int = 256

    # Device & Seed
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Validate and adjust device."""
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"


# =============================================================================
# Data Loading
# =============================================================================


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent / "data"


def get_output_dir() -> Path:
    """Get the output directory path."""
    return Path(__file__).parent / "output"


def load_data(data_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load data from data directory.

    Args:
        data_name: Name of the dataset folder.

    Returns:
        Tuple of (points_2d, labels, class_info).
    """
    data_dir = get_data_dir() / data_name

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}\nPlease run generate_data.py first.")

    # Load data
    data = np.load(data_dir / "data.npz")
    points_2d = data["points_2d"]
    labels = data["labels"]

    # Load class info
    with open(data_dir / "class_info.json") as f:
        class_info = json.load(f)

    return points_2d, labels, class_info


# =============================================================================
# Projection
# =============================================================================


def create_projection_matrix(d_in: int, d_out: int, seed: int = 42) -> np.ndarray:
    """Create a random column-orthogonal projection matrix.

    Args:
        d_in: Input dimension (2 for spiral).
        d_out: Output dimension (D).
        seed: Random seed.

    Returns:
        Projection matrix of shape (d_in, d_out).
    """
    rng = np.random.default_rng(seed)

    # Generate random matrix with d_in columns in d_out-dimensional space
    A = rng.standard_normal((d_out, d_in))

    # QR decomposition gives orthonormal columns
    Q, _ = np.linalg.qr(A)

    # Return transposed: (d_in, d_out) for: points @ proj_matrix
    return Q.T.astype(np.float32)


def project_to_high_dim(
    points_2d: np.ndarray,
    D: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 2D points to D-dimensional space.

    Args:
        points_2d: (N, 2) array of 2D points.
        D: Target dimension.
        seed: Random seed for projection matrix.

    Returns:
        Tuple of (projected_points, projection_matrix).
    """
    if D == 2:
        return points_2d.copy(), np.eye(2, dtype=np.float32)

    proj_matrix = create_projection_matrix(2, D, seed)
    projected = points_2d @ proj_matrix

    return projected, proj_matrix


# =============================================================================
# Models
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
    """MLP classifier for spiral data.

    Supports both binary classification (2 classes, sigmoid) and
    multi-class classification (3+ classes, softmax).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (128, 128, 128),
    ) -> None:
        """Initialize classifier.

        Args:
            input_dim: Input dimension (D).
            num_classes: Number of classes.
            hidden_dims: Tuple of hidden layer dimensions.
        """
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

        # Output layer: 1 for binary, num_classes for multi-class
        if num_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (B, D).

        Returns:
            For binary classification: Logits of shape (B, 1).
            For multi-class: Logits of shape (B, num_classes).
        """
        return self.net(x)

    def get_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for each class.

        Args:
            x: Input of shape (B, D).

        Returns:
            Log probabilities of shape (B, num_classes).
        """
        logits = self.forward(x)

        if self.num_classes == 2:
            # Binary: use sigmoid
            prob_1 = torch.sigmoid(logits)
            prob_0 = 1 - prob_1
            probs = torch.cat([prob_0, prob_1], dim=-1)
            return torch.log(probs + 1e-8)
        else:
            # Multi-class: use softmax
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


# =============================================================================
# Training Functions
# =============================================================================


def train_classifier(
    classifier: SpiralClassifier,
    data: torch.Tensor,
    labels: torch.Tensor,
    config: TrainConfig,
) -> SpiralClassifier:
    """Train the classifier on clean data.

    Args:
        classifier: Classifier model.
        data: Training data of shape (N, D).
        labels: Labels of shape (N,).
        config: Training configuration.

    Returns:
        Trained classifier.
    """
    device = config.device
    classifier = classifier.to(device)
    data = data.to(device)
    labels = labels.to(device)

    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=config.classifier_batch_size, shuffle=True)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.classifier_lr)

    # Use appropriate loss function
    if classifier.num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    classifier.train()
    pbar = tqdm(range(config.classifier_epochs), desc="Training Classifier")

    for _epoch in pbar:
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            optimizer.zero_grad()
            logits = classifier(x)

            if classifier.num_classes == 2:
                loss = criterion(logits.squeeze(-1), y.float())
                preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            else:
                loss = criterion(logits, y)
                preds = logits.argmax(dim=1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            total += x.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    classifier.eval()
    return classifier


def train_diffusion_model(
    model: SpiralDiffusionModel,
    data: torch.Tensor,
    config: TrainConfig,
) -> SpiralDiffusionModel:
    """Train the diffusion model.

    Args:
        model: Diffusion model.
        data: Training data of shape (N, D).
        config: Training configuration.

    Returns:
        Trained diffusion model.
    """
    device = config.device
    model = model.to(device)
    data = data.to(device)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=config.diffusion_batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.diffusion_lr)

    model.train()
    pbar = tqdm(
        range(config.diffusion_epochs),
        desc=f"Training Diffusion ({model.pred_target}-pred)",
    )

    for _epoch in pbar:
        total_loss = 0.0
        total = 0

        for (x,) in loader:
            optimizer.zero_grad()

            t = torch.rand(x.size(0), device=device)
            e = torch.randn_like(x)

            t_expanded = t.view(-1, 1)
            z = t_expanded * x + (1 - t_expanded) * e

            net_out = model(z, t)

            if model.pred_target == "x":
                target = x
            elif model.pred_target == "e":
                target = e
            elif model.pred_target == "v":
                target = x - e
            else:
                raise ValueError(f"Unknown pred_target: {model.pred_target}")

            loss = F.mse_loss(net_out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        avg_loss = total_loss / total
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    model.eval()
    return model


# =============================================================================
# Checkpoint Management
# =============================================================================


def save_projection_matrix(
    proj_matrix: np.ndarray,
    D: int,
    output_dir: Path,
) -> None:
    """Save projection matrix for dimension D."""
    d_dir = output_dir / f"D{D}"
    d_dir.mkdir(parents=True, exist_ok=True)
    np.save(d_dir / "proj_matrix.npy", proj_matrix)
    print(f"Saved projection matrix to {d_dir / 'proj_matrix.npy'}")


def save_classifier(
    classifier: SpiralClassifier,
    D: int,
    output_dir: Path,
) -> None:
    """Save classifier for dimension D."""
    d_dir = output_dir / f"D{D}"
    d_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "state_dict": classifier.state_dict(),
        "input_dim": classifier.input_dim,
        "num_classes": classifier.num_classes,
        "hidden_dims": classifier.hidden_dims,
    }
    torch.save(save_dict, d_dir / "classifier.pt")
    print(f"Saved classifier to {d_dir / 'classifier.pt'}")


def save_diffusion_model(
    model: SpiralDiffusionModel,
    D: int,
    pred_target: str,
    output_dir: Path,
) -> None:
    """Save diffusion model for dimension D and prediction target."""
    d_dir = output_dir / f"D{D}"
    d_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_blocks": model.num_blocks,
        "pred_target": model.pred_target,
    }
    torch.save(save_dict, d_dir / f"diffusion_{pred_target}.pt")
    print(f"Saved diffusion model to {d_dir / f'diffusion_{pred_target}.pt'}")


# =============================================================================
# Main Training Pipeline
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train models for spiral test")
    parser.add_argument(
        "--data",
        type=str,
        default="doublespiral",
        help="Dataset name in spiral_test/data/ (default: doublespiral)",
    )
    parser.add_argument(
        "--d_values",
        type=int,
        nargs="+",
        default=[2, 8, 32, 128, 512],
        help="D values to train (default: 2 8 32 128 512)",
    )
    parser.add_argument(
        "--diffusion_epochs",
        type=int,
        default=500,
        help="Number of diffusion training epochs (default: 500)",
    )
    parser.add_argument(
        "--classifier_epochs",
        type=int,
        default=100,
        help="Number of classifier training epochs (default: 100)",
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
    """Run the full training pipeline."""
    args = parse_args()

    config = TrainConfig(
        data_name=args.data,
        d_values=tuple(args.d_values),
        diffusion_epochs=args.diffusion_epochs,
        classifier_epochs=args.classifier_epochs,
        seed=args.seed,
        device=args.device,
    )

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.device == "cuda":
        torch.cuda.manual_seed(config.seed)

    # Load data
    print(f"Loading data from: {config.data_name}")
    points_2d, labels, class_info = load_data(config.data_name)
    labels_tensor = torch.from_numpy(labels)
    num_classes = class_info["num_classes"]

    print(f"Data loaded: {len(labels)} points, {num_classes} classes")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_dir() / f"{config.data_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_dict = asdict(config)
    for key, value in config_dict.items():
        if isinstance(value, tuple):
            config_dict[key] = list(value)
    config_dict["num_classes"] = num_classes

    with open(output_dir / "train_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nOutput directory: {output_dir}")
    print(f"Device: {config.device}")
    print("=" * 60)

    # Train for each D
    for D in config.d_values:
        print(f"\n{'=' * 60}")
        print(f"Processing D={D}")
        print("=" * 60)

        # Project to D-dimensional space
        points_D, proj_matrix = project_to_high_dim(points_2d, D, seed=config.seed)
        data_tensor = torch.from_numpy(points_D)

        # Save projection matrix
        save_projection_matrix(proj_matrix, D, output_dir)

        # Train and save classifier
        print(f"\n[Step 1] Training classifier for D={D}...")
        classifier = SpiralClassifier(
            input_dim=D,
            num_classes=num_classes,
            hidden_dims=config.classifier_hidden_dims,
        )
        classifier = train_classifier(classifier, data_tensor, labels_tensor, config)
        save_classifier(classifier, D, output_dir)

        # Train and save diffusion models for each prediction target
        for pred_target in ["x", "e", "v"]:
            print(f"\n[Step 2] Training {pred_target}-prediction diffusion model for D={D}...")
            model = SpiralDiffusionModel(
                input_dim=D,
                hidden_dim=config.diffusion_hidden_dim,
                num_blocks=config.diffusion_num_blocks,
                pred_target=pred_target,
            )
            model = train_diffusion_model(model, data_tensor, config)
            save_diffusion_model(model, D, pred_target, output_dir)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"All checkpoints saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
