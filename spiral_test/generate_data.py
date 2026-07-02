#!/usr/bin/env python3
"""Data generation script for JiT + TFG Spiral Test.

This script generates toy datasets for the spiral test experiment.

Supported dataset types:
- doublespiral: Two interleaved spirals (2 classes)
- concentric_rings: Inner and outer rings (2 classes)
- circular_gaussians: 8 Gaussians on a circle (4 classes)
- grid_gaussians: 3x3 grid of Gaussians (9 classes)
- crossed_lines: X-shaped crossed lines (2 classes)
- half_arcs: Upper and lower semicircle arcs (2 classes)

Generated files are saved to spiral_test/data/<dataset_name>/:
- data.npz: Contains points_2d and labels arrays
- visualization.png: Scatter plot of the data
- class_info.json: Class metadata (count, colors)

Usage:
    cd on-manifold-tfg
    uv run python spiral_test/generate_data.py --name doublespiral
    uv run python spiral_test/generate_data.py --name concentric_rings
    uv run python spiral_test/generate_data.py --name circular_gaussians
    uv run python spiral_test/generate_data.py --name grid_gaussians
    uv run python spiral_test/generate_data.py --name crossed_lines
    uv run python spiral_test/generate_data.py --name half_arcs
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DataConfig:
    """Configuration for data generation."""

    # Dataset
    name: str = "doublespiral"
    total_points: int = 12000
    noise: float = 0.5
    data_range: float = 3.0

    # Seed
    seed: int = 42


# Default colors for classes
CLASS_COLORS = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


# =============================================================================
# Data Generation Functions
# =============================================================================


def _compute_spiral_arc_length(theta: np.ndarray, a: float) -> np.ndarray:
    """Compute arc length of Archimedean spiral r = a * theta."""
    return (a / 2) * (theta * np.sqrt(1 + theta**2) + np.arcsinh(theta))


def _sample_uniform_arc_length(n: int, max_theta: float, a: float) -> np.ndarray:
    """Sample theta values to get uniform arc length distribution."""
    total_arc = _compute_spiral_arc_length(max_theta, a)
    target_arcs = np.linspace(0, total_arc, n)
    theta_dense = np.linspace(0, max_theta, n * 10)
    arc_dense = _compute_spiral_arc_length(theta_dense, a)
    theta_uniform = np.interp(target_arcs, arc_dense, theta_dense)
    return theta_uniform


def generate_double_spiral(
    total_points: int,
    noise: float = 0.5,
    data_range: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate 2D double spiral dataset with uniform density.

    Args:
        total_points: Total number of points (split evenly between classes).
        noise: Standard deviation of Gaussian noise.
        data_range: Range for normalization [-data_range, data_range].
        seed: Random seed.

    Returns:
        Tuple of (points, labels, num_classes).
    """
    rng = np.random.default_rng(seed)

    n = total_points // 2  # Points per class
    max_theta = 4 * np.pi
    a = 5 / max_theta

    # Class 0: First spiral
    theta0 = _sample_uniform_arc_length(n, max_theta, a)
    r0 = a * theta0
    x0 = r0 * np.cos(theta0) + rng.normal(0, noise, n) * 0.1
    y0 = r0 * np.sin(theta0) + rng.normal(0, noise, n) * 0.1

    # Class 1: Second spiral (rotated by 180 degrees)
    theta1 = _sample_uniform_arc_length(n, max_theta, a)
    r1 = a * theta1
    x1 = r1 * np.cos(theta1 + np.pi) + rng.normal(0, noise, n) * 0.1
    y1 = r1 * np.sin(theta1 + np.pi) + rng.normal(0, noise, n) * 0.1

    points = np.vstack([
        np.column_stack([x0, y0]),
        np.column_stack([x1, y1]),
    ]).astype(np.float32)

    labels = np.concatenate([
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
    ])

    points = points / np.abs(points).max() * data_range

    return points, labels, 2


def generate_concentric_rings(
    total_points: int,
    noise: float = 0.1,
    data_range: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate concentric rings dataset.

    Inner ring (class 0) and outer ring (class 1) with radius ratio 1:2.
    Point distribution is 1:2 to maintain uniform density.

    Args:
        total_points: Total number of points.
        noise: Standard deviation of radial noise.
        data_range: Range for normalization.
        seed: Random seed.

    Returns:
        Tuple of (points, labels, num_classes).
    """
    rng = np.random.default_rng(seed)

    # Split points 1:2 ratio (inner:outer) to maintain density
    n_inner = total_points // 3
    n_outer = total_points - n_inner

    r_inner = 1.0
    r_outer = 2.0

    # Inner ring (class 0)
    theta_inner = rng.uniform(0, 2 * np.pi, n_inner)
    r_inner_noisy = r_inner + rng.normal(0, noise, n_inner)
    x_inner = r_inner_noisy * np.cos(theta_inner)
    y_inner = r_inner_noisy * np.sin(theta_inner)

    # Outer ring (class 1)
    theta_outer = rng.uniform(0, 2 * np.pi, n_outer)
    r_outer_noisy = r_outer + rng.normal(0, noise, n_outer)
    x_outer = r_outer_noisy * np.cos(theta_outer)
    y_outer = r_outer_noisy * np.sin(theta_outer)

    points = np.vstack([
        np.column_stack([x_inner, y_inner]),
        np.column_stack([x_outer, y_outer]),
    ]).astype(np.float32)

    labels = np.concatenate([
        np.zeros(n_inner, dtype=np.int64),
        np.ones(n_outer, dtype=np.int64),
    ])

    points = points / np.abs(points).max() * data_range

    return points, labels, 2


def generate_circular_gaussians(
    total_points: int,
    noise: float = 0.3,
    data_range: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate 8 Gaussians arranged in a circle with 4 classes.

    Arrangement (45 degree intervals):
    - 0°, 180° (right, left): class 1
    - 90°, 270° (top, bottom): class 0
    - 45°, 225° (y=x direction): class 2
    - 135°, 315° (y=-x direction): class 3

    Args:
        total_points: Total number of points.
        noise: Standard deviation of each Gaussian.
        data_range: Range for normalization.
        seed: Random seed.

    Returns:
        Tuple of (points, labels, num_classes).
    """
    rng = np.random.default_rng(seed)

    # 8 Gaussians, each gets equal points
    n_per_gaussian = total_points // 8
    radius = 2.0  # Radius of the circle where means are placed

    # Angles: 0, 45, 90, 135, 180, 225, 270, 315 degrees
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    angles_rad = [np.deg2rad(a) for a in angles_deg]

    # Class assignments:
    # 0°, 180° -> class 1 (left-right)
    # 90°, 270° -> class 0 (top-bottom)
    # 45°, 225° -> class 2 (y=x direction)
    # 135°, 315° -> class 3 (y=-x direction)
    class_map = {
        0: 1,  # right -> class 1
        45: 2,  # y=x -> class 2
        90: 0,  # top -> class 0
        135: 3,  # y=-x -> class 3
        180: 1,  # left -> class 1
        225: 2,  # y=x -> class 2
        270: 0,  # bottom -> class 0
        315: 3,  # y=-x -> class 3
    }

    all_points = []
    all_labels = []

    for angle_deg, angle_rad in zip(angles_deg, angles_rad, strict=True):
        # Mean position on circle
        mean_x = radius * np.cos(angle_rad)
        mean_y = radius * np.sin(angle_rad)

        # Generate Gaussian samples
        x = rng.normal(mean_x, noise, n_per_gaussian)
        y = rng.normal(mean_y, noise, n_per_gaussian)

        all_points.append(np.column_stack([x, y]))
        all_labels.append(np.full(n_per_gaussian, class_map[angle_deg], dtype=np.int64))

    points = np.vstack(all_points).astype(np.float32)
    labels = np.concatenate(all_labels)

    points = points / np.abs(points).max() * data_range

    return points, labels, 4


def generate_grid_gaussians(
    total_points: int,
    noise: float = 0.2,
    data_range: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate 3x3 grid of Gaussians with 9 classes.

    Grid layout (class numbers):
    [0] [1] [2]
    [3] [4] [5]
    [6] [7] [8]

    Args:
        total_points: Total number of points.
        noise: Standard deviation of each Gaussian.
        data_range: Range for normalization.
        seed: Random seed.

    Returns:
        Tuple of (points, labels, num_classes).
    """
    rng = np.random.default_rng(seed)

    n_per_gaussian = total_points // 9
    spacing = 2.0  # Spacing between grid centers

    all_points = []
    all_labels = []

    for i in range(3):  # row
        for j in range(3):  # column
            class_id = i * 3 + j

            # Center position
            mean_x = (j - 1) * spacing  # -spacing, 0, +spacing
            mean_y = (1 - i) * spacing  # +spacing, 0, -spacing (top to bottom)

            # Generate Gaussian samples
            x = rng.normal(mean_x, noise, n_per_gaussian)
            y = rng.normal(mean_y, noise, n_per_gaussian)

            all_points.append(np.column_stack([x, y]))
            all_labels.append(np.full(n_per_gaussian, class_id, dtype=np.int64))

    points = np.vstack(all_points).astype(np.float32)
    labels = np.concatenate(all_labels)

    points = points / np.abs(points).max() * data_range

    return points, labels, 9


def generate_crossed_lines(
    total_points: int,
    noise: float = 0.1,
    data_range: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate X-shaped crossed lines dataset.

    Two diagonal lines crossing at the origin:
    - Line 1 (class 0): y = x direction
    - Line 2 (class 1): y = -x direction

    Args:
        total_points: Total number of points.
        noise: Standard deviation of perpendicular noise.
        data_range: Range for normalization.
        seed: Random seed.

    Returns:
        Tuple of (points, labels, num_classes).
    """
    rng = np.random.default_rng(seed)

    n_per_line = total_points // 2
    line_length = 4.0

    # Line 1: y = x (class 0)
    t1 = rng.uniform(-line_length / 2, line_length / 2, n_per_line)
    noise1 = rng.normal(0, noise, n_per_line)
    x1 = t1 + noise1 / np.sqrt(2)
    y1 = t1 - noise1 / np.sqrt(2)

    # Line 2: y = -x (class 1)
    t2 = rng.uniform(-line_length / 2, line_length / 2, n_per_line)
    noise2 = rng.normal(0, noise, n_per_line)
    x2 = t2 + noise2 / np.sqrt(2)
    y2 = -t2 + noise2 / np.sqrt(2)

    points = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2]),
    ]).astype(np.float32)

    labels = np.concatenate([
        np.zeros(n_per_line, dtype=np.int64),
        np.ones(n_per_line, dtype=np.int64),
    ])

    points = points / np.abs(points).max() * data_range

    return points, labels, 2


def generate_half_arcs(
    total_points: int,
    noise: float = 0.1,
    data_range: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate half-turn arc dataset (two semicircles).

    Two non-overlapping semicircle arcs:
    - Class 0: Upper semicircle arc (0 to pi), radius ~2
    - Class 1: Lower semicircle arc (pi to 2*pi), radius ~2

    Simple 1D manifold (arc) with clear on-manifold definition.

    Args:
        total_points: Total number of points.
        noise: Standard deviation of radial noise.
        data_range: Range for normalization.
        seed: Random seed.

    Returns:
        Tuple of (points, labels, num_classes).
    """
    rng = np.random.default_rng(seed)

    n_per_arc = total_points // 2
    radius = 2.0

    # Class 0: Upper semicircle (0 to pi)
    theta0 = rng.uniform(0, np.pi, n_per_arc)
    r0 = radius + rng.normal(0, noise, n_per_arc)
    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)

    # Class 1: Lower semicircle (pi to 2*pi)
    theta1 = rng.uniform(np.pi, 2 * np.pi, n_per_arc)
    r1 = radius + rng.normal(0, noise, n_per_arc)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    points = np.vstack([
        np.column_stack([x0, y0]),
        np.column_stack([x1, y1]),
    ]).astype(np.float32)

    labels = np.concatenate([
        np.zeros(n_per_arc, dtype=np.int64),
        np.ones(n_per_arc, dtype=np.int64),
    ])

    points = points / np.abs(points).max() * data_range

    return points, labels, 2


# =============================================================================
# Dataset Registry
# =============================================================================


DATASET_GENERATORS = {
    "doublespiral": generate_double_spiral,
    "concentric_rings": generate_concentric_rings,
    "circular_gaussians": generate_circular_gaussians,
    "grid_gaussians": generate_grid_gaussians,
    "crossed_lines": generate_crossed_lines,
    "half_arcs": generate_half_arcs,
}


# =============================================================================
# Saving Functions
# =============================================================================


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent / "data"


def save_data(points: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """Save data arrays."""
    np.savez(output_dir / "data.npz", points_2d=points, labels=labels)
    print(f"Saved data to {output_dir / 'data.npz'}")


def save_class_info(num_classes: int, output_dir: Path) -> None:
    """Save class information."""
    class_info = {
        "num_classes": num_classes,
        "classes": {
            str(i): {
                "label": i,
                "color": CLASS_COLORS[i % len(CLASS_COLORS)],
                "name": f"Class {i}",
            }
            for i in range(num_classes)
        },
    }

    with open(output_dir / "class_info.json", "w") as f:
        json.dump(class_info, f, indent=2)

    print(f"Saved class info to {output_dir / 'class_info.json'}")


def save_visualization(
    points: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    output_dir: Path,
    data_range: float = 3.0,
) -> None:
    """Save data visualization."""
    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(num_classes):
        mask = labels == i
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            c=color,
            s=2,
            alpha=0.6,
            label=f"Class {i}",
        )

    margin = data_range + 0.5
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Dataset Visualization")

    plt.tight_layout()
    plt.savefig(output_dir / "visualization.png", dpi=150)
    plt.close()

    print(f"Saved visualization to {output_dir / 'visualization.png'}")


def save_config(config: DataConfig, num_classes: int, output_dir: Path) -> None:
    """Save data generation configuration."""
    config_dict = asdict(config)
    config_dict["num_classes"] = num_classes
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved config to {output_dir / 'config.json'}")


# =============================================================================
# Main Pipeline
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate toy dataset for spiral test")
    parser.add_argument(
        "--name",
        type=str,
        default="doublespiral",
        choices=list(DATASET_GENERATORS.keys()),
        help="Dataset type (default: doublespiral)",
    )
    parser.add_argument(
        "--total_points",
        type=int,
        default=12000,
        help="Total number of points (default: 12000)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="Noise level (default: dataset-specific)",
    )
    parser.add_argument(
        "--data_range",
        type=float,
        default=3.0,
        help="Data range [-range, range] (default: 3.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Run data generation pipeline."""
    args = parse_args()

    # Get default noise for dataset type if not specified
    default_noise = {
        "doublespiral": 0.5,
        "concentric_rings": 0.1,
        "circular_gaussians": 0.3,
        "grid_gaussians": 0.2,
        "crossed_lines": 0.1,
        "half_arcs": 0.1,
    }
    noise = args.noise if args.noise is not None else default_noise.get(args.name, 0.5)

    config = DataConfig(
        name=args.name,
        total_points=args.total_points,
        noise=noise,
        data_range=args.data_range,
        seed=args.seed,
    )

    # Set seed
    np.random.seed(config.seed)

    # Create output directory
    data_dir = get_data_dir()
    output_dir = data_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset type: {config.name}")
    print(f"Total points: {config.total_points}")
    print(f"Noise: {config.noise}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Generate data
    if config.name not in DATASET_GENERATORS:
        raise ValueError(f"Unknown dataset type: {config.name}")

    print(f"\n[Step 1] Generating {config.name} data...")
    generator = DATASET_GENERATORS[config.name]
    points, labels, num_classes = generator(
        total_points=config.total_points,
        noise=config.noise,
        data_range=config.data_range,
        seed=config.seed,
    )

    print(f"Generated {len(points)} points with {num_classes} classes")

    # Save everything
    print("\n[Step 2] Saving data...")
    save_data(points, labels, output_dir)
    save_class_info(num_classes, output_dir)
    save_visualization(points, labels, num_classes, output_dir, config.data_range)
    save_config(config, num_classes, output_dir)

    print("\n" + "=" * 60)
    print("Data generation completed!")
    print(f"Files saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
