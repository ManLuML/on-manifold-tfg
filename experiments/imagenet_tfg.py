"""ImageNet-256 TFG-guided evaluation for all supported models.

This script evaluates diffusion models using CFG + TFG guidance for
ImageNet classification. TFG provides additional gradient-based guidance
from an external classifier on top of CFG, targeting the same class.

Experiment Design:
    - CFG labels = Target ImageNet class (for model's class-conditional generation)
    - TFG targets = Same target ImageNet class (for gradient-based guidance)
    - CFG is always active; TFG is additive on top of CFG
    - Guide model (gradient source) != Eval model (validity metric)

Guidance Modes:
    - tfg: TFG (rho=2.0, mu=0.5, sigma=0.1, iter_steps=4, recur_steps=1)
    - tfg-full: TFG with recurrence (same as tfg but recur_steps=4)
    - dps: DPS ablation within TFG framework (rho=2.0, mu=0, sigma=0)

Guidance Spaces:
    - x: TFG/DDPM heritage — Position correction (no dt scaling)
    - v: Flow Guidance heritage — Velocity modification (dt implicit)
    - v2: Per-NFE velocity modification for Heun (dt implicit)

    Compatibility:
        Heun: x, v, v2
        Euler: x, v
        DDPM/DDIM: x only

Evaluation Metrics:
    - Validity: Top-1 accuracy on DeiT-Small (ImageNetClassificationEvaluator)
    - FID: Against full ImageNet reference stats (degrades gracefully if absent)
    - IS: Inception Score

Usage:
    # DPS with JiT-H/16 (Heun, x-space), with a rho sweep
    uv run python experiments/imagenet_tfg.py --model jit-h-16 \
        --guidance_mode dps --rho_override 2.0

    # TFG with DiT (DDPM, x-space only)
    uv run python experiments/imagenet_tfg.py --model dit \
        --guidance_mode tfg --guidance_space x

    # SiT / PixelFlow
    uv run python experiments/imagenet_tfg.py --model sit --guidance_mode dps
    uv run python experiments/imagenet_tfg.py --model pixelflow --guidance_mode dps

    # Quick test
    uv run python experiments/imagenet_tfg.py --model jit-h-16 \
        --guidance_mode dps --guidance_space x \
        --num_classes 10 --images_per_class 10

Output Structure:
    outputs/imagenet_tfg/{model}/{sampling_method}/{guidance_space}/{guidance_mode}/
    ├── config.json       # Experiment configuration (incl. TFG hyperparams)
    ├── metrics.json      # FID, validity, is_mean, is_std
    └── images/           # Generated images (1000 classes × 50 = 50,000)
"""

import argparse
import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from jit_tfg.evaluation.generation import (
    calculate_fid_from_features,
    calculate_inception_score_from_logits,
    get_reference_stats_path,
)
from jit_tfg.evaluation.generation.inception import InceptionFeatureExtractor
from jit_tfg.evaluation.guidance import ImageNetClassificationEvaluator
from jit_tfg.tfg import TFGConfig, UnifiedSampler
from jit_tfg.tfg.guiders import HuggingFaceClassifierGuider


# =============================================================================
# Constants
# =============================================================================

GUIDE_MODEL = "google/vit-base-patch16-224"


# =============================================================================
# TFG / DPS Presets (from original TFG paper, imagenet_label.sh)
# =============================================================================

TFG_PRESET: dict[str, Any] = {
    "rho": 2.0,
    "mu": 0.5,
    "sigma": 0.1,
    "iter_steps": 4,
    "recur_steps": 1,
    "rho_schedule": "increase",
    "mu_schedule": "increase",
    "sigma_schedule": "decrease",
}

TFG_FULL_PRESET: dict[str, Any] = {
    "rho": 2.0,
    "mu": 0.5,
    "sigma": 0.1,
    "iter_steps": 4,
    "recur_steps": 4,
    "rho_schedule": "increase",
    "mu_schedule": "increase",
    "sigma_schedule": "decrease",
}

DPS_PRESET: dict[str, Any] = {
    "rho": 2.0,
    "mu": 0.0,
    "sigma": 0.0,
    "iter_steps": 1,
    "recur_steps": 1,
    "rho_schedule": "increase",
    "mu_schedule": "increase",
    "sigma_schedule": "decrease",
}

GUIDANCE_PRESETS: dict[str, dict[str, Any]] = {
    "tfg": TFG_PRESET,
    "tfg-full": TFG_FULL_PRESET,
    "dps": DPS_PRESET,
}


# =============================================================================
# NFE / Step Conversion
# =============================================================================


def nfe_to_steps(nfe: int, sampling_method: str) -> int:
    """Convert NFE (Number of Function Evaluations) to sampling steps.

    For most solvers (DDPM, DDIM, Euler), NFE equals the number of steps.
    For Heun's method, each step requires 2 function evaluations, so the
    number of steps is halved.

    Args:
        nfe: Number of Function Evaluations.
        sampling_method: Sampling method name.

    Returns:
        Number of sampling steps.
    """
    if sampling_method == "heun":
        return nfe // 2
    return nfe


# =============================================================================
# Model Configurations
# =============================================================================


def load_model_configs() -> dict[str, dict[str, Any]]:
    """Load model configurations from JSON file.

    Returns:
        Dictionary mapping model names to their configurations.
    """
    config_path = Path(__file__).parent / "model_configs.json"
    with open(config_path) as f:
        return json.load(f)


MODEL_CONFIGS: dict[str, dict[str, Any]] = load_model_configs()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvalResult:
    """Evaluation result for ImageNet TFG-guided generation."""

    model: str
    guidance_mode: str
    guidance_space: str
    validity: float
    fid: float
    is_mean: float
    is_std: float
    num_images: int
    validity_per_class: dict | None = None


# =============================================================================
# Model Loading Functions
# =============================================================================


def load_jit_model(config: dict[str, Any], device: str):
    """Load JiT denoiser.

    Args:
        config: Model configuration dictionary.
        device: Device for model.

    Returns:
        Loaded denoiser model.
    """
    from jit_tfg.models.jit.utils.checkpoint import load_checkpoint_for_inference

    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"JiT checkpoint not found: {checkpoint_path}")

    denoiser, _ = load_checkpoint_for_inference(
        checkpoint_path=checkpoint_path,
        device=device,
        use_ema=True,
    )
    return denoiser


def load_dit_model(config: dict[str, Any], device: str, num_steps: int):
    """Load DiT denoiser.

    Args:
        config: Model configuration dictionary.
        device: Device for model.
        num_steps: Number of sampling steps (converted from NFE).

    Returns:
        Loaded DiTDenoiser instance.
    """
    from jit_tfg.models.dit.denoiser import load_dit_denoiser

    denoiser = load_dit_denoiser(
        checkpoint_path=config.get("checkpoint"),
        from_pretrained=config.get("from_pretrained"),
        device=device,
        cfg_scale=config["cfg_scale"],
        num_sampling_steps=num_steps,
        vae_type=config.get("vae_type", "ema"),
    )
    return denoiser


def load_sit_model(config: dict[str, Any], device: str, num_steps: int):
    """Load SiT denoiser.

    Args:
        config: Model configuration dictionary.
        device: Device for model.
        num_steps: Number of sampling steps (converted from NFE).

    Returns:
        Loaded SiT denoiser instance.
    """
    from jit_tfg.models.sit.denoiser import load_sit_denoiser

    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SiT checkpoint not found: {checkpoint_path}\n"
            "Download from HuggingFace:\n"
            "  huggingface-cli download nyu-visionx/SiT-collections --local-dir checkpoints/sit"
        )

    denoiser = load_sit_denoiser(
        checkpoint_path=checkpoint_path,
        device=device,
        cfg_scale=config["cfg_scale"],
        num_sampling_steps=num_steps,
        vae_type=config.get("vae_type", "ema"),
    )
    return denoiser


def load_pixelflow_model(config: dict[str, Any], device: str, num_steps: int):
    """Load PixelFlow denoiser.

    Args:
        config: Model configuration dictionary.
        device: Device for model.
        num_steps: Number of sampling steps (converted from NFE).

    Returns:
        Loaded PixelFlow denoiser instance.
    """
    from jit_tfg.models.pixelflow.denoiser import load_pixelflow_denoiser

    checkpoint_path = config.get("checkpoint")
    from_pretrained = config.get("from_pretrained")

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"PixelFlow checkpoint not found: {checkpoint_path}\n"
                "Download from HuggingFace:\n"
                "  huggingface-cli download ShoufaChen/PixelFlow-Class2Image --local-dir checkpoints/pixelflow"
            )
        checkpoint_path = str(checkpoint_path)

    denoiser = load_pixelflow_denoiser(
        checkpoint_path=checkpoint_path,
        from_pretrained=from_pretrained,
        device=device,
        cfg_scale=config["cfg_scale"],
        num_sampling_steps=num_steps,
    )
    return denoiser


def load_model(model_name: str, device: str, num_steps: int):
    """Load model based on name.

    Args:
        model_name: Name of the model to load.
        device: Device for model.
        num_steps: Number of sampling steps (converted from NFE).

    Returns:
        Loaded denoiser model instance.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    model_type = config["model_type"]

    print(f"Loading {model_type} model: {model_name}...")

    if model_type == "JiT":
        return load_jit_model(config, device)
    elif model_type == "DiT":
        return load_dit_model(config, device, num_steps)
    elif model_type == "SiT":
        return load_sit_model(config, device, num_steps)
    elif model_type == "PixelFlow":
        return load_pixelflow_model(config, device, num_steps)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Image Utilities
# =============================================================================


def save_single_image(
    img: torch.Tensor,
    label: int,
    img_idx: int,
    output_dir: Path,
    prefix: str = "img",
) -> str:
    """Save a single generated image to disk.

    Args:
        img: Single image tensor of shape (C, H, W) in [-1, 1] range.
        label: ImageNet class label.
        img_idx: Index within the class (for filename).
        output_dir: Output directory.
        prefix: Filename prefix.

    Returns:
        Path to saved image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    img = (img + 1) / 2
    img = img.detach().cpu()

    gen_img = np.round(np.clip(img.numpy().transpose([1, 2, 0]) * 255, 0, 255))
    gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
    path = output_dir / f"{prefix}_class{label:04d}_{img_idx:04d}.png"
    cv2.imwrite(str(path), gen_img)
    return str(path)


def count_existing_images(output_dir: Path, prefix: str = "img") -> dict[int, int]:
    """Count existing images per class in output directory.

    Args:
        output_dir: Directory to scan.
        prefix: Filename prefix to match.

    Returns:
        Dict mapping class_id -> count of existing images.
    """
    import re

    counts: dict[int, int] = {}
    if not output_dir.exists():
        return counts

    pattern = re.compile(rf"^{prefix}_class(\d+)_(\d+)\.png$")

    for filepath in output_dir.iterdir():
        if not filepath.is_file():
            continue
        match = pattern.match(filepath.name)
        if match:
            class_id = int(match.group(1))
            counts[class_id] = counts.get(class_id, 0) + 1

    return counts


# =============================================================================
# Generation Functions
# =============================================================================


def generate_samples(
    sampler: UnifiedSampler,
    guider: HuggingFaceClassifierGuider,
    target_classes: list[int],
    images_per_class: int,
    batch_size: int,
    device: str,
    cfg_scale: float,
    num_steps: int,
    output_dir: Path,
    prefix: str = "img",
    show_progress: bool = True,
) -> int:
    """Generate ImageNet samples with CFG + TFG guidance.

    Key design: CFG label = TFG target = same ImageNet class.
    Both CFG and TFG point to the same class, so TFG acts as
    additional gradient-based guidance on top of CFG.

    Args:
        sampler: UnifiedSampler instance (with TFG config).
        guider: ImageNet classifier guider for TFG gradients.
        target_classes: List of ImageNet class indices to generate.
        images_per_class: Number of images per class.
        batch_size: Batch size for generation.
        device: Device for generation.
        cfg_scale: CFG scale.
        num_steps: Number of sampling steps.
        output_dir: Directory to save images.
        prefix: Filename prefix for saved images.
        show_progress: Whether to show progress bar.

    Returns:
        Total number of images generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = len(target_classes) * images_per_class

    existing_counts = count_existing_images(output_dir, prefix)
    total_existing = sum(min(existing_counts.get(cls, 0), images_per_class) for cls in target_classes)

    if total_existing > 0:
        print(f"  Found {total_existing}/{num_samples} existing images, resuming...")

    generation_tasks: list[tuple[int, int]] = []
    for cls in target_classes:
        existing = existing_counts.get(cls, 0)
        for idx in range(existing, images_per_class):
            generation_tasks.append((cls, idx))

    if len(generation_tasks) == 0:
        print("  All images already generated.")
        return num_samples

    generation_tasks.sort(key=lambda x: (x[0], x[1]))

    iterator = range(0, len(generation_tasks), batch_size)
    total_batches = (len(generation_tasks) + batch_size - 1) // batch_size

    if show_progress:
        desc = f"Generating ({total_existing}/{num_samples} done)"
        iterator = tqdm(iterator, desc=desc, total=total_batches)

    for batch_start in iterator:
        batch_tasks = generation_tasks[batch_start : batch_start + batch_size]
        batch_labels = [task[0] for task in batch_tasks]
        batch_indices = [task[1] for task in batch_tasks]

        cfg_labels = torch.tensor(batch_labels, device=device, dtype=torch.long)
        tfg_targets = torch.tensor(batch_labels, device=device, dtype=torch.long)

        with torch.amp.autocast(device, dtype=torch.bfloat16):
            images = sampler.generate(
                guidance=guider,
                tfg_targets=tfg_targets,
                cfg_labels=cfg_labels,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
                show_progress=False,
            )

        for img, label, idx in zip(images, batch_labels, batch_indices):
            save_single_image(img.float(), label, idx, output_dir, prefix)

        del images
        torch.cuda.empty_cache()

    if show_progress:
        print("  Generation complete.")

    return num_samples


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_generated_images(
    img_dir: Path,
    target_classes: list[int],
    images_per_class: int,
    batch_size: int,
    device: str,
    img_size: int,
    skip_fid: bool = False,
) -> EvalResult:
    """Evaluate generated images for ImageNet classification.

    Uses the same evaluation pipeline as imagenet_no_guidance.py:
    DeiT-Small for validity, Inception-v3 for FID and IS.

    FID degrades gracefully: if the reference statistics file is absent,
    FID is reported as 0.0 with a warning instead of crashing.

    Args:
        img_dir: Directory containing generated images.
        target_classes: List of target class indices.
        images_per_class: Number of images per class.
        batch_size: Batch size for evaluation.
        device: Device for evaluation.
        img_size: Image size for FID reference stats.
        skip_fid: Whether to skip FID/IS calculation.

    Returns:
        EvalResult with metrics.
    """
    num_samples = len(target_classes) * images_per_class

    print("  Loading validity evaluator (DeiT-Small)...")
    validity_evaluator = ImageNetClassificationEvaluator(device=device)

    print("  Computing validity...")
    validity_result = validity_evaluator.compute_validity_from_folder(
        folder_path=img_dir,
        target_classes=target_classes,
        images_per_class=images_per_class,
        batch_size=batch_size,
        prefix="img",
        show_progress=True,
    )

    fid = 0.0
    is_mean = 0.0
    is_std = 0.0

    if not skip_fid:
        print("  Extracting Inception features...")
        extractor = InceptionFeatureExtractor(device=device, batch_size=batch_size)
        inception_result = extractor.extract_from_folder(img_dir, show_progress=True)
        features = inception_result["features"]
        logits = inception_result["logits"]

        print("  Computing FID (global)...")
        try:
            stats_path = get_reference_stats_path(img_size=img_size)
            fid = calculate_fid_from_features(features, stats_path)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print("  WARNING: FID reference stats absent; reporting FID=0.0.")

        print("  Computing Inception Score...")
        is_mean, is_std = calculate_inception_score_from_logits(logits)

    return EvalResult(
        model="",
        guidance_mode="",
        guidance_space="",
        validity=validity_result["validity"],
        fid=fid,
        is_mean=is_mean,
        is_std=is_std,
        num_images=num_samples,
        validity_per_class=validity_result.get("validity_per_class"),
    )


# =============================================================================
# Validation
# =============================================================================


def validate_sampling_method(model_type: str, sampling_method: str) -> None:
    """Validate sampling method for the given model type.

    Args:
        model_type: Type of the model (JiT, DiT, SiT, PixelFlow).
        sampling_method: Sampling method to validate.

    Raises:
        AssertionError: If the sampling method is not compatible with the model.
    """
    if model_type == "DiT":
        assert sampling_method in ("ddim", "ddpm"), (
            f"DiT only supports 'ddim' or 'ddpm' sampling method, got '{sampling_method}'. "
            "DiT is a DDPM-based model and requires DDIM or DDPM sampling."
        )
    else:
        assert sampling_method not in ("ddim", "ddpm"), (
            f"{model_type} does not support '{sampling_method}' sampling method. "
            f"Use 'euler' or 'heun' instead. "
            "DDIM/DDPM is only for DDPM-based models like DiT."
        )


def validate_guidance_space(sampling_method: str, guidance_space: str) -> None:
    """Validate guidance space compatibility with sampling method.

    Compatibility matrix:
        Heun:      x, v, v2
        Euler:     x, v
        DDPM/DDIM: x

    Args:
        sampling_method: Sampling method name.
        guidance_space: Guidance space name.

    Raises:
        ValueError: If the guidance space is not compatible with the solver.
    """
    if sampling_method in ("ddpm", "ddim"):
        if guidance_space != "x":
            raise ValueError(
                f"DDPM/DDIM sampling only supports guidance_space='x', "
                f"got '{guidance_space}'. "
                "DDPM-based models use discrete timesteps incompatible with "
                "velocity-space guidance (v, v2)."
            )
    elif sampling_method == "euler":
        if guidance_space not in ("x", "v"):
            raise ValueError(
                f"Euler sampling supports guidance_space='x' or 'v', "
                f"got '{guidance_space}'. "
                "guidance_space='v2' requires Heun solver (per-NFE guidance)."
            )
    elif sampling_method == "heun":
        if guidance_space not in ("x", "v", "v2"):
            raise ValueError(f"Unknown guidance_space '{guidance_space}'. Heun supports: 'x', 'v', 'v2'.")


# =============================================================================
# Argument Parsing
# =============================================================================


def get_args_parser() -> argparse.ArgumentParser:
    """Create argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        "ImageNet TFG-guided Evaluation",
        add_help=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to evaluate",
    )
    parser.add_argument(
        "--guidance_mode",
        type=str,
        required=True,
        choices=list(GUIDANCE_PRESETS.keys()),
        help="Guidance mode: 'tfg', 'tfg-full' (tfg + recur_steps=4), or 'dps' (rho-only ablation)",
    )
    parser.add_argument(
        "--guidance_space",
        type=str,
        default="x",
        choices=["x", "v", "v2"],
        help="Guidance space. Heun: x/v/v2, Euler: x/v, DDPM/DDIM: x only.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/imagenet_tfg",
        help="Output directory base",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of classes to evaluate (default: 1000 = all ImageNet)",
    )
    parser.add_argument(
        "--images_per_class",
        type=int,
        default=50,
        help="Images per class (default: 50, total 50K for 1000 classes)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force re-run all evaluations (ignore cached results)",
    )
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip FID/IS calculation (validity only)",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default=None,
        choices=["euler", "heun", "ddim", "ddpm"],
        help="Sampling method (default: use model_configs.json). "
        "DiT: ddim or ddpm. JiT/SiT/PixelFlow: euler or heun only.",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        default=None,
        help="Number of Function Evaluations (default: use model_configs.json). "
        "For Heun solver, steps = nfe // 2. For others, steps = nfe.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override the model name used in the output directory path. "
        "If not provided, uses --model value (e.g., 'sit-nfe100').",
    )

    # rho/mu sweep and ablation options
    parser.add_argument(
        "--rho_override",
        type=float,
        default=None,
        help="Override rho value (for rho sweeps / DPS calibration experiments).",
    )
    parser.add_argument(
        "--mu_override",
        type=float,
        default=None,
        help="Override mu value (for mu sweeps / calibration experiments).",
    )

    return parser


# =============================================================================
# Main
# =============================================================================


def main(args: argparse.Namespace) -> None:
    """Run ImageNet TFG-guided evaluation.

    Args:
        args: Parsed command line arguments.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = args.model
    output_name = args.model_name or model_name
    config = MODEL_CONFIGS[model_name]
    model_type = config["model_type"]

    sampling_method = args.sampling_method or config["sampling_method"]
    validate_sampling_method(model_type, sampling_method)

    guidance_mode = args.guidance_mode
    guidance_space = args.guidance_space
    validate_guidance_space(sampling_method, guidance_space)

    nfe = args.nfe or config["nfe"]
    num_steps = nfe_to_steps(nfe, sampling_method)

    # Resolve effective rho/mu (apply overrides on top of the preset).
    guidance_params = GUIDANCE_PRESETS[guidance_mode]
    effective_rho = args.rho_override if args.rho_override is not None else guidance_params["rho"]
    effective_mu = args.mu_override if args.mu_override is not None else guidance_params["mu"]

    # Output: model / sampling_method / guidance_space / guidance_mode[-suffix]
    output_guidance_mode = guidance_mode
    if guidance_mode == "dps":
        output_guidance_mode = f"dps-rho{effective_rho:.2f}"
        if args.mu_override is not None:
            output_guidance_mode = f"{output_guidance_mode}_mu{args.mu_override:.2f}"
    elif args.rho_override is not None or args.mu_override is not None:
        parts = []
        if args.rho_override is not None:
            parts.append(f"rho{args.rho_override:.2f}")
        if args.mu_override is not None:
            parts.append(f"mu{args.mu_override:.2f}")
        output_guidance_mode = f"{guidance_mode}-{'_'.join(parts)}"

    output_dir = Path(args.output_dir) / output_name / sampling_method / guidance_space / output_guidance_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    target_classes = list(range(args.num_classes))
    num_samples = args.num_classes * args.images_per_class

    print("=" * 70)
    print("ImageNet TFG-guided Evaluation")
    print("=" * 70)
    print(f"Model: {model_name} ({model_type})")
    print(f"Guidance mode: {guidance_mode}")
    print(f"Guidance space: {guidance_space}")
    print(f"Output: {output_dir}")
    print(f"Classes: {args.num_classes}")
    print(f"Images/class: {args.images_per_class}")
    print(f"Total samples: {num_samples}")
    print(f"Image size: {config['img_size']}")
    print(f"Sampling method: {sampling_method}")
    print(f"CFG scale: {config['cfg_scale']}")
    print(f"NFE: {nfe} (num_steps: {num_steps})")
    print(f"Guide model: {GUIDE_MODEL}")
    print(
        f"TFG params: rho={effective_rho}, mu={effective_mu}, "
        f"sigma={guidance_params['sigma']}, iter_steps={guidance_params['iter_steps']}, "
        f"recur_steps={guidance_params['recur_steps']}"
    )
    print("=" * 70)

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not args.force_rerun:
        print(f"\nLoading existing results from {metrics_path}")
        with open(metrics_path) as f:
            result = json.load(f)
        print("\nResults:")
        print(f"  Validity: {result['validity']:.4f}")
        print(f"  FID: {result['fid']:.2f}")
        print(f"  IS: {result['is_mean']:.2f} ± {result['is_std']:.2f}")
        return

    tfg_config = TFGConfig(
        rho=effective_rho,
        mu=effective_mu,
        sigma=guidance_params["sigma"],
        iter_steps=guidance_params["iter_steps"],
        recur_steps=guidance_params["recur_steps"],
        rho_schedule=guidance_params["rho_schedule"],
        mu_schedule=guidance_params["mu_schedule"],
        sigma_schedule=guidance_params["sigma_schedule"],
        device=args.device,
        seed=args.seed,
    )

    if args.rho_override is not None or args.mu_override is not None:
        print(f"  [Override] rho={effective_rho}, mu={effective_mu}")

    denoiser = load_model(model_name, args.device, num_steps)

    print("\nCreating UnifiedSampler...")
    sampler = UnifiedSampler(
        model_type=model_type,
        denoiser=denoiser,
        tfg_config=tfg_config,
        sampling_method=sampling_method,
        guidance_space=guidance_space,
    )

    print(f"Loading guide model: {GUIDE_MODEL}...")
    guider = HuggingFaceClassifierGuider(
        model_name=GUIDE_MODEL,
        device=args.device,
    )

    exp_config = {
        "model": model_name,
        "model_type": model_type,
        "checkpoint": config.get("checkpoint") or config.get("from_pretrained"),
        "img_size": config["img_size"],
        "sampling_method": sampling_method,
        "guidance_mode": guidance_mode,
        "guidance_space": guidance_space,
        "nfe": nfe,
        "num_steps": num_steps,
        "cfg_scale": config["cfg_scale"],
        "tfg_params": guidance_params,
        "effective_rho": effective_rho,
        "effective_mu": effective_mu,
        "guide_model": GUIDE_MODEL,
        "num_classes": args.num_classes,
        "images_per_class": args.images_per_class,
        "num_samples": num_samples,
        "seed": args.seed,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    print("\n" + "=" * 70)
    print("Generating samples...")
    print("=" * 70)

    img_dir = output_dir / "images"
    generate_samples(
        sampler=sampler,
        guider=guider,
        target_classes=target_classes,
        images_per_class=args.images_per_class,
        batch_size=args.batch_size,
        device=args.device,
        cfg_scale=config["cfg_scale"],
        num_steps=num_steps,
        output_dir=img_dir,
    )

    print("\n" + "=" * 70)
    print("Evaluating generated images...")
    print("=" * 70)

    result = evaluate_generated_images(
        img_dir=img_dir,
        target_classes=target_classes,
        images_per_class=args.images_per_class,
        batch_size=args.batch_size,
        device=args.device,
        img_size=config["img_size"],
        skip_fid=args.skip_fid,
    )
    result.model = model_name
    result.guidance_mode = guidance_mode
    result.guidance_space = guidance_space

    result_dict = asdict(result)
    with open(metrics_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults for {model_name} ({guidance_mode}, {guidance_space}-space):")
    print(f"  Validity: {result.validity:.4f}")
    print(f"  FID: {result.fid:.2f}")
    print(f"  IS: {result.is_mean:.2f} ± {result.is_std:.2f}")
    print(f"\nResults saved to: {metrics_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
