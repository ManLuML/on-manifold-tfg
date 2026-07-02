"""Fine-grained bird TFG-guided evaluation for all supported models.

This script evaluates diffusion models using CFG + TFG guidance for
fine-grained bird species classification. Each fine-grained bird class
is guided using TFG with a bird species classifier, while CFG uses the
ImageNet parent class.

Experiment Design:
    - CFG labels = ImageNet parent class (for model's class-conditional generation)
    - TFG targets = Fine-grained bird species class (for gradient-based guidance)
    - Guide model (gradient source) != Eval model (validity metric)

Guidance Modes:
    - tfg: TFG (rho=0.5, mu=0.5, sigma=0.01, iter_steps=4, recur_steps=1)
    - tfg-full: TFG with recurrence (same as tfg but recur_steps=4)
    - dps: DPS ablation within TFG framework (rho=0.5, mu=0, sigma=0)

Guidance Spaces:
    - x: TFG/DDPM heritage — Position correction (no dt scaling)
    - v: Flow Guidance heritage — Velocity modification (dt implicit)
    - v2: Per-NFE velocity modification for Heun (dt implicit)

    Compatibility:
        Heun: x, v, v2
        Euler: x, v
        DDPM/DDIM: x only

Evaluation Metrics:
    - Validity: Top-1 accuracy on fine-grained eval model (bird-species-classifier)
    - FID (Parent): Against ImageNet parent classes
    - FID (Child): Against bird species dataset
    - IS: Inception Score

Usage:
    # TFG with JiT-H/16 (Heun, x-space)
    uv run python experiments/finegrained_bird_tfg.py --model jit-h-16 \
        --guidance_mode tfg --guidance_space x

    # DPS ablation with Euler, v-space
    uv run python experiments/finegrained_bird_tfg.py --model jit-h-16 \
        --guidance_mode dps --guidance_space v --sampling_method euler

    # TFG with DiT (DDIM, x-space only)
    uv run python experiments/finegrained_bird_tfg.py --model dit \
        --guidance_mode tfg --guidance_space x

    # Quick test
    uv run python experiments/finegrained_bird_tfg.py --model jit-h-16 \
        --guidance_mode tfg --guidance_space x --images_per_class 10

Output Structure:
    outputs/finegrained_bird_tfg/{model}/{sampling_method}/{guidance_space}/{guidance_mode}/
    ├── config.json       # Experiment configuration (incl. TFG hyperparams)
    ├── metrics.json      # validity, fid_parent, fid_child, is_mean, is_std
    └── images/           # Generated images (143 classes × 64 = 9,152)
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
    get_finegrained_stats_path,
)
from jit_tfg.evaluation.generation.inception import InceptionFeatureExtractor
from jit_tfg.evaluation.guidance import FinegrainedBirdEvaluator
from jit_tfg.tfg import TFGConfig, UnifiedSampler
from jit_tfg.tfg.guiders import HuggingFaceClassifierGuider


# =============================================================================
# Constants
# =============================================================================

GUIDE_MODEL = "dennisjooo/Birds-Classifier-EfficientNetB2"
EVAL_MODEL = "chriamue/bird-species-classifier"
MAPPING_FILE = Path(__file__).parent / "finegrained_bird_mapping.json"


# =============================================================================
# TFG / DPS Presets (from original TFG paper, bird_finegrained.sh)
# =============================================================================

TFG_PRESET: dict[str, Any] = {
    "rho": 0.5,
    "mu": 0.5,
    "sigma": 0.01,
    "iter_steps": 4,
    "recur_steps": 1,
    "rho_schedule": "increase",
    "mu_schedule": "increase",
    "sigma_schedule": "decrease",
}

TFG_FULL_PRESET: dict[str, Any] = {
    "rho": 0.5,
    "mu": 0.5,
    "sigma": 0.01,
    "iter_steps": 4,
    "recur_steps": 4,
    "rho_schedule": "increase",
    "mu_schedule": "increase",
    "sigma_schedule": "decrease",
}

DPS_PRESET: dict[str, Any] = {
    "rho": 0.5,
    "mu": 0.0,
    "sigma": 0.0,
    "iter_steps": 1,
    "recur_steps": 1,
    "rho_schedule": "increase",
    "mu_schedule": "increase",
    "sigma_schedule": "decrease",
}

# LGD (Song et al., ICML 2023): MC-averaged loss gradient, no direct gradient.
# In the TFG-framework, LGD = rho=0 + mu>0 + iter>1 + recur=1.
# Sweep variable at experiment time: --mu_override.
LGD_PRESET: dict[str, Any] = {
    "rho": 0.0,
    "mu": 0.5,                 # placeholder; OVERRIDDEN by --mu_override
    "sigma": 0.01,             # nonzero smoothing; sigma=0 disables MC noise in UnifiedSampler
    "iter_steps": 4,           # mean-guidance refinement iterations; MC count is TFGConfig.eps_bsz
    "recur_steps": 1,
    "rho_schedule": "increase",
    "mu_schedule": "increase",
    "sigma_schedule": "decrease",
}

# FreeDoM (Yu et al., ICCV 2023): direct gradient + time-travel recurrence.
# In the TFG-framework, FreeDoM = rho>0 + mu=0 + iter=1 + recur>1.
# Sweep variable at experiment time: --rho_override.
FREEDOM_PRESET: dict[str, Any] = {
    "rho": 0.5,                # placeholder; OVERRIDDEN by --rho_override
    "mu": 0.0,
    "sigma": 0.0,
    "iter_steps": 1,
    "recur_steps": 4,          # FreeDoM original 2-5 range
    "rho_schedule": "increase",
    "mu_schedule": "increase",
    "sigma_schedule": "decrease",
}

GUIDANCE_PRESETS: dict[str, dict[str, Any]] = {
    "tfg": TFG_PRESET,
    "tfg-full": TFG_FULL_PRESET,
    "dps": DPS_PRESET,
    "lgd": LGD_PRESET,
    "freedom": FREEDOM_PRESET,
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
    """Evaluation result for fine-grained bird TFG-guided generation."""

    model: str
    guidance_mode: str
    guidance_space: str
    validity: float
    fid_parent: float
    fid_child: float
    is_mean: float
    is_std: float
    num_images: int
    num_classes: int
    validity_per_class: dict | None = None


# =============================================================================
# Model Loading Functions
# =============================================================================


def load_class_mapping() -> dict:
    """Load fine-grained to ImageNet parent class mapping.

    Returns:
        Dictionary with mapping information.
    """
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {MAPPING_FILE}\n"
            "Please ensure finegrained_bird_mapping.json exists in experiments/."
        )

    with open(MAPPING_FILE) as f:
        return json.load(f)


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
            "  huggingface-cli download nyu-visionx/SiT-collections "
            "--local-dir checkpoints/sit"
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

    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"PixelFlow checkpoint not found: {checkpoint_path}\nPlease download the checkpoint and update the path."
        )

    denoiser = load_pixelflow_denoiser(
        checkpoint_path=checkpoint_path,
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
        label: Fine-grained class label.
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


def generate_finegrained_samples(
    sampler: UnifiedSampler,
    guider: HuggingFaceClassifierGuider,
    finegrained_classes: list[int],
    finegrained_to_imagenet: dict[str, dict],
    images_per_class: int,
    batch_size: int,
    device: str,
    cfg_scale: float,
    num_steps: int,
    output_dir: Path,
    prefix: str = "img",
    show_progress: bool = True,
) -> int:
    """Generate fine-grained bird samples with CFG + TFG guidance.

    Key design:
    - CFG labels = ImageNet parent class (for model's class-conditional generation)
    - TFG targets = Fine-grained class (for guider's gradient computation)
    - File names use fine-grained class index (for evaluation)

    Args:
        sampler: UnifiedSampler instance (with TFG config).
        guider: Fine-grained bird species guider for TFG gradients.
        finegrained_classes: List of fine-grained class indices to generate.
        finegrained_to_imagenet: Mapping from fine-grained to ImageNet classes.
        images_per_class: Number of images per fine-grained class.
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

    existing_counts = count_existing_images(output_dir, prefix)
    total_existing = sum(min(existing_counts.get(cls, 0), images_per_class) for cls in finegrained_classes)
    total_images = len(finegrained_classes) * images_per_class

    if total_existing > 0:
        print(f"  Found {total_existing}/{total_images} existing images, resuming...")

    generation_tasks: list[tuple[int, int, int]] = []
    for fg_cls in finegrained_classes:
        existing = existing_counts.get(fg_cls, 0)
        imagenet_cls = finegrained_to_imagenet[str(fg_cls)]["imagenet_class"]
        for idx in range(existing, images_per_class):
            generation_tasks.append((fg_cls, imagenet_cls, idx))

    if len(generation_tasks) == 0:
        print("  All images already generated.")
        return total_images

    generation_tasks.sort(key=lambda x: (x[0], x[2]))

    iterator = range(0, len(generation_tasks), batch_size)
    total_batches = (len(generation_tasks) + batch_size - 1) // batch_size

    if show_progress:
        desc = f"Generating ({total_existing}/{total_images} done)"
        iterator = tqdm(iterator, desc=desc, total=total_batches)

    for batch_start in iterator:
        batch_tasks = generation_tasks[batch_start : batch_start + batch_size]
        batch_fg_labels = [task[0] for task in batch_tasks]
        batch_in_labels = [task[1] for task in batch_tasks]
        batch_indices = [task[2] for task in batch_tasks]

        cfg_labels = torch.tensor(batch_in_labels, device=device, dtype=torch.long)
        tfg_targets = torch.tensor(batch_fg_labels, device=device, dtype=torch.long)

        with torch.amp.autocast(device, dtype=torch.bfloat16):
            images = sampler.generate(
                guidance=guider,
                tfg_targets=tfg_targets,
                cfg_labels=cfg_labels,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
                show_progress=False,
            )

        for img, fg_label, idx in zip(images, batch_fg_labels, batch_indices):
            save_single_image(img.float(), fg_label, idx, output_dir, prefix)

        del images
        torch.cuda.empty_cache()

    if show_progress:
        print("  Generation complete.")

    return total_images


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_generated_images(
    img_dir: Path,
    finegrained_classes: list[int],
    images_per_class: int,
    batch_size: int,
    device: str,
    skip_fid: bool = False,
) -> EvalResult:
    """Evaluate generated images for fine-grained bird classification.

    Args:
        img_dir: Directory containing generated images.
        finegrained_classes: List of fine-grained class indices.
        images_per_class: Number of images per class.
        batch_size: Batch size for evaluation.
        device: Device for evaluation.
        skip_fid: Whether to skip FID/IS calculation.

    Returns:
        EvalResult with all metrics.
    """
    num_images = len(finegrained_classes) * images_per_class

    print(f"  Loading fine-grained evaluator: {EVAL_MODEL}...")
    validity_evaluator = FinegrainedBirdEvaluator(
        model_name=EVAL_MODEL,
        device=device,
    )

    print("  Computing fine-grained validity...")
    validity_result = validity_evaluator.compute_validity_from_folder(
        folder_path=img_dir,
        target_classes=finegrained_classes,
        images_per_class=images_per_class,
        batch_size=batch_size,
        prefix="img",
        show_progress=True,
    )

    fid_parent = 0.0
    fid_child = 0.0
    is_mean = 0.0
    is_std = 0.0

    if not skip_fid:
        print("  Extracting Inception features...")
        extractor = InceptionFeatureExtractor(device=device, batch_size=batch_size)
        inception_result = extractor.extract_from_folder(img_dir, show_progress=True)
        features = inception_result["features"]
        logits = inception_result["logits"]

        print("  Computing FID (parent/ImageNet)...")
        try:
            parent_stats_path = get_finegrained_stats_path(stats_type="parent")
            fid_parent = calculate_fid_from_features(features, parent_stats_path)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

        print("  Computing FID (child/bird species)...")
        try:
            child_stats_path = get_finegrained_stats_path(stats_type="child")
            fid_child = calculate_fid_from_features(features, child_stats_path)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

        print("  Computing Inception Score...")
        is_mean, is_std = calculate_inception_score_from_logits(logits)

    return EvalResult(
        model="",
        guidance_mode="",
        guidance_space="",
        validity=validity_result["validity"],
        fid_parent=fid_parent,
        fid_child=fid_child,
        is_mean=is_mean,
        is_std=is_std,
        num_images=num_images,
        num_classes=len(finegrained_classes),
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
        "Fine-grained Bird TFG-guided Evaluation",
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
        help=(
            "Guidance mode. 'dps': rho-only minimal form. "
            "'lgd': mu-only MC smoothing (rho=0, iter=4); sweep with --mu_override. "
            "'freedom': rho + recurrence (mu=0, recur=4); sweep with --rho_override. "
            "'tfg': combined rho+mu+iter (TFG paper bird recipe). "
            "'tfg-full': tfg + recur_steps=4 (16x cost; not used in the paper)."
        ),
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
        default="outputs/finegrained_bird_tfg",
        help="Output directory base",
    )
    parser.add_argument(
        "--images_per_class",
        type=int,
        default=64,
        help="Number of images per fine-grained class (default: 64)",
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

    # v-space calibration and ablation options
    parser.add_argument(
        "--rho_override",
        type=float,
        default=None,
        help="Override rho value (for v-space calibration experiments).",
    )
    parser.add_argument(
        "--mu_override",
        type=float,
        default=None,
        help="Override mu value (for v-space calibration experiments).",
    )
    parser.add_argument(
        "--rescale_mode",
        type=str,
        default="clip",
        choices=["clip", "original"],
        help="Gradient rescaling mode. 'clip': clip image gradients (default). "
        "'original': no clipping, matching original TFG implementation.",
    )
    parser.add_argument(
        "--lambda_mode",
        type=str,
        default="auto",
        choices=["flow_guidance", "identity", "auto"],
        help="Lambda_t mode for v-space. 'auto' (default): pred_target-based "
        "(x-pred->identity, v-pred->flow_guidance). "
        "'flow_guidance': (1-t)/t. 'identity': lambda=1.",
    )
    parser.add_argument(
        "--auto_calibrate",
        action="store_true",
        help="Automatically calibrate rho/mu for v-space to match x-space "
        "cumulative guidance magnitude. Only applies when guidance_space "
        "is 'v' or 'v2'. Overrides --rho_override and --mu_override.",
    )

    return parser


# =============================================================================
# Main
# =============================================================================


def main(args: argparse.Namespace) -> None:
    """Run fine-grained bird TFG-guided evaluation.

    Args:
        args: Parsed command line arguments.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mapping = load_class_mapping()
    finegrained_to_imagenet = mapping["finegrained_to_imagenet"]
    finegrained_classes = [int(k) for k in finegrained_to_imagenet.keys()]
    finegrained_classes.sort()

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

    # Output: model / sampling_method / guidance_space / guidance_mode[-suffix]
    guidance_params = GUIDANCE_PRESETS[guidance_mode]
    output_guidance_mode = guidance_mode
    if guidance_mode == "dps":
        rho_val = args.rho_override if args.rho_override is not None else guidance_params["rho"]
        output_guidance_mode = f"dps-rho{rho_val:.2f}"
        if args.auto_calibrate and guidance_space in ("v", "v2"):
            output_guidance_mode = f"{output_guidance_mode}-calibrated"
        elif args.mu_override is not None:
            output_guidance_mode = f"{output_guidance_mode}_mu{args.mu_override:.2f}"
    elif args.auto_calibrate and guidance_space in ("v", "v2"):
        output_guidance_mode = f"{guidance_mode}-calibrated"
    elif args.rho_override is not None or args.mu_override is not None:
        parts = []
        if args.rho_override is not None:
            parts.append(f"rho{args.rho_override:.2f}")
        if args.mu_override is not None:
            parts.append(f"mu{args.mu_override:.2f}")
        output_guidance_mode = f"{guidance_mode}-{'_'.join(parts)}"
    if args.lambda_mode != "auto":
        output_guidance_mode = f"{output_guidance_mode}-lambda_{args.lambda_mode}"
    output_dir = Path(args.output_dir) / output_name / sampling_method / guidance_space / output_guidance_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(finegrained_classes)
    num_images = num_classes * args.images_per_class

    print("=" * 70)
    print("Fine-grained Bird TFG-guided Evaluation")
    print("=" * 70)
    print(f"Model: {model_name} ({model_type})")
    print(f"Guidance mode: {guidance_mode}")
    print(f"Guidance space: {guidance_space}")
    print(f"Output: {output_dir}")
    print(f"Fine-grained classes: {num_classes}")
    print(f"Images/class: {args.images_per_class}")
    print(f"Total samples: {num_images}")
    print(f"Image size: {config['img_size']}")
    print(f"Sampling method: {sampling_method}")
    print(f"CFG scale: {config['cfg_scale']}")
    print(f"NFE: {nfe} (num_steps: {num_steps})")
    print(f"Guide model: {GUIDE_MODEL}")
    print(f"Eval model: {EVAL_MODEL}")
    print(
        f"TFG params: rho={guidance_params['rho']}, mu={guidance_params['mu']}, "
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
        print(f"  FID (parent): {result['fid_parent']:.2f}")
        print(f"  FID (child): {result['fid_child']:.2f}")
        print(f"  IS: {result['is_mean']:.2f} ± {result['is_std']:.2f}")
        return

    # Build TFGConfig (apply overrides or auto-calibration)
    auto_calibrated = False
    cal_lambda = args.lambda_mode  # resolved lambda_mode (may be overwritten by auto-calibrate)
    effective_rho = args.rho_override if args.rho_override is not None else guidance_params["rho"]
    effective_mu = args.mu_override if args.mu_override is not None else guidance_params["mu"]

    if args.auto_calibrate and guidance_space in ("v", "v2"):
        from jit_tfg.tfg.calibration import compute_equivalent_rho_mu

        # Resolve effective lambda_mode for calibration.
        # model_type -> pred_target: JiT->x, SiT->v, DiT->e, PixelFlow->v
        _MODEL_TYPE_PRED_TARGET = {"JiT": "x", "SiT": "v", "DiT": "e", "PixelFlow": "v"}
        cal_lambda = args.lambda_mode
        if cal_lambda == "auto":
            pred_target = _MODEL_TYPE_PRED_TARGET.get(model_type, "x")
            cal_lambda = "identity" if pred_target == "x" else "flow_guidance"

        equiv = compute_equivalent_rho_mu(
            x_rho=guidance_params["rho"],
            x_mu=guidance_params["mu"],
            num_steps=num_steps,
            t_eps=0.05,
            lambda_mode=cal_lambda,
        )
        effective_rho = equiv["v_rho_equivalent"]
        effective_mu = equiv["v_mu_equivalent"]
        auto_calibrated = True
        print(f"  [Auto-calibrate] lambda_mode={cal_lambda}")
        print(
            f"  [Auto-calibrate] x-space rho={guidance_params['rho']} -> v-space rho={effective_rho:.2f} (x{equiv['ratio_delta_t']:.1f})"
        )
        print(
            f"  [Auto-calibrate] x-space mu={guidance_params['mu']} -> v-space mu={effective_mu:.2f} (x{equiv['ratio_delta_0']:.1f})"
        )
    elif args.auto_calibrate and guidance_space == "x":
        print("  [Auto-calibrate] Ignored for x-space (calibration only applies to v/v2-space)")

    tfg_config = TFGConfig(
        rho=effective_rho,
        mu=effective_mu,
        sigma=guidance_params["sigma"],
        iter_steps=guidance_params["iter_steps"],
        recur_steps=guidance_params["recur_steps"],
        rho_schedule=guidance_params["rho_schedule"],
        mu_schedule=guidance_params["mu_schedule"],
        sigma_schedule=guidance_params["sigma_schedule"],
        rescale_mode=args.rescale_mode,
        lambda_mode=args.lambda_mode,
        device=args.device,
        seed=args.seed,
    )

    if args.rho_override is not None or args.mu_override is not None:
        print(f"  [Override] rho={effective_rho}, mu={effective_mu}")
    if args.rescale_mode != "clip":
        print(f"  [Override] rescale_mode={args.rescale_mode}")
    if args.lambda_mode != "auto":
        print(f"  [Override] lambda_mode={args.lambda_mode}")

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

    # Save experiment config
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
        "lambda_mode": args.lambda_mode,
        "lambda_mode_resolved": cal_lambda,
        "rescale_mode": args.rescale_mode,
        "auto_calibrated": auto_calibrated,
        "guide_model": GUIDE_MODEL,
        "eval_model": EVAL_MODEL,
        "num_finegrained_classes": num_classes,
        "finegrained_classes": finegrained_classes,
        "images_per_class": args.images_per_class,
        "num_images": num_images,
        "seed": args.seed,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    # Generate samples
    print("\n" + "=" * 70)
    print("Generating samples...")
    print("=" * 70)

    img_dir = output_dir / "images"
    generate_finegrained_samples(
        sampler=sampler,
        guider=guider,
        finegrained_classes=finegrained_classes,
        finegrained_to_imagenet=finegrained_to_imagenet,
        images_per_class=args.images_per_class,
        batch_size=args.batch_size,
        device=args.device,
        cfg_scale=config["cfg_scale"],
        num_steps=num_steps,
        output_dir=img_dir,
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating generated images...")
    print("=" * 70)

    result = evaluate_generated_images(
        img_dir=img_dir,
        finegrained_classes=finegrained_classes,
        images_per_class=args.images_per_class,
        batch_size=args.batch_size,
        device=args.device,
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
    print(f"  FID (parent): {result.fid_parent:.2f}")
    print(f"  FID (child): {result.fid_child:.2f}")
    print(f"  IS: {result.is_mean:.2f} ± {result.is_std:.2f}")
    print(f"\nResults saved to: {metrics_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
