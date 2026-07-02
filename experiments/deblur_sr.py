"""Inverse problem (deblur, super-resolution) TFG-guided evaluation for all supported models.

This script evaluates diffusion models using CFG + TFG guidance for inverse
problems: Gaussian deblur and 4x super-resolution. Reference ImageNet images
are loaded, degraded, and TFG guides generation toward reconstructing them.

Experiment Design:
    - Load reference images from ImageNet validation set (1st image per class)
    - Apply degradation (blur or 4x downsample) + Gaussian noise (sigma=0.05)
    - CFG labels = ImageNet class indices (model's class-conditional generation)
    - TFG targets = measurement indices (for guider's per-sample lookup)
    - Evaluate: LPIPS, PSNR, SSIM (generated vs. reference)

Guidance Modes:
    - dps: DPS (rho-only, no mean guidance)
    - tfg: TFG (rho + mu, iter_steps=4)
    - tfg-full: TFG with recurrence (same as tfg but recur_steps=4)

Tasks:
    - deblur: Gaussian blur (kernel=61, std=3.0) + noise
    - super_resolution: 4x bicubic downsample + noise

Usage:
    # Deblur with JiT-H/16
    uv run python experiments/deblur_sr.py --task deblur --model jit-h-16 \
        --guidance_mode dps --imagenet_dir data/imagenet/val

    # Super-resolution with DiT
    uv run python experiments/deblur_sr.py --task super_resolution --model dit \
        --guidance_mode tfg --imagenet_dir data/imagenet/val

    # Per-model rho sweep (paper appendix; each rho gets its own run dir)
    uv run python experiments/deblur_sr.py --task deblur --model jit-h-16 \
        --guidance_mode dps --rho_override 16

    # Quick test
    uv run python experiments/deblur_sr.py --task deblur --model jit-h-16 \
        --guidance_mode dps --num_classes 2 --batch_size 2

Output Structure:
    outputs/deblur/
    ├── reference/                    # Original 256x256 images (shared)
    ├── degraded/                     # Degraded + noisy images (shared)
    └── jit-h-16/heun/x/dps/         # Per-model experiment
        # (with --rho_override R the leaf dir is e.g. dps-rho16.00)
        ├── config.json
        ├── metrics.json              # {lpips, psnr, ssim, num_images}
        └── images/
"""

import argparse
import datetime
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from jit_tfg.tfg import TFGConfig, UnifiedSampler
from jit_tfg.tfg.guiders.inverse import create_inverse_guider


# =============================================================================
# Constants
# =============================================================================

CLASS_INDEX_PATH = Path(__file__).parent / "imagenet_class_index.json"
DEFAULT_IMAGENET_DIR = Path("data/imagenet/val")

# Models valid for 256x256 inverse problems (same as finegrained)
VALID_256_MODELS = [
    "jit-b-16",
    "jit-l-16",
    "jit-h-16",
    "dit",
    "sit",
    "pixelflow",
]


# =============================================================================
# TFG / DPS Presets (from original TFG scripts/gaussian_deblur.sh and super_resolution.sh)
# =============================================================================

DEBLUR_PRESETS: dict[str, dict[str, Any]] = {
    "tfg": {
        "rho": 1.0,
        "mu": 8.0,
        "sigma": 0.01,
        "iter_steps": 4,
        "recur_steps": 1,
        "rho_schedule": "increase",
        "mu_schedule": "increase",
        "sigma_schedule": "decrease",
    },
    "tfg-full": {
        "rho": 1.0,
        "mu": 8.0,
        "sigma": 0.01,
        "iter_steps": 4,
        "recur_steps": 4,
        "rho_schedule": "increase",
        "mu_schedule": "increase",
        "sigma_schedule": "decrease",
    },
    "dps": {
        "rho": 1.0,
        "mu": 0.0,
        "sigma": 0.0,
        "iter_steps": 1,
        "recur_steps": 1,
        "rho_schedule": "increase",
        "mu_schedule": "increase",
        "sigma_schedule": "decrease",
    },
}

SR_PRESETS: dict[str, dict[str, Any]] = {
    "tfg": {
        "rho": 4.0,
        "mu": 2.0,
        "sigma": 0.01,
        "iter_steps": 4,
        "recur_steps": 1,
        "rho_schedule": "increase",
        "mu_schedule": "increase",
        "sigma_schedule": "decrease",
    },
    "tfg-full": {
        "rho": 4.0,
        "mu": 2.0,
        "sigma": 0.01,
        "iter_steps": 4,
        "recur_steps": 4,
        "rho_schedule": "increase",
        "mu_schedule": "increase",
        "sigma_schedule": "decrease",
    },
    "dps": {
        "rho": 4.0,
        "mu": 0.0,
        "sigma": 0.0,
        "iter_steps": 1,
        "recur_steps": 1,
        "rho_schedule": "increase",
        "mu_schedule": "increase",
        "sigma_schedule": "decrease",
    },
}

TASK_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "deblur": DEBLUR_PRESETS,
    "super_resolution": SR_PRESETS,
}


# =============================================================================
# NFE / Step Conversion
# =============================================================================


def nfe_to_steps(nfe, sampling_method):
    if sampling_method == "heun":
        return nfe // 2
    return nfe


# =============================================================================
# Model Configurations
# =============================================================================


def load_model_configs():
    config_path = Path(__file__).parent / "model_configs.json"
    with open(config_path) as f:
        return json.load(f)


MODEL_CONFIGS = load_model_configs()


# =============================================================================
# ImageNet Reference Loading
# =============================================================================


def load_imagenet_references(imagenet_dir, class_index_path, num_classes=1000, images_per_class=1):
    """Load reference images from ImageNet validation set.

    Loads the first `images_per_class` images per class (sorted by filename),
    resized and center-cropped to 256x256, normalized to [-1, 1].

    Uses imagenet_class_index.json to map class indices to wordnet IDs,
    then finds the matching directory in the validation set.

    Args:
        imagenet_dir: Path to ImageNet validation directory.
        class_index_path: Path to imagenet_class_index.json.
        num_classes: Number of classes to load (first N). Default: 1000.
        images_per_class: Number of images per class. Default: 1.

    Returns:
        Tuple of:
            - images: Tensor of shape (N*images_per_class, 3, 256, 256) in [-1, 1]
            - class_indices: List of (class_idx, img_idx) tuples
    """
    imagenet_dir = Path(imagenet_dir)
    class_index_path = Path(class_index_path)

    if not class_index_path.exists():
        raise FileNotFoundError(f"Class index file not found: {class_index_path}")

    with open(class_index_path) as f:
        class_index = json.load(f)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Build directory lookup: wordnet_id -> dir_path
    val_dirs = {}
    for d in imagenet_dir.iterdir():
        if d.is_dir():
            # ImageNet val dirs are typically named by wordnet ID (e.g., n01440764)
            val_dirs[d.name] = d

    images = []
    class_info = []  # (class_idx, img_idx)

    for cls_idx in range(num_classes):
        key = str(cls_idx)
        if key not in class_index:
            continue

        wordnet_id, class_name = class_index[key]

        # Try matching by wordnet_id first (most reliable)
        dir_path = val_dirs.get(wordnet_id)
        if dir_path is None:
            # Fallback: try matching by class name prefix
            search = class_name.replace("_", " ").lower()
            dir_path = next(
                (d for name, d in val_dirs.items() if name.lower().startswith(search)),
                None,
            )

        if dir_path is None:
            print(f"  WARNING: No directory found for class {cls_idx} ({wordnet_id}, {class_name}), skipping")
            continue

        # Get sorted image files
        img_files = sorted(dir_path.glob("*.JPEG"))
        if not img_files:
            img_files = (
                sorted(dir_path.glob("*.jpeg")) + sorted(dir_path.glob("*.jpg")) + sorted(dir_path.glob("*.png"))
            )

        for img_idx in range(min(images_per_class, len(img_files))):
            img = Image.open(img_files[img_idx]).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)
            class_info.append((cls_idx, img_idx))

    if not images:
        raise RuntimeError(
            f"No images found in {imagenet_dir}. "
            "Ensure the directory contains ImageNet validation images "
            "organized by wordnet ID (e.g., n01440764/)."
        )

    return torch.stack(images), class_info


# =============================================================================
# Model Loading (reused from finegrained_bird_tfg.py)
# =============================================================================


def load_jit_model(config, device):
    from jit_tfg.models.jit.utils.checkpoint import load_checkpoint_for_inference

    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"JiT checkpoint not found: {checkpoint_path}")
    denoiser, _ = load_checkpoint_for_inference(checkpoint_path=checkpoint_path, device=device, use_ema=True)
    return denoiser


def load_dit_model(config, device, num_steps):
    from jit_tfg.models.dit.denoiser import load_dit_denoiser

    return load_dit_denoiser(
        checkpoint_path=config.get("checkpoint"),
        from_pretrained=config.get("from_pretrained"),
        device=device,
        cfg_scale=config["cfg_scale"],
        num_sampling_steps=num_steps,
        vae_type=config.get("vae_type", "ema"),
    )


def load_sit_model(config, device, num_steps):
    from jit_tfg.models.sit.denoiser import load_sit_denoiser

    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SiT checkpoint not found: {checkpoint_path}")
    return load_sit_denoiser(
        checkpoint_path=checkpoint_path,
        device=device,
        cfg_scale=config["cfg_scale"],
        num_sampling_steps=num_steps,
        vae_type=config.get("vae_type", "ema"),
    )


def load_pixelflow_model(config, device, num_steps):
    from jit_tfg.models.pixelflow.denoiser import load_pixelflow_denoiser

    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PixelFlow checkpoint not found: {checkpoint_path}")
    return load_pixelflow_denoiser(
        checkpoint_path=checkpoint_path,
        device=device,
        cfg_scale=config["cfg_scale"],
        num_sampling_steps=num_steps,
    )


def load_model(model_name, device, num_steps):
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


def save_single_image(img, label, img_idx, output_dir, prefix="img"):
    """Save a single image tensor to disk.

    Args:
        img: Image tensor (C, H, W) in [-1, 1] range.
        label: Class label for filename.
        img_idx: Image index for filename.
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


def count_existing_images(output_dir, prefix="img"):
    """Count existing images per class in output directory."""
    counts = {}
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
# Reference Preparation
# =============================================================================


def prepare_references(
    task,
    imagenet_dir,
    class_index_path,
    num_classes,
    images_per_class,
    output_dir,
    device,
    noise_sigma=0.05,
    seed=42,
):
    """Prepare reference and degraded images. Idempotent - skips if already done.

    Args:
        task: Task name ("deblur" or "super_resolution").
        imagenet_dir: Path to ImageNet validation directory.
        class_index_path: Path to imagenet_class_index.json.
        num_classes: Number of classes.
        images_per_class: Images per class.
        output_dir: Base output directory for the task.
        device: Device for degradation computation.
        noise_sigma: Measurement noise sigma.
        seed: Random seed.

    Returns:
        Tuple of:
            - guider: InverseProblemGuider instance
            - ref_images: Reference images tensor (N, 3, 256, 256)
            - class_info: List of (class_idx, img_idx) tuples
    """
    ref_dir = output_dir / "reference"
    deg_dir = output_dir / "degraded"

    # Load reference images
    print("Loading ImageNet reference images...")
    ref_images, class_info = load_imagenet_references(
        imagenet_dir,
        class_index_path,
        num_classes,
        images_per_class,
    )
    print(f"  Loaded {len(ref_images)} reference images from {num_classes} classes")

    # Check if reference/degraded already prepared
    expected_count = len(class_info)
    existing_ref = count_existing_images(ref_dir, prefix="ref")
    existing_deg = count_existing_images(deg_dir, prefix="deg")
    total_ref = sum(existing_ref.values())
    total_deg = sum(existing_deg.values())

    if total_ref >= expected_count and total_deg >= expected_count:
        print(f"  Reference ({total_ref}) and degraded ({total_deg}) images already prepared, skipping...")
    else:
        print("  Saving reference images...")
        ref_dir.mkdir(parents=True, exist_ok=True)
        for i, (cls_idx, img_idx) in enumerate(tqdm(class_info, desc="  Saving refs")):
            save_single_image(ref_images[i], cls_idx, img_idx, ref_dir, prefix="ref")

    # Create guider (always needed for generation)
    print(f"  Creating {task} degradation operator and measurements...")
    guider, measurements, degraded_clean = create_inverse_guider(
        task=task,
        reference_images=ref_images,
        device=device,
        noise_sigma=noise_sigma,
        seed=seed,
    )

    # Save degraded images if not already done
    if total_deg < expected_count:
        print("  Saving degraded images...")
        deg_dir.mkdir(parents=True, exist_ok=True)
        for i, (cls_idx, img_idx) in enumerate(tqdm(class_info, desc="  Saving degraded")):
            save_single_image(degraded_clean[i], cls_idx, img_idx, deg_dir, prefix="deg")

    return guider, ref_images, class_info


# =============================================================================
# Generation
# =============================================================================


def generate_inverse_samples(
    sampler,
    guider,
    class_info,
    batch_size,
    device,
    cfg_scale,
    num_steps,
    output_dir,
    prefix="img",
    show_progress=True,
):
    """Generate guided samples for inverse problem reconstruction.

    Args:
        sampler: UnifiedSampler instance (with TFG config).
        guider: InverseProblemGuider instance.
        class_info: List of (class_idx, img_idx) tuples.
        batch_size: Batch size for generation.
        device: Device for generation.
        cfg_scale: CFG scale.
        num_steps: Number of sampling steps.
        output_dir: Directory to save images.
        prefix: Filename prefix.
        show_progress: Whether to show progress bar.

    Returns:
        Total number of images generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_counts = count_existing_images(output_dir, prefix)
    total_images = len(class_info)

    # Build generation tasks: (measurement_index, class_idx, img_idx)
    generation_tasks = []
    for meas_idx, (cls_idx, img_idx) in enumerate(class_info):
        existing = existing_counts.get(cls_idx, 0)
        if existing > img_idx:
            continue  # Already generated
        generation_tasks.append((meas_idx, cls_idx, img_idx))

    total_existing = total_images - len(generation_tasks)
    if total_existing > 0:
        print(f"  Found {total_existing}/{total_images} existing images, resuming...")

    if len(generation_tasks) == 0:
        print("  All images already generated.")
        return total_images

    iterator = range(0, len(generation_tasks), batch_size)
    total_batches = (len(generation_tasks) + batch_size - 1) // batch_size

    if show_progress:
        desc = f"Generating ({total_existing}/{total_images} done)"
        iterator = tqdm(iterator, desc=desc, total=total_batches)

    for batch_start in iterator:
        batch_tasks = generation_tasks[batch_start : batch_start + batch_size]
        batch_meas_indices = [task[0] for task in batch_tasks]
        batch_cls_indices = [task[1] for task in batch_tasks]
        batch_img_indices = [task[2] for task in batch_tasks]

        # CFG labels = ImageNet class indices (model's class conditioning)
        cfg_labels = torch.tensor(batch_cls_indices, device=device, dtype=torch.long)
        # TFG targets = measurement indices (guider indexes into measurements)
        tfg_targets = torch.tensor(batch_meas_indices, device=device, dtype=torch.long)

        with torch.amp.autocast(device, dtype=torch.bfloat16):
            images = sampler.generate(
                guidance=guider,
                tfg_targets=tfg_targets,
                cfg_labels=cfg_labels,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
                show_progress=False,
            )

        for img, cls_idx, img_idx in zip(images, batch_cls_indices, batch_img_indices):
            save_single_image(img.float(), cls_idx, img_idx, output_dir, prefix)

        del images
        torch.cuda.empty_cache()

    if show_progress:
        print("  Generation complete.")

    return total_images


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_inverse_results(img_dir, ref_dir, class_info, batch_size, device):
    """Evaluate generated images against references using LPIPS, PSNR, SSIM.

    Args:
        img_dir: Directory containing generated images.
        ref_dir: Directory containing reference images.
        class_info: List of (class_idx, img_idx) tuples.
        batch_size: Batch size for metric computation.
        device: Device for evaluation.

    Returns:
        Dict with lpips, psnr, ssim, num_images.
    """
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    print("  Initializing metrics...")
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1] range
    ])

    count = 0

    print("  Computing metrics...")
    for i in tqdm(range(0, len(class_info), batch_size), desc="  Evaluating"):
        batch_info = class_info[i : i + batch_size]
        gen_imgs = []
        ref_imgs = []

        for cls_idx, img_idx in batch_info:
            gen_path = img_dir / f"img_class{cls_idx:04d}_{img_idx:04d}.png"
            ref_path = ref_dir / f"ref_class{cls_idx:04d}_{img_idx:04d}.png"

            if not gen_path.exists():
                continue

            gen_img = transform(Image.open(gen_path).convert("RGB"))
            ref_img = transform(Image.open(ref_path).convert("RGB"))
            gen_imgs.append(gen_img)
            ref_imgs.append(ref_img)

        if not gen_imgs:
            continue

        gen_batch = torch.stack(gen_imgs).to(device)
        ref_batch = torch.stack(ref_imgs).to(device)

        with torch.no_grad():
            lpips_metric.update(gen_batch, ref_batch)
            psnr_metric.update(gen_batch, ref_batch)
            ssim_metric.update(gen_batch, ref_batch)

        count += len(gen_imgs)

    if count == 0:
        print("  WARNING: No image pairs found for evaluation!")
        return {"lpips": 0.0, "psnr": 0.0, "ssim": 0.0, "num_images": 0}

    lpips_avg = lpips_metric.compute().item()
    psnr_avg = psnr_metric.compute().item()
    ssim_avg = ssim_metric.compute().item()

    return {
        "lpips": lpips_avg,
        "psnr": psnr_avg,
        "ssim": ssim_avg,
        "num_images": count,
    }


# =============================================================================
# Validation
# =============================================================================


def validate_sampling_method(model_type, sampling_method):
    if model_type == "DiT":
        assert sampling_method in ("ddim", "ddpm"), f"DiT only supports 'ddim' or 'ddpm', got '{sampling_method}'."
    else:
        assert sampling_method not in ("ddim", "ddpm"), (
            f"{model_type} does not support '{sampling_method}'. Use 'euler' or 'heun'."
        )


def validate_guidance_space(sampling_method, guidance_space):
    if sampling_method in ("ddpm", "ddim"):
        if guidance_space != "x":
            raise ValueError(f"DDPM/DDIM only supports guidance_space='x', got '{guidance_space}'.")
    elif sampling_method == "euler":
        if guidance_space not in ("x", "v"):
            raise ValueError(f"Euler supports guidance_space='x' or 'v', got '{guidance_space}'.")
    elif sampling_method == "heun":
        if guidance_space not in ("x", "v", "v2"):
            raise ValueError(f"Unknown guidance_space '{guidance_space}'. Heun supports: 'x', 'v', 'v2'.")


# =============================================================================
# Argument Parsing
# =============================================================================


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Inverse Problem (Deblur/SR) TFG-guided Evaluation",
        add_help=True,
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["deblur", "super_resolution"],
        help="Inverse problem task: 'deblur' or 'super_resolution'",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=VALID_256_MODELS,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--guidance_mode",
        type=str,
        required=True,
        choices=["dps", "tfg", "tfg-full"],
        help="Guidance mode: 'dps', 'tfg', or 'tfg-full'",
    )
    parser.add_argument(
        "--guidance_space",
        type=str,
        default="x",
        choices=["x", "v", "v2"],
        help="Guidance space. Heun: x/v/v2, Euler: x/v, DDPM/DDIM: x only.",
    )
    parser.add_argument(
        "--imagenet_dir",
        type=str,
        default=str(DEFAULT_IMAGENET_DIR),
        help="Path to ImageNet validation directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory base (default: outputs/{task})",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of ImageNet classes to use (default: 1000)",
    )
    parser.add_argument(
        "--images_per_class",
        type=int,
        default=1,
        help="Number of images per class (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
        "--sampling_method",
        type=str,
        default=None,
        choices=["euler", "heun", "ddim", "ddpm"],
        help="Sampling method override (default: from model_configs.json)",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        default=None,
        help="Number of Function Evaluations override (default: from model_configs.json)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name in output path (e.g., 'sit-nfe100')",
    )
    parser.add_argument(
        "--rho_override",
        type=float,
        default=None,
        help="Override the preset rho value. The paper appendix sweeps "
        "model-specific rho (up to 16 for deblur, up to 32 for SR); the "
        "presets only pin the original TFG defaults (deblur: 1.0, SR: 4.0). "
        "Each rho value writes to its own output directory.",
    )

    return parser


# =============================================================================
# Main
# =============================================================================


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task = args.task
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

    # Get TFG preset for this task + mode (apply --rho_override if given)
    presets = TASK_PRESETS[task]
    guidance_params = presets[guidance_mode]
    effective_rho = args.rho_override if args.rho_override is not None else guidance_params["rho"]

    # Output directory. The effective rho is encoded in the directory name when
    # overridden, so sweep points write to distinct run dirs and resume/cache
    # logic (metrics.json, images/) stays per-rho.
    output_guidance_mode = guidance_mode
    if args.rho_override is not None:
        output_guidance_mode = f"{guidance_mode}-rho{args.rho_override:.2f}"
    base_output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / task
    exp_output_dir = base_output_dir / output_name / sampling_method / guidance_space / output_guidance_mode
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Inverse Problem Evaluation: {task}")
    print("=" * 70)
    print(f"Model: {model_name} ({model_type})")
    print(f"Task: {task}")
    print(f"Guidance mode: {guidance_mode}")
    print(f"Guidance space: {guidance_space}")
    print(f"Output: {exp_output_dir}")
    print(f"Classes: {args.num_classes}, Images/class: {args.images_per_class}")
    print(f"Image size: {config['img_size']}")
    print(f"Sampling: {sampling_method}, NFE: {nfe} (steps: {num_steps})")
    print(f"CFG scale: {config['cfg_scale']}")
    print(
        f"TFG params: rho={effective_rho}, mu={guidance_params['mu']}, "
        f"sigma={guidance_params['sigma']}, iter={guidance_params['iter_steps']}, "
        f"recur={guidance_params['recur_steps']}"
    )
    if args.rho_override is not None:
        print(f"  [Override] rho={effective_rho} (preset: {guidance_params['rho']})")
    print("=" * 70)

    # Check cached results
    metrics_path = exp_output_dir / "metrics.json"
    if metrics_path.exists() and not args.force_rerun:
        print(f"\nLoading existing results from {metrics_path}")
        with open(metrics_path) as f:
            result = json.load(f)
        print("\nResults:")
        print(f"  LPIPS: {result['lpips']:.4f}")
        print(f"  PSNR:  {result['psnr']:.2f}")
        print(f"  SSIM:  {result['ssim']:.4f}")
        return

    # Prepare references (shared across models for same task)
    print("\n" + "=" * 70)
    print("Preparing reference images...")
    print("=" * 70)

    guider, ref_images, class_info = prepare_references(
        task=task,
        imagenet_dir=args.imagenet_dir,
        class_index_path=CLASS_INDEX_PATH,
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
        output_dir=base_output_dir,
        device=args.device,
        seed=args.seed,
    )

    # Build TFGConfig
    tfg_config = TFGConfig(
        rho=effective_rho,
        mu=guidance_params["mu"],
        sigma=guidance_params["sigma"],
        iter_steps=guidance_params["iter_steps"],
        recur_steps=guidance_params["recur_steps"],
        rho_schedule=guidance_params["rho_schedule"],
        mu_schedule=guidance_params["mu_schedule"],
        sigma_schedule=guidance_params["sigma_schedule"],
        device=args.device,
        seed=args.seed,
    )

    # Load model
    denoiser = load_model(model_name, args.device, num_steps)

    print("\nCreating UnifiedSampler...")
    sampler = UnifiedSampler(
        model_type=model_type,
        denoiser=denoiser,
        tfg_config=tfg_config,
        sampling_method=sampling_method,
        guidance_space=guidance_space,
    )

    # Save experiment config
    exp_config = {
        "task": task,
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
        "tfg_params": {**guidance_params, "rho": effective_rho},
        "rho_override": args.rho_override,
        "num_classes": args.num_classes,
        "images_per_class": args.images_per_class,
        "num_images": len(class_info),
        "noise_sigma": 0.05,
        "seed": args.seed,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(exp_output_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    # Generate samples
    print("\n" + "=" * 70)
    print("Generating samples...")
    print("=" * 70)

    img_dir = exp_output_dir / "images"
    generate_inverse_samples(
        sampler=sampler,
        guider=guider,
        class_info=class_info,
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

    ref_dir = base_output_dir / "reference"
    result = evaluate_inverse_results(
        img_dir=img_dir,
        ref_dir=ref_dir,
        class_info=class_info,
        batch_size=args.batch_size,
        device=args.device,
    )

    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults for {model_name} ({task}, {guidance_mode}, {guidance_space}-space):")
    print(f"  LPIPS: {result['lpips']:.4f}")
    print(f"  PSNR:  {result['psnr']:.2f}")
    print(f"  SSIM:  {result['ssim']:.4f}")
    print(f"  Images: {result['num_images']}")
    print(f"\nResults saved to: {metrics_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
