"""Shared utilities for latent diffusion model evaluation.

This module provides common utilities for DiT and SiT evaluation scripts,
including VAE decode + save pipeline and multi-GPU utilities.

Used by:
- dit_fid_eval.py
- sit_fid_eval.py
- dit_hyperparam_search.py
- sit_hyperparam_search.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from jit_tfg.models.dit.vae import VAEHandler


def decode_and_save_images(
    latents: torch.Tensor,
    vae: VAEHandler,
    output_folder: Path,
    start_idx: int,
    decode_batch_size: int = 16,
) -> int:
    """Decode latents to images and save to disk.

    Memory-efficient implementation that decodes in smaller batches to avoid OOM
    on GPUs with limited VRAM (e.g., RTX 5070 Ti with 16GB).

    Args:
        latents: Latent tensor of shape (B, 4, H, W).
        vae: VAEHandler for latent -> pixel conversion.
        output_folder: Folder to save images.
        start_idx: Starting image index for filenames.
        decode_batch_size: Batch size for VAE decode (smaller = less memory).

    Returns:
        Number of images saved.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    num_saved = 0

    for i in range(0, len(latents), decode_batch_size):
        batch = latents[i : i + decode_batch_size]

        # Decode to pixel space: (B, 3, 256, 256) in [-1, 1]
        with torch.no_grad():
            images = vae.decode(batch)

        # Convert to [0, 1] and save
        images = (images + 1) / 2
        images = images.detach().cpu()

        for b_id in range(images.size(0)):
            img_id = start_idx + i + b_id
            gen_img = np.round(np.clip(images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]  # RGB -> BGR for cv2
            cv2.imwrite(str(output_folder / f"{str(img_id).zfill(5)}.png"), gen_img)
            num_saved += 1

        # Free memory
        del images

    return num_saved


def count_existing_images(output_dir: Path) -> int:
    """Count existing PNG images in output directory.

    Args:
        output_dir: Directory to scan.

    Returns:
        Number of existing PNG images.
    """
    if not output_dir.exists():
        return 0

    return len(list(output_dir.glob("*.png")))


def get_max_existing_index(output_dir: Path) -> int:
    """Get the maximum existing image index in output directory.

    Args:
        output_dir: Directory to scan.

    Returns:
        Maximum image index found, or -1 if no images exist.
    """
    if not output_dir.exists():
        return -1

    max_idx = -1
    pattern = re.compile(r"^(\d+)\.png$")

    for filepath in output_dir.iterdir():
        if not filepath.is_file():
            continue
        match = pattern.match(filepath.name)
        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)

    return max_idx


def get_generation_tasks(
    num_images: int,
    existing_count: int,
    world_size: int,
    rank: int,
    batch_size: int,
) -> tuple[int, int, int]:
    """Calculate generation tasks for distributed execution.

    Accounts for existing images and distributes remaining work across GPUs.

    Args:
        num_images: Total number of images to generate.
        existing_count: Number of images already generated.
        world_size: Total number of processes.
        rank: Current process rank.
        batch_size: Batch size for generation.

    Returns:
        Tuple of (start_idx, num_to_generate, num_steps).
        - start_idx: Starting image index for this rank.
        - num_to_generate: Number of images this rank should generate.
        - num_steps: Number of generation steps (batches).
    """
    remaining = num_images - existing_count
    if remaining <= 0:
        return 0, 0, 0

    # Each rank handles a portion of remaining images
    images_per_rank = remaining // world_size
    extra = remaining % world_size

    # Distribute extra images to lower ranks
    if rank < extra:
        num_to_generate = images_per_rank + 1
        start_idx = existing_count + rank * (images_per_rank + 1)
    else:
        num_to_generate = images_per_rank
        start_idx = existing_count + extra * (images_per_rank + 1) + (rank - extra) * images_per_rank

    num_steps = (num_to_generate + batch_size - 1) // batch_size

    return start_idx, num_to_generate, num_steps


def get_class_labels_for_balanced_generation(
    num_images: int,
    num_classes: int = 1000,
) -> np.ndarray:
    """Generate class labels for balanced class-conditional generation.

    Creates labels such that each class has approximately equal representation.

    Args:
        num_images: Total number of images to generate.
        num_classes: Number of classes (default: 1000 for ImageNet).

    Returns:
        Array of class labels of shape (num_images,).
    """
    # Repeat each class label to get balanced distribution
    labels = np.arange(0, num_classes).repeat(num_images // num_classes)

    # Add padding for remaining images (cycle through classes)
    remaining = num_images - len(labels)
    if remaining > 0:
        labels = np.hstack([labels, np.arange(0, remaining)])

    return labels


def save_single_image(
    img: torch.Tensor,
    label: int,
    img_idx: int,
    output_dir: Path,
    prefix: str = "img",
) -> str:
    """Save a single generated image to disk.

    Uses the same conversion as jit_fid_eval.py:
    [-1, 1] -> [0, 1] -> [0, 255] uint8 with rounding.

    Args:
        img: Single image tensor of shape (C, H, W) in [-1, 1] range.
        label: Class label.
        img_idx: Index within the class (for filename).
        output_dir: Output directory.
        prefix: Filename prefix.

    Returns:
        Path to saved image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert from [-1, 1] to [0, 1]
    img = (img + 1) / 2
    img = img.detach().cpu()

    # Same conversion as jit_fid_eval.py: round, clip, uint8, RGB->BGR
    gen_img = np.round(np.clip(img.numpy().transpose([1, 2, 0]) * 255, 0, 255))
    gen_img = gen_img.astype(np.uint8)[:, :, ::-1]  # RGB -> BGR for cv2
    path = output_dir / f"{prefix}_class{label:04d}_{img_idx:04d}.png"
    cv2.imwrite(str(path), gen_img)
    return str(path)


def count_images_per_class(output_dir: Path, prefix: str = "img") -> dict[int, int]:
    """Count existing images per class in output directory.

    Args:
        output_dir: Directory to scan.
        prefix: Filename prefix to match.

    Returns:
        Dict mapping class_id -> count of existing images.
    """
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
