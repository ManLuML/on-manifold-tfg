"""Checkpoint loading utilities for inference.

This module provides utilities for loading trained JiT checkpoints
for inference, ensuring that the model configuration (pred_target,
loss_type, etc.) matches the saved checkpoint.

Supports two checkpoint formats:
    1. Our format: Contains 'model', 'args', optionally 'model_ema1', 'model_ema2'
    2. Original JiT format: Contains 'model', 'model_ema1', 'model_ema2' without 'args'
       - Original weights use 'net.' prefix in state dict keys
       - Requires manual configuration specification

Usage:
    from jit_tfg.models.jit.utils.checkpoint import load_checkpoint_for_inference

    model, config = load_checkpoint_for_inference(
        checkpoint_path="output/20250629/imagenet256-JiTB16-x-143052/checkpoint-last.pth",
        device="cuda",
    )

    labels = torch.randint(0, 1000, (4,), device="cuda")
    images = model.generate(labels)
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Optional

import torch

from jit_tfg.models.jit.denoiser import Denoiser

logger = logging.getLogger(__name__)


# JiT official checkpoint configurations based on model variant
JIT_OFFICIAL_CONFIGS = {
    "JiT-B/16": {"img_size": 256, "noise_scale": 1.0, "cfg": 3.0},
    "JiT-B/32": {"img_size": 512, "noise_scale": 2.0, "cfg": 3.0},
    "JiT-L/16": {"img_size": 256, "noise_scale": 1.0, "cfg": 2.4},
    "JiT-L/32": {"img_size": 512, "noise_scale": 2.0, "cfg": 2.5},
    "JiT-H/16": {"img_size": 256, "noise_scale": 1.0, "cfg": 2.2},
    "JiT-H/32": {"img_size": 512, "noise_scale": 2.0, "cfg": 2.3},
}


def infer_model_from_filename(filename: str) -> str | None:
    """Infer model variant from checkpoint filename.

    Parses filenames like 'jit-b-16.pth', 'jit-l-32.pth', 'jit-h-16.pth'
    to determine the model variant.

    Args:
        filename: Checkpoint filename (with or without path).

    Returns:
        Model name like 'JiT-B/16' or None if cannot infer.

    Examples:
        >>> infer_model_from_filename('jit-b-16.pth')
        'JiT-B/16'
        >>> infer_model_from_filename('jit-h-32.pth')
        'JiT-H/32'
    """
    basename = Path(filename).stem.lower()

    pattern = r"jit-([blh])-(\d+)"
    match = re.search(pattern, basename)

    if match:
        size = match.group(1).upper()
        patch = match.group(2)
        return f"JiT-{size}/{patch}"

    return None


def _create_default_args_for_jit(
    model: str = "JiT-B/16",
    checkpoint_path: str | None = None,
) -> argparse.Namespace:
    """Create default args for JiT models (original checkpoint format).

    The original JiT checkpoints from https://github.com/LTH14/JiT don't contain
    'args'. This function creates a compatible args namespace with default values.

    If checkpoint_path is provided, attempts to infer the model variant from
    the filename (e.g., 'jit-l-16.pth' -> 'JiT-L/16').

    Args:
        model: Model variant name (e.g., 'JiT-B/16', 'JiT-L/32').
        checkpoint_path: Optional path to checkpoint for model inference.

    Returns:
        argparse.Namespace with model configuration.
    """
    if checkpoint_path:
        inferred = infer_model_from_filename(checkpoint_path)
        if inferred:
            model = inferred
            print(f"Inferred model from filename: {model}")

    config = JIT_OFFICIAL_CONFIGS.get(model, JIT_OFFICIAL_CONFIGS["JiT-B/16"])

    return argparse.Namespace(
        model=model,
        img_size=config["img_size"],
        class_num=1000,
        attn_dropout=0.0,
        proj_dropout=0.0,
        pred_target="x",
        loss_type="v",
        label_drop_prob=0.1,
        P_mean=-0.4,
        P_std=1.0,
        t_eps=5e-2,
        noise_scale=config["noise_scale"],
        ema_decay1=0.9999,
        ema_decay2=0.99994,
        sampling_method="heun",
        num_sampling_steps=50,
        cfg=config["cfg"],
        interval_min=0.1,
        interval_max=1.0,
    )


def _create_default_args_for_jit_b16() -> argparse.Namespace:
    """Create default args for JiT-B/16 model (backward compatibility)."""
    return _create_default_args_for_jit("JiT-B/16")


def _detect_checkpoint_format(checkpoint: dict) -> str:
    """Detect the checkpoint format ('our_format' or 'original_jit')."""
    if "args" in checkpoint and checkpoint["args"] is not None:
        return "our_format"
    return "original_jit"


def _get_args_from_checkpoint(
    checkpoint: dict,
    checkpoint_path: str | None = None,
) -> argparse.Namespace:
    """Extract or create args from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary.
        checkpoint_path: Optional path to checkpoint for model inference from filename.

    Returns:
        argparse.Namespace with model configuration.
    """
    ckpt_format = _detect_checkpoint_format(checkpoint)

    if ckpt_format == "our_format":
        args = checkpoint["args"]
        print("Detected our checkpoint format (with args)")
    else:
        print("Detected original JiT checkpoint format (no args)")
        args = _create_default_args_for_jit(checkpoint_path=checkpoint_path)

    if not hasattr(args, "pred_target"):
        logger.warning("Checkpoint missing 'pred_target', defaulting to 'x'")
        args.pred_target = "x"
    if not hasattr(args, "loss_type"):
        logger.warning("Checkpoint missing 'loss_type', defaulting to 'v'")
        args.loss_type = "v"

    return args


def _select_state_dict(checkpoint: dict, use_ema: bool, ema_index: int) -> dict:
    """Select the appropriate state dict from checkpoint."""
    if use_ema and f"model_ema{ema_index}" in checkpoint:
        ema_key = f"model_ema{ema_index}"
        print(f"Loading EMA weights ({ema_key})")
        return checkpoint[ema_key]
    if "model" in checkpoint:
        print("Loading model weights (non-EMA)")
        return checkpoint["model"]
    ema_key = "model_ema1" if "model_ema1" in checkpoint else "model_ema2"
    print(f"Loading EMA weights ({ema_key}) - 'model' key not found")
    return checkpoint[ema_key]


def _load_state_dict_with_validation(model: Denoiser, state_dict: dict) -> None:
    """Load state dict and validate missing/unexpected keys."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        non_critical_missing = [k for k in missing if "freqs_cos" in k or "freqs_sin" in k]
        critical_missing = [k for k in missing if k not in non_critical_missing]
        if critical_missing:
            raise RuntimeError(f"Critical keys missing in state_dict: {critical_missing}")
        if non_critical_missing:
            logger.info(f"Non-critical buffers missing (RoPE freqs): {len(non_critical_missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys in state_dict: {unexpected}")


def load_checkpoint_for_inference(
    checkpoint_path: str | Path,
    device: str = "cuda",
    use_ema: bool = True,
    ema_index: int = 1,
) -> tuple[Denoiser, argparse.Namespace]:
    """Load a trained checkpoint for inference.

    This function loads the checkpoint and reconstructs the model with
    the exact same configuration (pred_target, loss_type, etc.) that
    was used during training.

    Supports both our checkpoint format (with args) and the original JiT
    checkpoint format (without args).

    Args:
        checkpoint_path: Path to the checkpoint file (.pth).
        device: Device to load the model on ("cuda", "cpu", "mps").
        use_ema: Whether to use EMA weights (recommended for inference).
        ema_index: Which EMA to use (1 or 2). Default: 1 (slower decay, more stable).

    Returns:
        Tuple of (model, config):
            - model: Denoiser model loaded with trained weights
            - config: argparse.Namespace with the training configuration

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        KeyError: If checkpoint is missing required keys.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model" not in checkpoint and "model_ema1" not in checkpoint:
        raise KeyError("Checkpoint missing model weights ('model' or 'model_ema1')")

    args = _get_args_from_checkpoint(checkpoint, checkpoint_path=str(ckpt_path))
    print(f"Model config: {args.model}, pred_target={args.pred_target}, loss_type={args.loss_type}")

    model = Denoiser(args)
    model.to(device)

    state_dict = _select_state_dict(checkpoint, use_ema, ema_index)
    _load_state_dict_with_validation(model, state_dict)
    model.eval()

    print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, args


def get_checkpoint_info(checkpoint_path: str | Path) -> dict[str, Any]:
    """Get information about a checkpoint without loading the full model.

    Useful for inspecting checkpoints before loading.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Dictionary containing checkpoint metadata.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_format = _detect_checkpoint_format(checkpoint)
    if ckpt_format == "our_format":
        args = checkpoint["args"]
    else:
        args = _create_default_args_for_jit(checkpoint_path=str(ckpt_path))

    return {
        "epoch": checkpoint.get("epoch", "unknown"),
        "model": getattr(args, "model", "unknown"),
        "pred_target": getattr(args, "pred_target", "x"),
        "loss_type": getattr(args, "loss_type", "v"),
        "img_size": getattr(args, "img_size", 256),
        "has_ema1": "model_ema1" in checkpoint,
        "has_ema2": "model_ema2" in checkpoint,
        "has_optimizer": "optimizer" in checkpoint,
        "format": ckpt_format,
    }


def print_checkpoint_info(checkpoint_path: str | Path) -> None:
    """Print detailed information about a checkpoint."""
    info = get_checkpoint_info(checkpoint_path)

    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)
    print(f"  Format:       {info['format']}")
    print(f"  Epoch:        {info['epoch']}")
    print(f"  Model:        {info['model']}")
    print(f"  Image Size:   {info['img_size']}")
    print(f"  Pred Target:  {info['pred_target']}")
    print(f"  Loss Type:    {info['loss_type']}")
    print(f"  Has EMA1:     {info['has_ema1']}")
    print(f"  Has EMA2:     {info['has_ema2']}")
    print(f"  Has Optimizer:{info['has_optimizer']}")
    print("=" * 60)
