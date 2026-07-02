"""Utility functions for TFG.

This module provides utility functions used throughout the TFG implementation,
including gradient rescaling and numerical stability helpers.
"""

import torch


def rescale_grad(
    grad: torch.Tensor,
    clip_scale: float = 100.0,
    node_mask: torch.Tensor | None = None,
    mode: str = "clip",
    **kwargs,
) -> torch.Tensor:
    """Rescale gradients to prevent explosion.

    Clips the gradient norm to a maximum value while preserving direction.
    This is crucial for stable guidance, especially at high noise levels.

    For molecule data with node masks, computes scale per-sample considering
    only valid nodes.

    Args:
        grad: Gradient tensor of shape (B, ...).
        clip_scale: Maximum allowed gradient scale. Default: 100.0.
        node_mask: Optional mask for valid nodes in molecular data.
            Shape: (B, N, 1) where N is the number of nodes.
        mode: Rescaling mode for image data (node_mask=None).
            - 'clip': Clip gradient if RMS exceeds clip_scale.
            - 'original': Return gradient unchanged, matching original TFG
              (edm/tasks/utils.py) which has no image gradient clipping.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Rescaled gradient tensor of the same shape as input.

    Example:
        >>> grad = torch.randn(4, 3, 256, 256) * 1000  # Large gradients
        >>> rescaled = rescale_grad(grad, clip_scale=10.0)
        >>> assert (rescaled ** 2).mean().sqrt() <= 10.0 * 10  # Approximately bounded
    """
    # Compute per-element squared values
    scale = (grad**2).mean(dim=-1)

    if node_mask is not None:
        # For molecular data: compute scale per sample, considering only valid nodes
        # node_mask: (B, N, 1), scale: (B, N)
        scale_sum = scale.sum(dim=-1)  # (B,)
        num_valid = node_mask.float().squeeze(-1).sum(dim=-1)  # (B,)
        scale = scale_sum / num_valid.clamp_min(1.0)  # (B,)

        clipped_scale = torch.clamp(scale, max=clip_scale)
        coef = clipped_scale / scale.clamp_min(1e-8)  # (B,)
        grad = grad * coef.view(-1, 1, 1)
    elif mode == "original":
        # Match original TFG (edm/tasks/utils.py:20-32): for image data
        # (node_mask=None), the original computes scale but returns gradient
        # unchanged — no clipping is applied.
        pass
    else:
        # For image data: simple gradient clipping
        # Compute overall scale and clip
        overall_scale = scale.mean().sqrt()
        if overall_scale > clip_scale:
            grad = grad * (clip_scale / overall_scale)

    return grad


def ban_requires_grad(module: torch.nn.Module) -> None:
    """Disable gradient computation for all parameters in a module.

    Used to freeze pretrained models (e.g., classifiers) during guidance.

    Args:
        module: PyTorch module to freeze.
    """
    for param in module.parameters():
        param.requires_grad = False


def check_grad_fn(x: torch.Tensor) -> None:
    """Assert that a tensor requires gradients.

    Used to verify that tensors are properly set up for gradient computation
    before computing guidance gradients.

    Args:
        x: Tensor to check.

    Raises:
        AssertionError: If x.requires_grad is False.
    """
    if not x.requires_grad:
        raise ValueError("Tensor should require grad for guidance computation")
