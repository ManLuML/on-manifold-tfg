"""Inverse problem guiders for TFG guidance (deblur, super-resolution).

This module provides guiders for inverse problem guidance, where
TFG steers generation toward reconstructing degraded reference images.

Degradation Operators:
    - GaussianBlurOperator: Gaussian blur (kernel_size=61, std=3.0)
    - SuperResolutionOperator: 4x bicubic downsampling

Guider:
    - InverseProblemGuider: Energy = -||y - A(x)||_2

Factory:
    - create_inverse_guider: Creates operator + guider from task name

Reference:
    TFG Paper: gaussian_deblur.py, super_resolution.py
    Original operators: image_inverse_operator.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jit_tfg.tfg.guiders.base import BaseGuider
from jit_tfg.tfg.utils import check_grad_fn, rescale_grad


class GaussianBlurOperator(nn.Module):
    """Gaussian blur degradation operator.

    Applies Gaussian blur via depthwise convolution with reflection padding.
    Matches the original TFG implementation (image_inverse_operator.py:1224).

    Args:
        kernel_size: Size of the Gaussian kernel. Default: 61.
        std: Standard deviation of the Gaussian kernel. Default: 3.0.
    """

    def __init__(self, kernel_size: int = 61, std: float = 3.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.std = std

        kernel = self._make_gaussian_kernel(kernel_size, std)
        # Shape: (3, 1, K, K) for depthwise conv over 3 channels
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel)

        self.pad = nn.ReflectionPad2d(kernel_size // 2)

    @staticmethod
    def _make_gaussian_kernel(size: int, std: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel.

        Args:
            size: Kernel size (must be odd).
            std: Standard deviation.

        Returns:
            Normalized 2D Gaussian kernel of shape (size, size).
        """
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * std**2))
        kernel = g.outer(g)
        return kernel / kernel.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur.

        Args:
            x: Input images of shape (B, 3, H, W).

        Returns:
            Blurred images of shape (B, 3, H, W).
        """
        return F.conv2d(self.pad(x), self.kernel, groups=3)


class SuperResolutionOperator(nn.Module):
    """4x bicubic downsampling operator for super-resolution.

    Downsamples images by factor of 4 using bicubic interpolation.
    Uses PyTorch's built-in F.interpolate with antialias, which is
    the modern equivalent of the original TFG's custom Resizer.

    Args:
        scale_factor: Downsampling factor. Default: 4.
    """

    def __init__(self, scale_factor: int = 4) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 4x bicubic downsampling.

        Args:
            x: Input images of shape (B, 3, H, W).

        Returns:
            Downsampled images of shape (B, 3, H//4, W//4).
        """
        h, w = x.shape[2], x.shape[3]
        return F.interpolate(
            x,
            size=(h // self.scale_factor, w // self.scale_factor),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )


class InverseProblemGuider(BaseGuider):
    """Guider for inverse problems (deblur, super-resolution).

    Computes guidance gradients using the measurement consistency energy:
        E(x) = -||y - A(x)||_2

    where y is the degraded measurement, A is the degradation operator,
    and x is the current x_0 prediction.

    The sampler handles latent->pixel decoding via _decode_latent_for_guidance(),
    so this guider always receives pixel-space inputs regardless of model type.

    Measurements are stored on CPU to save VRAM and moved to GPU per-batch.
    The `targets` parameter is used as measurement indices (reusing the
    existing BaseGuider interface).

    Attributes:
        operator: Degradation operator (blur or downsample).
        measurements: Pre-computed degraded+noisy reference images on CPU.
        clip_scale: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        operator: nn.Module,
        measurements: torch.Tensor,
        clip_scale: float = 1.0,
        device: str = "cuda",
    ) -> None:
        """Initialize the inverse problem guider.

        Args:
            operator: Differentiable degradation operator (e.g., GaussianBlurOperator).
            measurements: Pre-computed degraded+noisy reference images of shape
                (N, C, H', W'). Stored on CPU, moved to GPU per-batch.
            clip_scale: Maximum gradient norm for clipping. Default: 1.0.
            device: Device for computation.
        """
        super().__init__(device)
        self.operator = operator.to(device).eval()
        self.measurements = measurements.cpu()  # Store on CPU to save VRAM
        self.clip_scale = clip_scale

        for param in self.operator.parameters():
            param.requires_grad = False

    @torch.enable_grad()
    def get_guidance(
        self,
        x: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        return_logp: bool = False,
        check_grad: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Compute inverse problem guidance gradient.

        Energy: log_prob = -||y - A(x)||_2 (per-sample L2 norm).

        Args:
            x: Input images of shape (B, 3, H, W) in [-1, 1] range,
                requiring gradients. Always pixel-space (sampler handles
                latent decode via _decode_latent_for_guidance()).
            targets: Measurement indices of shape (B,). Each value indexes
                into self.measurements to select the corresponding degraded
                reference. MC expansion is handled by the sampler (targets
                already repeated for eps_bsz).
            return_logp: If True, return log-probability instead of gradient.
            check_grad: If True, verify x.requires_grad is set.
            **kwargs: Additional arguments (ignored).

        Returns:
            If return_logp:
                Log-probability tensor of shape (B,).
            Else:
                Gradient tensor of shape (B, 3, H, W).

        Raises:
            ValueError: If targets is None.
        """
        if check_grad:
            check_grad_fn(x)

        if targets is None:
            raise ValueError(
                "targets must be provided for inverse problem guidance. Pass measurement indices as targets."
            )

        y = self.measurements[targets.cpu()].to(x.device)
        diff = y - self.operator(x)
        log_probs = -torch.norm(diff.reshape(diff.shape[0], -1), p=2, dim=1)

        if return_logp:
            return log_probs

        grad = torch.autograd.grad(log_probs.sum(), x)[0]
        return rescale_grad(grad, clip_scale=self.clip_scale, **kwargs)


def create_inverse_guider(
    task: str,
    reference_images: torch.Tensor,
    device: str = "cuda",
    noise_sigma: float = 0.05,
    seed: int = 42,
    clip_scale: float = 1.0,
) -> tuple[InverseProblemGuider, torch.Tensor, torch.Tensor]:
    """Create an inverse problem guider with degraded measurements.

    Factory function that:
    1. Creates the appropriate degradation operator
    2. Applies degradation + Gaussian noise to reference images
    3. Returns the guider, measurements, and degraded images

    Args:
        task: Task name. One of "deblur" or "super_resolution".
        reference_images: Clean reference images of shape (N, 3, H, W)
            in [-1, 1] range.
        device: Device for the guider's operator.
        noise_sigma: Standard deviation of measurement noise. Default: 0.05.
        seed: Random seed for reproducible noise. Default: 42.
        clip_scale: Maximum gradient norm for clipping. Default: 1.0.

    Returns:
        Tuple of:
            - InverseProblemGuider instance
            - measurements: Degraded+noisy images (N, 3, H', W') on CPU
            - degraded_clean: Degraded images without noise (N, 3, H', W') on CPU
              (for visualization/saving)

    Raises:
        ValueError: If task is not recognized.
    """
    if task == "deblur":
        operator = GaussianBlurOperator(kernel_size=61, std=3.0)
    elif task == "super_resolution":
        operator = SuperResolutionOperator(scale_factor=4)
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'deblur' or 'super_resolution'.")

    operator = operator.to(device).eval()

    # Apply degradation in batches to avoid OOM for large reference sets
    batch_size = 64
    degraded_list = []
    with torch.no_grad():
        for i in range(0, len(reference_images), batch_size):
            batch = reference_images[i : i + batch_size].to(device)
            degraded_list.append(operator(batch).cpu())

    degraded_clean = torch.cat(degraded_list, dim=0)

    # Add measurement noise (deterministic)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(degraded_clean.shape, generator=generator) * noise_sigma
    measurements = degraded_clean + noise

    guider = InverseProblemGuider(
        operator=operator,
        measurements=measurements,
        clip_scale=clip_scale,
        device=device,
    )

    return guider, measurements, degraded_clean
