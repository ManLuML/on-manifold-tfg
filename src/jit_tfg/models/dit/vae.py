"""VAE Handler for DiT latent diffusion.

This module provides the VAEHandler class for encoding/decoding between
pixel space and VAE latent space, required for DiT which operates in
the Stable Diffusion VAE latent space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from diffusers import AutoencoderKL


class VAEHandler:
    """Handler for Stable Diffusion VAE operations.

    DiT operates in the latent space of the SD VAE:
    - Pixel space: (B, 3, 256, 256) in [-1, 1]
    - Latent space: (B, 4, 32, 32) scaled by 0.18215

    This class provides:
    1. Encoding: Pixel -> Latent (for inpainting, conditioning)
    2. Decoding: Latent -> Pixel (for final output, classifier guidance)
    3. Differentiable decoding for TFG gradient backpropagation

    Attributes:
        vae: The AutoencoderKL model from HuggingFace diffusers.
        device: Computation device.
        dtype: Data type for computation.

    Example:
        >>> vae = VAEHandler(device="cuda")
        >>> x = torch.randn(1, 3, 256, 256, device="cuda")  # Pixel image
        >>> z = vae.encode(x)  # (1, 4, 32, 32) latent
        >>> x_recon = vae.decode(z)  # (1, 3, 256, 256) reconstruction
    """

    SCALE_FACTOR = 0.18215  # SD VAE scaling factor

    def __init__(
        self,
        vae_type: str = "mse",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize VAE from HuggingFace.

        Args:
            vae_type: VAE variant - "mse" or "ema".
                - "mse": Fine-tuned with MSE loss (better for reconstruction)
                - "ema": EMA version (original SD VAE)
            device: Computation device.
            dtype: Data type for VAE operations.
        """
        from diffusers import AutoencoderKL

        self.device = torch.device(device)
        self.dtype = dtype

        # Load VAE from HuggingFace Hub
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(
            self.device, dtype=self.dtype
        )

        # Freeze VAE parameters
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel images to VAE latents.

        Args:
            x: Images of shape (B, 3, H, W) in [-1, 1] range.
                H, W should be multiples of 8 (typically 256 or 512).

        Returns:
            Latents of shape (B, 4, H//8, W//8).
        """
        x = x.to(self.device, dtype=self.dtype)
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample() * self.SCALE_FACTOR
        return z

    @torch.no_grad()
    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel images to VAE latents (deterministic, using mean).

        Args:
            x: Images of shape (B, 3, H, W) in [-1, 1] range.

        Returns:
            Latents of shape (B, 4, H//8, W//8).
        """
        x = x.to(self.device, dtype=self.dtype)
        posterior = self.vae.encode(x).latent_dist
        z = posterior.mean * self.SCALE_FACTOR
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to pixel images.

        Args:
            z: Latents of shape (B, 4, H//8, W//8).

        Returns:
            Images of shape (B, 3, H, W) in [-1, 1] range.
        """
        z = z.to(self.device, dtype=self.dtype)
        z_scaled = z / self.SCALE_FACTOR
        x = self.vae.decode(z_scaled).sample
        return x

    def decode_with_grad(self, z: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents with gradient tracking.

        Required for TFG guidance where classifier operates in pixel space
        but guidance gradients must flow back to latent space.

        Note:
            VAE parameters remain frozen. Only the computation graph
            from z to the output is preserved for gradient flow.

        Args:
            z: Latents of shape (B, 4, H//8, W//8) with requires_grad=True.

        Returns:
            Images of shape (B, 3, H, W) in [-1, 1] range.
            The returned tensor preserves gradients w.r.t. z.
        """
        z = z.to(self.device, dtype=self.dtype)
        z_scaled = z / self.SCALE_FACTOR
        x = self.vae.decode(z_scaled).sample
        return x

    def to(self, device: str | torch.device) -> VAEHandler:
        """Move VAE to a different device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self.vae = self.vae.to(self.device)
        return self
