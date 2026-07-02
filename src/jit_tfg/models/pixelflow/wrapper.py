"""PixelFlow Wrapper for JiT-TFG interface compatibility.

This module provides the PixelFlowWrapper class that adapts PixelFlow's interface
to match the JiT-TFG framework's expectations.

Key characteristics:
    - Same time convention as JiT (t=0: noise, t=1: clean)
    - NO timestep inversion needed (unlike DiT)
    - Pixel space (3, 256, 256) not latent space
    - Uses v-prediction (velocity = x - epsilon)
    - patch_size=4 gives 64x64 patch grid for 256x256 images

This makes PixelFlow integration similar to SiT, but operating in pixel space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from jit_tfg.models.pixelflow.model import PixelFlowModel


class PixelFlowWrapper(nn.Module):
    """Wrapper that adapts PixelFlow to the JiT-TFG interface.

    Key responsibilities:
    1. Interface compatibility: Match forward signature with JiT's Denoiser.net
    2. CFG handling: Classifier-free guidance computation
    3. RoPE embedding: Pre-compute and cache 2D rotary position embeddings

    Unlike DiTWrapper, NO timestep conversion is needed because PixelFlow
    uses the same flow matching convention as JiT:
    - t=0 is pure noise
    - t=1 is clean data

    Unlike SiTWrapper, PixelFlow operates in PIXEL space (3, 256, 256)
    not latent space (4, 32, 32), so no VAE is needed.

    Attributes:
        model: The underlying PixelFlow model.
        num_classes: Number of classes for conditioning (1000 for ImageNet).
        in_channels: Input channels (3 for pixel space).
        img_size: Image size (256).
        patch_size: Patch size (4).
        latent_size: Number of patches per dimension (64 for 256/4).

    Example:
        >>> model = PixelFlowModel(...)
        >>> wrapper = PixelFlowWrapper(model, img_size=256)
        >>> z = torch.randn(2, 3, 256, 256)  # Pixel images
        >>> t = torch.tensor([0.5, 0.5])     # Continuous timestep [0, 1]
        >>> y = torch.tensor([207, 360])     # Class labels
        >>> v = wrapper(z, t, y)             # Velocity prediction
    """

    def __init__(
        self,
        model: PixelFlowModel,
        img_size: int = 256,
    ) -> None:
        """Initialize the wrapper.

        Args:
            model: The PixelFlow model to wrap.
            img_size: Image size (default: 256).
        """
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.patch_size = model.patch_size  # 4
        self.latent_size = img_size // self.patch_size  # 64
        self.num_classes = model.num_classes  # 1000
        self.in_channels = 3  # Pixel space

        # Pre-compute 2D RoPE for fixed resolution
        self._cached_rope: torch.Tensor | None = None

    def _get_rope_embed(self, device: torch.device) -> torch.Tensor:
        """Get 2D RoPE embeddings, computing and caching if needed.

        Args:
            device: Target device.

        Returns:
            RoPE embeddings tensor.
        """
        if self._cached_rope is None or self._cached_rope.device != device:
            self._cached_rope = self._compute_rope_embed(device)
        return self._cached_rope

    def _compute_rope_embed(self, device: torch.device) -> torch.Tensor:
        """Compute 2D RoPE for latent_size x latent_size patches.

        For 256x256 images with patch_size=4, we have 64x64 patches.

        Args:
            device: Target device.

        Returns:
            RoPE embeddings of shape (latent_size*latent_size, attention_head_dim//2, 2).
        """
        from diffusers.models.embeddings import get_2d_rotary_pos_embed

        # Use the new API with output_type='pt' for diffusers >= 0.33.0
        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=self.model.attention_head_dim,
            crops_coords=((0, 0), (self.latent_size, self.latent_size)),
            grid_size=(self.latent_size, self.latent_size),
            output_type="pt",
            device=device,
        )
        # pos_embed is a tuple of (cos, sin), each of shape (H*W, D//2)
        # Stack to (H*W, D//2, 2) for apply_rotary_emb
        return torch.stack(pos_embed, dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning velocity prediction.

        Args:
            z: Pixel images of shape (B, 3, 256, 256).
            t: Continuous timestep of shape (B,) in [0, 1].
                Or can be (B, 1, 1, 1) which will be squeezed.
            y: Class labels of shape (B,) in [0, num_classes-1].
                Use num_classes for unconditional.

        Returns:
            Velocity prediction of shape (B, 3, 256, 256).
        """
        # Handle t shape variations
        if t.ndim > 1:
            t = t.flatten()

        # PixelFlow uses discrete timesteps in [0, 1000)
        # Original training uses t in [0, 1] scaled to [0, 1000)
        timestep = t * 1000.0

        # Fixed latent_size for single resolution (256/4 = 64)
        latent_size = torch.full(
            (z.size(0),),
            self.latent_size,
            dtype=torch.int32,
            device=z.device,
        )

        # Get RoPE embeddings
        pos_embed = self._get_rope_embed(z.device)

        # Forward through PixelFlow
        v_pred = self.model(
            hidden_states=z,
            timestep=timestep,
            class_labels=y,
            latent_size=latent_size,
            pos_embed=pos_embed,
        )

        return v_pred

    def forward_with_variance(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass returning velocity prediction (no variance).

        PixelFlow doesn't predict variance, so this returns None for var_pred.

        Args:
            z: Pixel images of shape (B, 3, H, W).
            t: Continuous timestep of shape (B,) in [0, 1].
            y: Class labels of shape (B,).

        Returns:
            Tuple of (v_pred, None), where v_pred has shape (B, 3, H, W).
        """
        return self.forward(z, t, y), None

    def forward_multires(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        img_h: int,
    ) -> torch.Tensor:
        """Forward pass supporting multi-resolution inputs.

        Used by multi-stage pyramid sampling where resolution changes per stage.

        Args:
            z: Pixel images of shape (B, 3, H, W) where H, W can vary.
            t: Continuous timestep of shape (B,) in [0, 1].
            y: Class labels of shape (B,).
            img_h: Current image height (used to compute latent_size).

        Returns:
            Velocity prediction of shape (B, 3, H, W).
        """
        from diffusers.models.embeddings import get_2d_rotary_pos_embed

        if t.ndim > 1:
            t = t.flatten()

        timestep = t * 1000.0
        latent_size = img_h // self.patch_size

        # Compute RoPE for current resolution
        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=self.model.attention_head_dim,
            crops_coords=((0, 0), (latent_size, latent_size)),
            grid_size=(latent_size, latent_size),
            output_type="pt",
            device=z.device,
        )
        rope_pos = torch.stack(pos_embed, dim=-1)

        size_tensor = torch.tensor([latent_size], dtype=torch.int32, device=z.device)

        v_pred = self.model(
            hidden_states=z,
            timestep=timestep,
            class_labels=y,
            latent_size=size_tensor,
            pos_embed=rope_pos,
        )

        return v_pred

    @torch.no_grad()
    def forward_cfg(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float = 4.0,
    ) -> torch.Tensor:
        """Forward pass with classifier-free guidance.

        Runs conditional and unconditional forward passes and combines
        them using the CFG formula.

        Args:
            z: Pixel images of shape (B, 3, 256, 256).
            t: Continuous timestep of shape (B,) in [0, 1].
            y: Class labels of shape (B,).
            cfg_scale: Guidance scale (1.0 = no guidance).

        Returns:
            CFG-guided velocity prediction of shape (B, 3, 256, 256).
        """
        if t.ndim > 1:
            t = t.flatten()

        # Conditional forward
        v_cond = self(z, t, y)

        # Unconditional forward (use null class index)
        y_uncond = torch.full_like(y, self.num_classes)
        v_uncond = self(z, t, y_uncond)

        # CFG combination
        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

        return v_guided
