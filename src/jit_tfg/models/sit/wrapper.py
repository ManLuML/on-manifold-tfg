"""SiT Wrapper for JiT-TFG interface compatibility.

This module provides the SiTWrapper class that adapts SiT's interface
to match the JiT-TFG framework's expectations.

Key difference from DiTWrapper:
    SiT uses the SAME time convention as JiT (flow matching):
    - t=0 is pure noise
    - t=1 is clean data
    - NO timestep inversion needed!

This makes SiT integration simpler than DiT.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from jit_tfg.models.sit.model import SiT

# Type alias for CFG channel mode
CFGChannelMode = Literal["all", "first3"]


class SiTWrapper(nn.Module):
    """Wrapper that adapts SiT to the JiT-TFG interface.

    Key responsibilities:
    1. Interface compatibility: Match forward signature with JiT's Denoiser.net
    2. CFG handling: Classifier-free guidance computation
    3. Output extraction: Handle learn_sigma outputs

    Unlike DiTWrapper, NO timestep conversion is needed because SiT
    uses the same flow matching convention as JiT:
    - t=0 is pure noise
    - t=1 is clean data

    CFG Channel Mode:
        - 'first3': Apply CFG only to first 3 channels, preserve rest
            (original SiT behavior, default)
        - 'all': Apply CFG to all output channels (for TFG research)

    Attributes:
        sit: The underlying SiT model.
        num_classes: Number of classes for conditioning.
        in_channels: Input channels (4 for VAE latents).
        cfg_channel_mode: How to apply CFG across output channels.

    Example:
        >>> sit = SiT_XL_2(input_size=32, num_classes=1000)
        >>> wrapper = SiTWrapper(sit)
        >>> z = torch.randn(2, 4, 32, 32)  # VAE latents
        >>> t = torch.tensor([0.5, 0.5])   # Continuous timestep [0, 1]
        >>> y = torch.tensor([207, 360])   # Class labels
        >>> v = wrapper(z, t, y)           # Velocity prediction
    """

    def __init__(
        self,
        sit_model: SiT,
        cfg_channel_mode: CFGChannelMode = "first3",
    ) -> None:
        """Initialize the wrapper.

        Args:
            sit_model: The SiT model to wrap.
            cfg_channel_mode: How to apply CFG across channels.
                - 'first3': Apply CFG only to first 3 channels (original SiT, default)
                - 'all': Apply CFG to all 4 channels (for TFG research)
        """
        super().__init__()
        self.sit = sit_model
        self.num_classes = sit_model.num_classes
        self.in_channels = sit_model.in_channels
        self.cfg_channel_mode = cfg_channel_mode

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass matching JiT interface.

        Args:
            z: VAE latents of shape (B, 4, 32, 32).
            t: Continuous timestep of shape (B,) in [0, 1].
                Or can be (B, 1, 1, 1) which will be squeezed.
            y: Class labels of shape (B,) in [0, num_classes-1].
                Use num_classes for unconditional.

        Returns:
            Velocity prediction of shape (B, 4, 32, 32).
        """
        # Handle t shape variations
        if t.ndim > 1:
            t = t.flatten()

        # Forward through SiT (no timestep conversion needed!)
        v_pred = self.sit(z, t, y)

        return v_pred

    def forward_with_variance(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning both velocity and variance predictions.

        Args:
            z: VAE latents of shape (B, 4, H, W).
            t: Continuous timestep of shape (B,) in [0, 1].
            y: Class labels of shape (B,).

        Returns:
            Tuple of (v_pred, var_pred), each of shape (B, 4, H, W).
            var_pred is None if learn_sigma=False.
        """
        if t.ndim > 1:
            t = t.flatten()

        # Get full output with potential variance
        x = self.sit.x_embedder(z) + self.sit.pos_embed
        t_emb = self.sit.t_embedder(t)
        y_emb = self.sit.y_embedder(y, self.sit.training)
        c = t_emb + y_emb

        for block in self.sit.blocks:
            x = block(x, c)
        x = self.sit.final_layer(x, c)
        output = self.sit.unpatchify(x)

        if output.shape[1] == self.in_channels * 2:
            v_pred = output[:, : self.in_channels]
            var_pred = output[:, self.in_channels :]
            return v_pred, var_pred
        else:
            return output, None

    @torch.no_grad()
    def forward_cfg(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float = 1.5,
        cfg_channel_mode: CFGChannelMode | None = None,
    ) -> torch.Tensor:
        """Forward pass with classifier-free guidance.

        Runs conditional and unconditional forward passes and combines
        them using the CFG formula.

        Args:
            z: VAE latents of shape (B, 4, 32, 32).
            t: Continuous timestep of shape (B,) in [0, 1].
            y: Class labels of shape (B,).
            cfg_scale: Guidance scale (1.0 = no guidance).
            cfg_channel_mode: Override instance cfg_channel_mode if provided.
                - 'all': Apply CFG to all channels
                - 'first3': Apply CFG only to first 3 channels (original SiT)

        Returns:
            CFG-guided velocity prediction of shape (B, 4, 32, 32).
        """
        if t.ndim > 1:
            t = t.flatten()

        mode = cfg_channel_mode or self.cfg_channel_mode

        # Conditional forward
        v_cond = self(z, t, y)

        # Unconditional forward (use null class index)
        y_uncond = torch.full_like(y, self.num_classes)
        v_uncond = self(z, t, y_uncond)

        if mode == "first3":
            # Apply CFG only to first 3 channels (original SiT behavior)
            v_cfg = v_uncond[:, :3] + cfg_scale * (v_cond[:, :3] - v_uncond[:, :3])
            # Use conditional prediction for remaining channels
            v_guided = torch.cat([v_cfg, v_cond[:, 3:]], dim=1)
        else:  # mode == "all"
            # Standard CFG: apply to all channels
            v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

        return v_guided
