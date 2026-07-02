"""DiT Wrapper for JiT-TFG interface compatibility.

This module provides the DiTWrapper class that adapts DiT's interface
to match the JiT-TFG framework's expectations, handling timestep
convention conversion and output extraction.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from jit_tfg.models.dit.model import DiT

# Type alias for CFG channel mode
CFGChannelMode = Literal["all", "first3"]


class DiTWrapper(nn.Module):
    """Wrapper that adapts DiT to the JiT-TFG interface.

    Key responsibilities:
    1. Timestep conversion: Flow matching continuous [0,1] <-> DDPM discrete [0, T-1]
    2. Output extraction: Extract epsilon prediction from DiT output
    3. Interface compatibility: Match forward signature with JiT's Denoiser.net

    Timestep Convention:
        - JiT (Flow Matching): t=0 is pure noise, t=1 is clean data
        - DDPM (DiT): t=0 is clean data, t=T-1 is noisy

        Mapping: t_ddpm = (1 - t_jit) * (T - 1)

    CFG Channel Mode:
        - 'first3': Apply CFG only to first 3 channels, preserve rest
            (original Meta behavior for exact reproducibility with official checkpoints, default)
        - 'all': Apply CFG to all output channels (standard approach)

    Attributes:
        dit: The underlying DiT model.
        num_timesteps: Total number of DDPM timesteps (default: 1000).
        num_classes: Number of classes for conditioning.
        cfg_channel_mode: How to apply CFG across output channels.

    Example:
        >>> dit = DiT_XL_2(input_size=32, num_classes=1000)
        >>> wrapper = DiTWrapper(dit)
        >>> z = torch.randn(2, 4, 32, 32)  # VAE latents
        >>> t = torch.tensor([0.5, 0.5])   # Continuous timestep [0, 1]
        >>> y = torch.tensor([207, 360])   # Class labels
        >>> eps = wrapper(z, t, y)         # Epsilon prediction

        >>> # For exact reproducibility with Meta checkpoints:
        >>> wrapper = DiTWrapper(dit, cfg_channel_mode='first3')
    """

    def __init__(
        self,
        dit_model: DiT,
        num_timesteps: int = 1000,
        cfg_channel_mode: CFGChannelMode = "first3",
    ) -> None:
        """Initialize the wrapper.

        Args:
            dit_model: The DiT model to wrap.
            num_timesteps: Total number of DDPM timesteps.
            cfg_channel_mode: How to apply CFG across channels.
                - 'first3': Apply CFG only to first 3 channels (Meta original, default)
                - 'all': Apply CFG to all 4 channels (alternative)
        """
        super().__init__()
        self.dit = dit_model
        self.num_timesteps = num_timesteps
        self.num_classes = dit_model.num_classes
        self.in_channels = dit_model.in_channels
        self.cfg_channel_mode = cfg_channel_mode

    def t_continuous_to_discrete(self, t: torch.Tensor) -> torch.Tensor:
        """Convert continuous timestep to discrete DDPM timestep.

        The conversion accounts for the opposite time direction:
        - JiT: t=0 is noise, t=1 is clean
        - DDPM: t=0 is clean, t=T-1 is noise

        Args:
            t: Continuous timestep tensor in [0, 1].

        Returns:
            Discrete timestep tensor in [0, T-1] as long tensor.

        Example:
            >>> wrapper.t_continuous_to_discrete(torch.tensor([0.0]))
            tensor([999])  # t=0 (noise in JiT) -> t=999 (noise in DDPM)
            >>> wrapper.t_continuous_to_discrete(torch.tensor([1.0]))
            tensor([0])    # t=1 (clean in JiT) -> t=0 (clean in DDPM)
        """
        # Clamp to valid range [0, 1]
        t = t.clamp(0.0, 1.0)
        # Map: t_jit=0 -> t_ddpm=T-1, t_jit=1 -> t_ddpm=0
        t_discrete = ((1.0 - t) * (self.num_timesteps - 1)).long()
        return t_discrete

    def t_discrete_to_continuous(self, t: torch.Tensor) -> torch.Tensor:
        """Convert discrete DDPM timestep to continuous timestep.

        Args:
            t: Discrete timestep tensor in [0, T-1].

        Returns:
            Continuous timestep tensor in [0, 1].
        """
        return 1.0 - (t.float() / (self.num_timesteps - 1))

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
            Epsilon prediction of shape (B, 4, 32, 32).
        """
        # Handle t shape variations
        if t.ndim > 1:
            t = t.flatten()

        # Convert continuous to discrete timestep
        t_discrete = self.t_continuous_to_discrete(t)

        # Forward through DiT
        output = self.dit(z, t_discrete, y)

        # Extract epsilon prediction (DiT outputs 8 channels when learn_sigma=True)
        eps_pred = output[:, : self.in_channels] if output.shape[1] == self.in_channels * 2 else output

        return eps_pred

    def forward_with_variance(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both epsilon and variance predictions.

        Args:
            z: VAE latents of shape (B, 4, 32, 32).
            t: Continuous timestep of shape (B,) in [0, 1].
            y: Class labels of shape (B,).

        Returns:
            Tuple of (eps_pred, var_pred), each of shape (B, 4, 32, 32).
            var_pred is None if learn_sigma=False.
        """
        if t.ndim > 1:
            t = t.flatten()

        t_discrete = self.t_continuous_to_discrete(t)
        output = self.dit(z, t_discrete, y)

        if output.shape[1] == self.in_channels * 2:
            eps_pred = output[:, : self.in_channels]
            var_pred = output[:, self.in_channels :]
            return eps_pred, var_pred
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
                - 'all': Apply CFG to all channels (standard)
                - 'first3': Apply CFG only to first 3 channels (Meta behavior)

        Returns:
            CFG-guided epsilon prediction of shape (B, 4, 32, 32).
        """
        if t.ndim > 1:
            t = t.flatten()

        mode = cfg_channel_mode or self.cfg_channel_mode

        # Conditional forward
        eps_cond = self(z, t, y)

        # Unconditional forward
        y_uncond = torch.full_like(y, self.num_classes)
        eps_uncond = self(z, t, y_uncond)

        if mode == "first3":
            # Apply CFG only to first 3 channels (original Meta behavior)
            # "For exact reproducibility reasons, we apply CFG on only 3 channels"
            eps_cfg = eps_uncond[:, :3] + cfg_scale * (eps_cond[:, :3] - eps_uncond[:, :3])
            # Use conditional prediction for remaining channels
            eps_guided = torch.cat([eps_cfg, eps_cond[:, 3:]], dim=1)
        else:  # mode == "all"
            # Standard CFG: apply to all channels
            eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        return eps_guided
