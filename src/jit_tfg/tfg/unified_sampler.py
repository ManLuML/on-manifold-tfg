"""Unified TFG Sampler for all diffusion models.

This module provides the UnifiedSampler class that integrates TFG guidance
with all supported diffusion models: JiT, DiT, SiT, and PixelFlow.

The sampler supports:
- CFG-only generation (guidance=None)
- CFG + TFG generation (guidance provided)
- Multiple sampling methods: Euler, Heun (flow matching) and DDIM (DDPM)
- Multiple guidance spaces:
  - x: TFG/DDPM heritage - Position correction (no dt scaling)
  - v: Flow Guidance heritage - Velocity modification (dt implicit)
  - v2: Our extension - Per-NFE velocity modification for Heun (dt implicit)

Usage:
    >>> from jit_tfg.tfg import UnifiedSampler, TFGConfig
    >>> from jit_tfg.tfg.guiders import create_classifier_guider
    >>>
    >>> # Create sampler
    >>> sampler = UnifiedSampler(
    ...     model_type="JiT",
    ...     denoiser=denoiser,
    ...     tfg_config=TFGConfig(rho=1.0, mu=0.5),
    ...     sampling_method="euler",
    ... )
    >>>
    >>> # CFG + TFG generation
    >>> guider = create_classifier_guider(classifier, targets=[207], denoiser=denoiser)
    >>> images = sampler.generate(cfg_labels=labels, guidance=guider)
    >>>
    >>> # CFG-only generation
    >>> images = sampler.generate(cfg_labels=labels, guidance=None)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import grad
from tqdm import tqdm

from jit_tfg.models.dit.diffusion.sampling import (
    get_ddim_timestep_sequence,
    get_ddpm_timestep_sequence,
)
from jit_tfg.tfg.config import TFGConfig
from jit_tfg.tfg.utils import rescale_grad

if TYPE_CHECKING:
    from jit_tfg.tfg.guiders.base import BaseGuider

# Type aliases
ModelType = Literal["JiT", "DiT", "SiT", "PixelFlow"]
SamplingMethod = Literal["euler", "heun", "dopri5", "ddpm", "ddim"]
GuidanceSpace = Literal["x", "v", "v2"]
PredTarget = Literal["x", "v", "e"]


class UnifiedSampler:
    """Unified TFG Sampler for all diffusion models.

    Integrates TFG guidance with JiT, DiT, SiT, and PixelFlow models.
    Supports both CFG-only and CFG+TFG generation modes.

    Attributes:
        model_type: Type of diffusion model ("JiT", "DiT", "SiT", "PixelFlow").
        denoiser: Denoiser model instance.
        tfg_config: TFG configuration (None for CFG-only).
        sampling_method: ODE solver ("euler", "heun", "dopri5").
        guidance_space: Where to apply guidance ("x", "v", "v2").
        pred_target: Network prediction target ("x", "v", "e").
        eta: DDIM stochasticity (DiT only).

    Example:
        >>> # JiT with TFG
        >>> sampler = UnifiedSampler("JiT", denoiser, TFGConfig())
        >>> images = sampler.generate(cfg_labels=labels, guidance=guider)
        >>>
        >>> # DiT CFG-only
        >>> sampler = UnifiedSampler("DiT", denoiser, tfg_config=None)
        >>> images = sampler.generate(cfg_labels=labels)
    """

    def __init__(
        self,
        model_type: ModelType,
        denoiser: nn.Module,
        tfg_config: TFGConfig | None = None,
        sampling_method: SamplingMethod = "euler",
        guidance_space: GuidanceSpace = "x",
        eta: float = 0.0,
    ) -> None:
        """Initialize UnifiedSampler.

        Args:
            model_type: Type of diffusion model.
            denoiser: Denoiser model instance (JiT Denoiser, DiTDenoiser, etc.).
            tfg_config: TFG configuration. If None, only CFG is used.
            sampling_method: Sampling method.
                For flow matching models (JiT, SiT, PixelFlow):
                - "euler": First-order Euler method
                - "heun": Second-order Heun method (improved Euler)
                - "dopri5": Adaptive Runge-Kutta (PixelFlow only, not implemented)
                For DiT:
                - "ddpm": DDPM sampling (original DiT, default)
                - "ddim": DDIM sampling (accelerated)
            guidance_space: Where to apply TFG guidance.
                - "x": TFG/DDPM heritage - Position correction (no dt scaling)
                - "v": Flow Guidance heritage - Velocity modification (dt implicit)
                - "v2": Per-NFE velocity modification (Heun only, dt implicit)
            eta: DDIM stochasticity for DiT DDIM sampling (0=deterministic, 1=stochastic).

        Raises:
            AssertionError: If invalid option combination is provided.
            NotImplementedError: If unimplemented option is requested.
        """
        # Validate model_type
        assert model_type in ["JiT", "DiT", "SiT", "PixelFlow"], (
            f"model_type must be one of ['JiT', 'DiT', 'SiT', 'PixelFlow'], got {model_type}"
        )

        self.model_type = model_type
        self.denoiser = denoiser
        self.tfg_config = tfg_config
        self.eta = eta

        # Get pred_target from denoiser
        self.pred_target: PredTarget = getattr(denoiser, "pred_target", "x")

        # Validate pred_target for each model
        if model_type == "SiT":
            assert self.pred_target == "v", f"SiT must use pred_target='v', got '{self.pred_target}'"
        elif model_type == "PixelFlow":
            assert self.pred_target == "v", f"PixelFlow must use pred_target='v', got '{self.pred_target}'"
        elif model_type == "DiT":
            assert self.pred_target == "e", f"DiT must use pred_target='e', got '{self.pred_target}'"

        # DiT uses DDPM/DDIM, not ODE solvers
        if model_type == "DiT":
            if guidance_space != "x":
                raise ValueError(
                    f"DiT only supports guidance_space='x' (DDPM discrete timesteps), got '{guidance_space}'"
                )
            # Default to DDPM (original DiT), allow DDIM as option
            if sampling_method in ["ddpm", "ddim"]:
                self.sampling_method = sampling_method
            else:
                self.sampling_method = "ddpm"  # Default to original DiT
            self.guidance_space = "x"  # Fixed for DiT
        else:
            # Validate sampling_method
            assert sampling_method in ["euler", "heun", "dopri5"], (
                f"sampling_method must be one of ['euler', 'heun', 'dopri5'], got {sampling_method}"
            )

            # dopri5 is only for PixelFlow and not implemented
            if sampling_method == "dopri5":
                assert model_type == "PixelFlow", "dopri5 sampling is only available for PixelFlow"
                raise NotImplementedError("dopri5 sampling is not yet implemented. Use 'euler' or 'heun' instead.")

            self.sampling_method = sampling_method
            self.guidance_space = guidance_space

            # Validate guidance_space if TFG is enabled
            if tfg_config is not None:
                assert guidance_space in ["x", "v", "v2"], (
                    f"guidance_space must be one of ['x', 'v', 'v2'], got {guidance_space}"
                )

                if guidance_space == "v2":
                    assert sampling_method == "heun", "guidance_space='v2' requires sampling_method='heun'"

        # Get model-specific attributes
        self.num_classes = getattr(denoiser, "num_classes", 1000)
        self.t_eps = getattr(denoiser, "t_eps", 5e-2)
        self.noise_scale = getattr(denoiser, "noise_scale", 1.0)
        self.is_latent = getattr(denoiser, "is_latent_diffusion", False)

        # Initialize random generator for TFG
        if tfg_config is not None:
            self.device = torch.device(tfg_config.device)
            self.generator = torch.Generator(device=self.device).manual_seed(tfg_config.seed)
        else:
            self.device = None
            self.generator = None

        # PixelFlow-specific attributes
        if model_type == "PixelFlow":
            self._setup_pixelflow_attributes()

    def _setup_pixelflow_attributes(self) -> None:
        """Setup PixelFlow-specific attributes for multi-stage sampling."""
        self.num_stages = getattr(self.denoiser, "num_stages", 4)
        self.gamma = getattr(self.denoiser, "gamma", -1 / 3)
        self.img_size = getattr(self.denoiser, "img_size", 256)

        # Stage schedule from denoiser
        self.stage_range = [x / self.num_stages for x in range(self.num_stages + 1)]
        self.original_start_t: dict[int, float] = {}
        self.start_t: dict[int, float] = {}
        self.end_t: dict[int, float] = {}
        self.stage_distance: list[float] = []

        for stage_idx in range(self.num_stages):
            st = self.stage_range[stage_idx]
            et = self.stage_range[stage_idx + 1]
            self.original_start_t[stage_idx] = st
            if stage_idx > 0:
                st *= self._cal_rectify_ratio(st)
            self.start_t[stage_idx] = st
            self.end_t[stage_idx] = et
            self.stage_distance.append(et - st)

        self.total_stage_distance = sum(self.stage_distance)

    def _cal_rectify_ratio(self, start_t: float) -> float:
        """Calculate rectification ratio for PixelFlow stage transition."""
        return 1 / (math.sqrt(1 - (1 / self.gamma)) * (1 - start_t) + start_t)

    def _get_default_num_steps(self) -> int:
        """Get default number of sampling steps for each model."""
        if self.model_type == "JiT":
            return getattr(self.denoiser, "steps", 50)
        elif self.model_type == "DiT":
            return getattr(self.denoiser, "num_sampling_steps", 250)
        elif self.model_type == "SiT":
            return getattr(self.denoiser, "num_sampling_steps", 125)
        elif self.model_type == "PixelFlow":
            return getattr(self.denoiser, "num_sampling_steps", 30)  # per stage, paper default
        else:
            return 50

    def _forward_flow_with_grad(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Grad-enabled forward pass for flow matching models (JiT/SiT).

        Unlike denoiser._forward_sample (which has @torch.no_grad()),
        this function preserves gradient tracking through the model.
        This matches the original TFG implementation where the model
        Jacobian is included in delta_t = grad_{z_t} J(x_hat_0).

        Falls back to a simple net(z, t, y) call if the denoiser does
        not have the expected internal methods (e.g., in test mocks).

        Args:
            z: Current noisy sample (requires_grad=True for TFG).
            t: Timestep tensor of shape (B, 1, 1, 1).
            labels: Class labels of shape (B,).

        Returns:
            CFG-guided velocity prediction (with gradient tracking).
        """
        denoiser = self.denoiser

        if self.model_type == "JiT":
            # Check if denoiser has full JiT interface (real denoiser)
            if hasattr(denoiser, "_convert_prediction") and hasattr(denoiser, "cfg_interval"):
                net_cond = denoiser.net(z, t.flatten(), labels)
                _, v_cond, _ = denoiser._convert_prediction(net_cond, z, t)

                uncond_labels = torch.full_like(labels, denoiser.num_classes)
                net_uncond = denoiser.net(z, t.flatten(), uncond_labels)
                _, v_uncond, _ = denoiser._convert_prediction(net_uncond, z, t)

                low, high = denoiser.cfg_interval
                interval_mask = (t < high) & ((low == 0) | (t > low))
                cfg_scale_interval = torch.where(interval_mask, denoiser.cfg_scale, 1.0)

                return v_uncond + cfg_scale_interval * (v_cond - v_uncond)
            else:
                # Mock/simplified denoiser: just call net directly (with grad)
                t_flat = t.flatten() if t.ndim > 1 else t
                return denoiser.net(z, t_flat, labels)

        elif self.model_type == "SiT":
            # Check if denoiser.net has SiTWrapper interface
            if hasattr(denoiser.net, "num_classes"):
                # Reproduce SiTWrapper.forward_cfg logic without @torch.no_grad()
                t_flat = t.flatten() if t.ndim > 1 else t
                cfg_channel_mode = getattr(denoiser, "cfg_channel_mode", "first3")
                cfg_scale = denoiser.cfg_scale

                # Conditional forward (calls SiTWrapper.__call__ → forward)
                # SiTWrapper.forward() returns a single tensor, not a tuple
                v_cond = denoiser.net(z, t_flat, labels)
                # Unconditional forward
                y_uncond = torch.full_like(labels, denoiser.net.num_classes)
                v_uncond = denoiser.net(z, t_flat, y_uncond)

                if cfg_channel_mode == "first3":
                    v_cfg = v_uncond[:, :3] + cfg_scale * (v_cond[:, :3] - v_uncond[:, :3])
                    v_guided = torch.cat([v_cfg, v_cond[:, 3:]], dim=1)
                else:
                    v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

                return v_guided
            else:
                # Mock/simplified denoiser: just call net directly (with grad)
                t_flat = t.flatten() if t.ndim > 1 else t
                return denoiser.net(z, t_flat, labels)

        else:
            raise ValueError(f"_forward_flow_with_grad is only for JiT/SiT, got {self.model_type}")

    def _get_lambda_t(self, t: float) -> float:
        """Get lambda_t scaling factor for v-space/v2-space guidance.

        Args:
            t: Current timestep in [0, 1].

        Returns:
            Lambda_t value based on lambda_mode config.
        """
        if self.tfg_config is None:
            return 1.0

        mode = self.tfg_config.lambda_mode
        if mode == "auto":
            # x-prediction → identity (already stable gradients)
            # v/e-prediction → flow_guidance (theoretical derivation)
            mode = "identity" if self.pred_target == "x" else "flow_guidance"

        if mode == "flow_guidance":
            t_clamped = max(t, self.t_eps)
            return (1 - t) / t_clamped
        else:  # identity
            return 1.0

    def _get_cfg_scale(self, cfg_scale: float | None) -> float:
        """Get CFG scale from input or denoiser default."""
        if cfg_scale is not None:
            return cfg_scale
        return getattr(self.denoiser, "cfg_scale", 1.5)

    # =========================================================================
    # Main generate() method
    # =========================================================================

    @torch.no_grad()
    def generate(
        self,
        guidance: BaseGuider | None = None,
        tfg_targets: torch.Tensor | None = None,
        cfg_labels: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        batch_size: int | None = None,
        device: str | torch.device | None = None,
        num_steps: int | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate images with optional CFG and TFG guidance.

        Args:
            guidance: TFG guider for training-free guidance. If None, CFG-only.
            tfg_targets: Per-sample TFG target classes of shape (B,).
            cfg_labels: Labels for CFG conditioning. If None, uses null token.
            cfg_scale: CFG scale. If None, uses denoiser default.
            batch_size: Number of images. Required if cfg_labels and tfg_targets are None.
            device: Device for generation. Required if cfg_labels and tfg_targets are None.
            num_steps: Number of sampling steps. If None, uses model default.
            show_progress: Whether to show progress bar.

        Returns:
            Generated images of shape (B, 3, H, W) in [-1, 1] range.
            Always returns pixel-space images (VAE decode applied for latent models).
        """
        # Determine batch_size and device
        if cfg_labels is not None:
            _bsz, _device = cfg_labels.size(0), cfg_labels.device
        elif tfg_targets is not None:
            _bsz, _device = tfg_targets.size(0), tfg_targets.device
        elif batch_size is not None and device is not None:
            _bsz = batch_size
            _device = torch.device(device) if isinstance(device, str) else device
        else:
            raise ValueError(
                "Cannot determine batch size and device. "
                "Provide either 'cfg_labels', 'tfg_targets', or both 'batch_size' and 'device'."
            )

        # Update device for TFG generator if needed
        if self.generator is not None and self.device != _device:
            self.device = _device
            self.generator = torch.Generator(device=_device).manual_seed(
                self.tfg_config.seed if self.tfg_config else 42
            )

        # Get num_steps and cfg_scale
        num_steps = num_steps if num_steps is not None else self._get_default_num_steps()
        cfg_scale = self._get_cfg_scale(cfg_scale)

        # Dispatch to model-specific generation
        if self.model_type == "DiT":
            return self._generate_dit(
                guidance=guidance,
                tfg_targets=tfg_targets,
                cfg_labels=cfg_labels,
                cfg_scale=cfg_scale,
                bsz=_bsz,
                device=_device,
                num_steps=num_steps,
                show_progress=show_progress,
            )
        elif self.model_type == "PixelFlow":
            return self._generate_pixelflow(
                guidance=guidance,
                tfg_targets=tfg_targets,
                cfg_labels=cfg_labels,
                cfg_scale=cfg_scale,
                bsz=_bsz,
                device=_device,
                num_steps=num_steps,
                show_progress=show_progress,
            )
        else:
            # JiT or SiT (flow matching)
            return self._generate_flow_matching(
                guidance=guidance,
                tfg_targets=tfg_targets,
                cfg_labels=cfg_labels,
                cfg_scale=cfg_scale,
                bsz=_bsz,
                device=_device,
                num_steps=num_steps,
                show_progress=show_progress,
            )

    # =========================================================================
    # Flow Matching Generation (JiT, SiT)
    # =========================================================================

    def _generate_flow_matching(
        self,
        guidance: BaseGuider | None,
        tfg_targets: torch.Tensor | None,
        cfg_labels: torch.Tensor | None,
        cfg_scale: float,
        bsz: int,
        device: torch.device,
        num_steps: int,
        show_progress: bool,
    ) -> torch.Tensor:
        """Generate images using flow matching ODE integration.

        Used for JiT and SiT models.
        """
        # Determine spatial size
        if self.model_type == "JiT":
            img_size = getattr(self.denoiser, "img_size", 256)
            channels = 3
        else:  # SiT
            img_size = getattr(self.denoiser, "latent_size", 32)
            channels = 4

        # Initialize from pure noise (t=0)
        z = self.noise_scale * torch.randn(bsz, channels, img_size, img_size, device=device)

        # Timesteps from 0 to 1
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        # Labels for CFG
        labels = cfg_labels if cfg_labels is not None else torch.full((bsz,), self.num_classes, device=device)

        # Temporarily store original cfg_scale and override
        original_cfg_scale = getattr(self.denoiser, "cfg_scale", None)
        if original_cfg_scale is not None:
            self.denoiser.cfg_scale = cfg_scale

        iterator = range(num_steps)
        if show_progress:
            desc = f"{self.model_type} ({self.sampling_method})"
            if guidance is not None:
                desc += " + TFG"
            iterator = tqdm(iterator, desc=desc, total=num_steps)

        try:
            for i in iterator:
                t = timesteps[i].item()
                t_next = timesteps[i + 1].item()

                if self.sampling_method == "euler":
                    z = self._euler_step(z, t, t_next, labels, guidance, tfg_targets)
                elif self.sampling_method == "heun":
                    # Use Euler for last step (standard practice)
                    if i == num_steps - 1:
                        z = self._euler_step(z, t, t_next, labels, guidance, tfg_targets)
                    else:
                        z = self._heun_step(z, t, t_next, labels, guidance, tfg_targets)
        finally:
            # Restore original cfg_scale
            if original_cfg_scale is not None:
                self.denoiser.cfg_scale = original_cfg_scale

        # Decode latents to images if needed
        if self.is_latent:
            with torch.no_grad():
                z = self.denoiser.vae.decode(z)

        return z

    def _euler_step(
        self,
        z: torch.Tensor,
        t: float,
        t_next: float,
        labels: torch.Tensor,
        guidance: BaseGuider | None,
        tfg_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single Euler step with optional TFG guidance."""
        batch_size = z.shape[0]
        device = z.device

        t_tensor = torch.full((batch_size,), t, device=device)
        t_reshaped = t_tensor.view(-1, 1, 1, 1)

        # Get velocity prediction with CFG
        v_pred = self.denoiser._forward_sample(z, t_reshaped, labels)

        # Basic Euler step
        dt = t_next - t
        z_next = z + dt * v_pred

        # TFG guidance
        if guidance is not None and self.tfg_config is not None:
            if self.guidance_space == "x":
                # x-space: TFG/DDPM heritage - Position correction (NO dt)
                #
                # Original TFG (Algorithm 1, Line 9):
                #   x_prev += Δ_t/√α_t + √ᾱ_{t-1}·Δ_0
                # Flow matching adaptation:
                #   z_next += (t_next/t)·δ_t + t_next·δ_0
                #
                # NOTE: Step-count dependent by design (like original TFG).
                # Changing num_steps requires re-tuning rho/mu.
                delta_t, delta_0 = self._compute_flow_guidance_deltas(
                    z=z,
                    v_pred=v_pred,
                    t=t,
                    labels=labels,
                    guidance=guidance,
                    tfg_targets=tfg_targets,
                )
                t_clamped = max(t, self.t_eps)
                z_next = z_next + (t_next / t_clamped) * delta_t + t_next * delta_0
            elif self.guidance_space in ("v", "v2"):
                # v-space: Flow Guidance heritage - Velocity modification (dt implicit)
                # v2-space also uses this path on the last Heun step (Euler fallback),
                # since per-NFE is not applicable with a single function evaluation.
                #
                # Flow Guidance paper g^{cov-G} framework:
                #   g_t = λ_t · ∇_{x_t} J(x̂_1), where λ_t = (1-t)/t
                # Applied as:
                #   v_guided = v + λ_t·δ_t + δ_0
                #   z_next = z + dt · v_guided   (dt implicit through ODE)
                #
                # lambda_t applied to delta_t only, NOT delta_0:
                #   - g^{cov-G} = λ_t · ∇_{x_t} J → corresponds to delta_t
                #   - delta_0 is TFG mean guidance (x_0-space direct shift),
                #     not part of Flow Guidance framework
                delta_t, delta_0 = self._compute_flow_guidance_deltas(
                    z=z,
                    v_pred=v_pred,
                    t=t,
                    labels=labels,
                    guidance=guidance,
                    tfg_targets=tfg_targets,
                )
                lambda_t = self._get_lambda_t(t)

                v_guided = v_pred + lambda_t * delta_t + delta_0
                z_next = z + dt * v_guided

        return z_next

    def _heun_step(
        self,
        z: torch.Tensor,
        t: float,
        t_next: float,
        labels: torch.Tensor,
        guidance: BaseGuider | None,
        tfg_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single Heun step with optional TFG guidance."""
        batch_size = z.shape[0]
        device = z.device
        dt = t_next - t

        t_tensor = torch.full((batch_size,), t, device=device)
        t_next_tensor = torch.full((batch_size,), t_next, device=device)
        t_reshaped = t_tensor.view(-1, 1, 1, 1)
        t_next_reshaped = t_next_tensor.view(-1, 1, 1, 1)

        # First velocity estimate at t
        v1 = self.denoiser._forward_sample(z, t_reshaped, labels)

        # v2-space with guidance: special path where corrector uses GUIDED Euler state.
        # Handled first (early return) to avoid computing an unguided v2 that would be discarded.
        if guidance is not None and self.tfg_config is not None and self.guidance_space == "v2":
            # v2-space: Per-NFE velocity modification (our contribution)
            #
            # Each velocity is guided individually, then combined by Heun.
            # Key difference from v-space:
            #   v-space:  single guidance at t → applied to v_avg
            #   v2-space: separate guidance at t and t_next → average guided velocities
            #
            # Corrector evaluates at GUIDED Euler state, ensuring per-NFE
            # consistency: both v2 and δ_t_2 see the guided intermediate
            # point z + dt·v1_guided (not the unguided z + dt·v1).
            #
            # Uses time-appropriate lambda at each NFE:
            #   v1: λ_t = (1-t)/t
            #   v2: λ_{t_next} = (1-t_next)/t_next  (different from λ_t!)

            # Guidance at v1 (predictor step)
            delta_t_1, delta_0_1 = self._compute_flow_guidance_deltas(
                z=z,
                v_pred=v1,
                t=t,
                labels=labels,
                guidance=guidance,
                tfg_targets=tfg_targets,
            )
            lambda_t = self._get_lambda_t(t)
            v1_guided = v1 + lambda_t * delta_t_1 + delta_0_1

            # Guided Euler step for corrector state
            z_euler_guided = z + dt * v1_guided

            # Second velocity at GUIDED intermediate point
            v2 = self.denoiser._forward_sample(z_euler_guided, t_next_reshaped, labels)

            # Guidance at v2 (corrector step) — evaluated at guided state
            delta_t_2, delta_0_2 = self._compute_flow_guidance_deltas(
                z=z_euler_guided,
                v_pred=v2,
                t=t_next,
                labels=labels,
                guidance=guidance,
                tfg_targets=tfg_targets,
            )
            # NOTE: Use t_next for lambda, not t!
            lambda_t_next = self._get_lambda_t(t_next)
            v2_guided = v2 + lambda_t_next * delta_t_2 + delta_0_2

            # Heun with guided velocities (dt implicit through ODE)
            return z + dt * 0.5 * (v1_guided + v2_guided)

        # Standard path: unguided Euler step for corrector
        z_euler = z + dt * v1
        v2 = self.denoiser._forward_sample(z_euler, t_next_reshaped, labels)
        z_next = z + dt * 0.5 * (v1 + v2)

        # TFG guidance (x-space and v-space)
        if guidance is not None and self.tfg_config is not None:
            if self.guidance_space == "x":
                # x-space: TFG/DDPM heritage - Position correction (NO dt)
                #
                # Original TFG (Algorithm 1, Line 9):
                #   x_prev += Δ_t/√α_t + √ᾱ_{t-1}·Δ_0
                # Flow matching adaptation:
                #   z_next += (t_next/t)·δ_t + t_next·δ_0
                #
                # NOTE: Step-count dependent by design (like original TFG).
                # Changing num_steps requires re-tuning rho/mu.
                delta_t, delta_0 = self._compute_flow_guidance_deltas(
                    z=z,
                    v_pred=v1,  # Use v1 for guidance computation
                    t=t,
                    labels=labels,
                    guidance=guidance,
                    tfg_targets=tfg_targets,
                )
                t_clamped = max(t, self.t_eps)
                z_next = z_next + (t_next / t_clamped) * delta_t + t_next * delta_0
            elif self.guidance_space == "v":
                # v-space: Flow Guidance heritage - Velocity modification (dt implicit)
                #
                # Guidance computed at v1 only (not v2). Rationale:
                #   - v-space applies guidance as a single velocity correction per step
                #   - Per-NFE guidance (separate correction at each velocity) is v2-space
                #   - This matches per-timestep guidance timing (vs per-NFE)
                #
                # lambda_t applied to delta_t only, NOT delta_0:
                #   - g^{cov-G} = λ_t · ∇_{x_t} J → corresponds to delta_t
                #   - delta_0 is TFG mean guidance (x_0-space direct shift),
                #     not part of Flow Guidance framework
                delta_t, delta_0 = self._compute_flow_guidance_deltas(
                    z=z,
                    v_pred=v1,
                    t=t,
                    labels=labels,
                    guidance=guidance,
                    tfg_targets=tfg_targets,
                )
                lambda_t = self._get_lambda_t(t)

                v_avg_guided = (v1 + v2) / 2 + lambda_t * delta_t + delta_0
                z_next = z + dt * v_avg_guided

        return z_next

    @torch.enable_grad()
    def _compute_flow_guidance_deltas(
        self,
        z: torch.Tensor,
        v_pred: torch.Tensor,
        t: float,
        labels: torch.Tensor,
        guidance: BaseGuider,
        tfg_targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute TFG guidance deltas for flow matching models.

        This implements the core TFG algorithm with:
        - Variance guidance (delta_t): gradient w.r.t. z_t
        - Mean guidance (delta_0): gradient w.r.t. x_0 estimate

        Args:
            z: Current noisy state.
            v_pred: Velocity prediction from outer scope. Currently unused
                because autograd requires a fresh forward pass from z_grad.
                Kept for API consistency with _pixelflow_compute_guidance_deltas
                (which uses its v_guided in the rho=0 path).
            t: Current time.
            labels: Class labels for CFG.
            guidance: Guidance function (classifier, etc.).
            tfg_targets: Target labels for guidance.

        Returns:
            Tuple of (delta_t, delta_0).
        """
        config = self.tfg_config
        device = z.device
        batch_size = z.shape[0]

        # Get time-dependent parameters
        rho = self._get_schedule_value(config.rho, config.rho_schedule, t, config.normalize_schedules)
        mu = self._get_schedule_value(config.mu, config.mu_schedule, t, config.normalize_schedules)
        # Sigma is NEVER normalized - it controls Monte Carlo smoothing kernel width,
        # not guidance strength. Original TFG: get_std() has no normalization.
        sigma = self._get_schedule_value(config.sigma, config.sigma_schedule, t, normalize=False)

        t_tensor = torch.full((batch_size,), t, device=device)
        t_reshaped = t_tensor.view(-1, 1, 1, 1)

        z_current = z.detach()

        # Initialize deltas
        delta_t = torch.zeros_like(z)
        delta_0 = torch.zeros_like(z)

        for recur_idx in range(config.recur_steps):
            # Monte Carlo noise for smoothing
            mc_eps = self._get_mc_noise(z.shape, sigma)

            # Variance guidance (delta_t)
            if rho != 0.0:
                z_grad = z_current.clone().detach().requires_grad_(True)

                # Get velocity with gradients through the model.
                # Uses grad-enabled forward (not _forward_sample which has @torch.no_grad())
                # to include model Jacobian in delta_t, matching original TFG.
                v_cfg = self._forward_flow_with_grad(z_grad, t_reshaped, labels)

                # Predict x0 from velocity
                x0 = (1 - t_reshaped) * v_cfg + z_grad

                if config.clip_x0:
                    clip_range = config.clip_sample_range if not self.is_latent else config.clip_sample_range_latent
                    x0 = torch.clamp(x0, -clip_range, clip_range)

                # Compute smoothed log probability
                logprobs = self._compute_smoothed_logprob(x0, mc_eps, guidance, tfg_targets=tfg_targets)
                delta_t = grad(logprobs.sum(), z_grad)[0]
                delta_t = torch.nan_to_num(delta_t, nan=0.0, posinf=0.0, neginf=0.0)
                delta_t = rescale_grad(delta_t, clip_scale=config.clip_scale, mode=config.rescale_mode)
                delta_t = delta_t * rho
            else:
                # Still need x0 for mean guidance
                with torch.no_grad():
                    v_cfg = self.denoiser._forward_sample(z_current, t_reshaped, labels)
                    x0 = (1 - t_reshaped) * v_cfg + z_current

                    if config.clip_x0:
                        clip_range = config.clip_sample_range if not self.is_latent else config.clip_sample_range_latent
                        x0 = torch.clamp(x0, -clip_range, clip_range)

            # Mean guidance (delta_0)
            new_x0 = x0.clone().detach()
            for _ in range(config.iter_steps):
                if mu != 0.0:
                    new_x0_grad = new_x0.detach().requires_grad_(True)
                    guidance_grad = self._compute_smoothed_grad(new_x0_grad, mc_eps, guidance, tfg_targets=tfg_targets)
                    new_x0 = new_x0 + mu * guidance_grad

            delta_0 = new_x0 - x0.detach()

            # Recurrence: complete re-sample (TFG paper original intent)
            # At the same timestep t, re-sample z with new noise realization
            if recur_idx < config.recur_steps - 1:
                with torch.no_grad():
                    v_pred_recur = self.denoiser._forward_sample(z_current, t_reshaped, labels)
                    # x estimate: z_t = t*x + (1-t)*ε → x ≈ z + (1-t)*v
                    x_est = z_current + (1 - t) * v_pred_recur
                    if config.clip_x0:
                        clip_range = config.clip_sample_range if not self.is_latent else config.clip_sample_range_latent
                        x_est = torch.clamp(x_est, -clip_range, clip_range)

                # Complete re-sample with new noise
                new_noise = torch.randn_like(z_current)
                z_current = t * x_est + (1 - t) * new_noise

        return delta_t, delta_0

    # =========================================================================
    # DiT (DDPM) Generation
    # =========================================================================

    def _generate_dit(
        self,
        guidance: BaseGuider | None,
        tfg_targets: torch.Tensor | None,
        cfg_labels: torch.Tensor | None,
        cfg_scale: float,
        bsz: int,
        device: torch.device,
        num_steps: int,
        show_progress: bool,
    ) -> torch.Tensor:
        """Generate images using DDPM or DDIM sampling for DiT."""
        latent_size = getattr(self.denoiser, "latent_size", 32)

        # Initialize from pure noise
        z = torch.randn(bsz, 4, latent_size, latent_size, device=device)

        # Labels for CFG
        labels = cfg_labels if cfg_labels is not None else torch.full((bsz,), self.num_classes, device=device)

        # Get DDPM schedule
        schedule = self.denoiser.schedule

        # Get timestep sequence
        ts = self._get_dit_timestep_sequence(num_steps, device)
        alpha_prod_ts, alpha_prod_t_prevs = self._get_dit_alpha_schedules(ts)

        # Update guider targets if provided
        if guidance is not None and tfg_targets is not None and hasattr(guidance, "targets"):
            guidance.targets = tfg_targets.tolist()

        iterator = range(num_steps)
        if show_progress:
            method_name = "DDPM" if self.sampling_method == "ddpm" else "DDIM"
            desc = f"DiT ({method_name})"
            if guidance is not None:
                desc += " + TFG"
            iterator = tqdm(iterator, desc=desc, total=num_steps)

        for t_idx in iterator:
            if guidance is not None and self.tfg_config is not None:
                # TFG guidance
                if self.sampling_method == "ddpm":
                    z = self._dit_tfg_ddpm_step(
                        z=z,
                        t_idx=t_idx,
                        ts=ts,
                        alpha_prod_ts=alpha_prod_ts,
                        alpha_prod_t_prevs=alpha_prod_t_prevs,
                        labels=labels,
                        cfg_scale=cfg_scale,
                        guidance=guidance,
                        tfg_targets=tfg_targets,
                    )
                else:
                    z = self._dit_tfg_step(
                        z=z,
                        t_idx=t_idx,
                        ts=ts,
                        alpha_prod_ts=alpha_prod_ts,
                        alpha_prod_t_prevs=alpha_prod_t_prevs,
                        labels=labels,
                        cfg_scale=cfg_scale,
                        guidance=guidance,
                        tfg_targets=tfg_targets,
                    )
            else:
                # CFG-only
                if self.sampling_method == "ddpm":
                    z = self._dit_ddpm_step(
                        z=z,
                        t_idx=t_idx,
                        ts=ts,
                        labels=labels,
                        cfg_scale=cfg_scale,
                    )
                else:
                    z = self._dit_ddim_step(
                        z=z,
                        t_idx=t_idx,
                        ts=ts,
                        alpha_prod_ts=alpha_prod_ts,
                        alpha_prod_t_prevs=alpha_prod_t_prevs,
                        labels=labels,
                        cfg_scale=cfg_scale,
                    )

        # Decode latents to images
        images = self.denoiser.vae.decode(z)
        return images

    def _get_dit_timestep_sequence(self, num_steps: int, device: torch.device) -> torch.LongTensor:
        """Get discrete timestep sequence for DiT sampling.

        Dispatches by sampling method:
        - DDPM: Fractional stride matching space_timesteps("N") from original DiT.
          Always includes both endpoints (0 and 999). E.g., 250 steps → [999, 995, ..., 0].
        - DDIM: Uniform integer stride matching space_timesteps("ddimN").
          E.g., 100 steps with 1000 total → range(0,1000,10) → [990, 980, ..., 0].
        """
        total_timesteps = self.denoiser.schedule.num_timesteps
        if self.sampling_method == "ddpm":
            return get_ddpm_timestep_sequence(total_timesteps, num_steps, device)
        else:
            return get_ddim_timestep_sequence(total_timesteps, num_steps, device).long()

    def _get_dit_alpha_schedules(self, ts: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get alpha schedules for DDIM sampling."""
        device = ts.device
        alphas_cumprod = self.denoiser.schedule.alphas_cumprod.to(device)

        alpha_prod_ts = alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.ones_like(alpha_prod_ts)
        alpha_prod_t_prevs[:-1] = alphas_cumprod[ts[1:]]
        alpha_prod_t_prevs[-1] = 1.0

        return alpha_prod_ts, alpha_prod_t_prevs

    def _dit_ddim_step(
        self,
        z: torch.Tensor,
        t_idx: int,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Single DDIM step without TFG guidance."""
        batch_size = z.shape[0]
        device = z.device

        alpha_prod_t = alpha_prod_ts[t_idx]
        alpha_prod_t_prev = alpha_prod_t_prevs[t_idx]
        t = ts[t_idx]

        # Get epsilon with CFG
        eps_pred = self._dit_forward_epsilon_with_cfg(z, t, labels, cfg_scale)

        # Predict x0
        x0 = self._dit_predict_x0(z, eps_pred, alpha_prod_t)

        # DDIM step
        z_next = self._dit_ddim_step_from_x0(z, x0, alpha_prod_t, alpha_prod_t_prev, t.item())

        return z_next

    def _dit_posterior_mean_and_sample(
        self,
        x0: torch.Tensor,
        z: torch.Tensor,
        t: int | torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        model_var_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute DDPM posterior q(x_{t-1}|x_t,x_0) and sample.

        Args:
            x0: Predicted clean sample.
            z: Current noisy sample.
            t: Current discrete timestep (int or scalar tensor).
            alpha_prod_t: Cumulative alpha at t.
            alpha_prod_t_prev: Cumulative alpha at t-1.
            model_var_values: Variance channels (4-7) from model output.
                If provided, uses LEARNED_RANGE interpolation (original DiT).
                If None, uses FIXED_SMALL variance.
        """
        device = z.device
        alpha_t = alpha_prod_t / alpha_prod_t_prev.clamp_min(1e-8)
        beta_t = 1 - alpha_t

        # Posterior variance (FIXED_SMALL)
        posterior_variance = beta_t * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t).clamp_min(1e-8)

        # Posterior mean
        coef1 = beta_t * alpha_prod_t_prev**0.5 / (1 - alpha_prod_t).clamp_min(1e-8)
        coef2 = (1 - alpha_prod_t_prev) * alpha_t**0.5 / (1 - alpha_prod_t).clamp_min(1e-8)
        posterior_mean = coef1 * x0 + coef2 * z

        if model_var_values is not None:
            # LEARNED_RANGE interpolation (matches original SpacedDiffusion).
            # Use rebased beta_t / posterior_variance computed on-the-fly above,
            # NOT the original 1000-step schedule values.  SpacedDiffusion creates
            # new betas for the selected timesteps; each rebased beta spans
            # multiple original timesteps and is ~(stride)x larger.
            min_log = torch.log(posterior_variance.clamp_min(1e-20))
            max_log = torch.log(beta_t.clamp_min(1e-20))
            frac = (model_var_values + 1) / 2
            posterior_log_variance = frac * max_log + (1 - frac) * min_log
        else:
            # FIXED_SMALL fallback
            posterior_log_variance = torch.log(posterior_variance.clamp_min(1e-20))

        # Sample x_{t-1}
        t_val = t if isinstance(t, int) else t.item()
        if t_val > 0:
            if self.generator is not None:
                noise = torch.randn(z.shape, device=device, generator=self.generator)
            else:
                noise = torch.randn_like(z)
        else:
            noise = torch.zeros_like(z)
        return posterior_mean + (0.5 * posterior_log_variance).exp() * noise

    def _dit_ddpm_step(
        self,
        z: torch.Tensor,
        t_idx: int,
        ts: torch.LongTensor,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Single DDPM step without TFG guidance (original DiT sampling)."""
        device = z.device
        schedule = self.denoiser.schedule
        t = ts[t_idx]

        # Get alpha schedule values
        alphas_cumprod = schedule.alphas_cumprod.to(device)
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[ts[t_idx + 1]] if t_idx < len(ts) - 1 else torch.tensor(1.0, device=device)

        # Get epsilon (and variance channels) with CFG
        eps_pred, model_var_values = self._dit_forward_with_cfg(z, t, labels, cfg_scale)

        # Predict x0 from epsilon
        x0 = self._dit_predict_x0(z, eps_pred, alpha_prod_t)

        return self._dit_posterior_mean_and_sample(
            x0=x0,
            z=z,
            t=t,
            alpha_prod_t=alpha_prod_t,
            alpha_prod_t_prev=alpha_prod_t_prev,
            model_var_values=model_var_values,
        )

    @torch.enable_grad()
    def _dit_tfg_step(
        self,
        z: torch.Tensor,
        t_idx: int,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
        guidance: BaseGuider,
        tfg_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single DDIM step with TFG guidance (DiT).

        Implements TFG (Training-Free Guidance) with DDIM sampling.
        Following TFG paper (Algorithm 1, line 8), guidance is added AFTER
        the sampling step with specific coefficients:

            x_{t-1} = DDIM_sample(z_t, x_0) + Δ_t/√α_t + √ᾱ_{t-1} × Δ_0

        where:
            - Δ_t: variance guidance (gradient w.r.t. z_t), scaled by 1/√α_t
            - Δ_0: mean guidance (gradient w.r.t. x_0), scaled by √ᾱ_{t-1}

        These coefficients ensure guidance magnitude is preserved correctly
        across the sampling step, matching the theoretical analysis in TFG
        paper Lemma 3.3.
        """
        config = self.tfg_config

        alpha_prod_t = alpha_prod_ts[t_idx]
        alpha_prod_t_prev = alpha_prod_t_prevs[t_idx]
        alpha_t = alpha_prod_t / alpha_prod_t_prev.clamp_min(1e-8)
        t = ts[t_idx]

        # Get time-dependent TFG parameters
        rho = self._get_dit_rho(t_idx, alpha_prod_ts, alpha_prod_t_prevs)
        mu = self._get_dit_mu(t_idx, alpha_prod_ts, alpha_prod_t_prevs)
        sigma = self._get_dit_sigma(t_idx, alpha_prod_ts)

        for _recur_step in range(config.recur_steps):
            mc_eps = self._get_mc_noise(z.shape, sigma)

            # Variance guidance: Δ_t = ρ × ∇_{z_t} log f̃(x_{0|t})
            if rho != 0.0:
                z_g = z.clone().detach().requires_grad_(True)
                eps_pred = self._dit_forward_epsilon_with_cfg(z_g, t, labels, cfg_scale)
                x0 = self._dit_predict_x0(z_g, eps_pred, alpha_prod_t)

                logprobs = self._compute_smoothed_logprob(x0, mc_eps, guidance, tfg_targets=tfg_targets)
                delta_t = grad(logprobs.sum(), z_g)[0]
                delta_t = torch.nan_to_num(delta_t, nan=0.0, posinf=0.0, neginf=0.0)
                delta_t = rescale_grad(delta_t, clip_scale=config.clip_scale, mode=config.rescale_mode)
                delta_t = delta_t * rho
            else:
                delta_t = torch.zeros_like(z)
                eps_pred = self._dit_forward_epsilon_with_cfg(z, t, labels, cfg_scale)
                x0 = self._dit_predict_x0(z, eps_pred, alpha_prod_t)

            # Mean guidance: Δ_0 = Σ_{i=1}^{N_iter} μ × ∇_{x_0} log f̃(x_0)
            new_x0 = x0.clone().detach()
            for _ in range(config.iter_steps):
                if mu != 0.0:
                    new_x0_grad = new_x0.detach().requires_grad_(True)
                    guidance_grad = self._compute_smoothed_grad(new_x0_grad, mc_eps, guidance, tfg_targets=tfg_targets)
                    new_x0 = new_x0 + mu * guidance_grad

            delta_0 = new_x0 - x0.detach()

            # DDIM sampling step (without guidance)
            x_prev = self._dit_ddim_step_from_x0(z, x0.detach(), alpha_prod_t, alpha_prod_t_prev, t.item())

            # Apply TFG guidance (TFG paper Algorithm 1, line 8)
            # x_{t-1} += Δ_t/√α_t + √ᾱ_{t-1} × Δ_0
            x_prev = x_prev + delta_t / alpha_t**0.5 + delta_0 * alpha_prod_t_prev**0.5

            # Recurrence: add noise back to get z_t from z_{t-1}
            if _recur_step < config.recur_steps - 1:
                z = self._dit_predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev)
            else:
                z = x_prev

        return x_prev

    @torch.enable_grad()
    def _dit_tfg_ddpm_step(
        self,
        z: torch.Tensor,
        t_idx: int,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
        guidance: BaseGuider,
        tfg_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single DDPM step with TFG guidance (DiT).

        Implements TFG (Training-Free Guidance) with DDPM posterior sampling.
        Following TFG paper (Algorithm 1, line 8), guidance is added AFTER
        the sampling step with specific coefficients:

            x_{t-1} = DDPM_sample(z_t, x_0) + Δ_t/√α_t + √ᾱ_{t-1} × Δ_0

        where:
            - Δ_t: variance guidance (gradient w.r.t. z_t), scaled by 1/√α_t
            - Δ_0: mean guidance (gradient w.r.t. x_0), scaled by √ᾱ_{t-1}

        These coefficients ensure guidance magnitude is preserved correctly
        across the sampling step, following the TFG paper's formula exactly.

        Note:
            The TFG guidance formula was originally derived for DDIM sampling
            (Lemma 3.3). The original TFG implementation uses DDIM with eta
            parameter for stochasticity, not true DDPM posterior sampling.

            This implementation uses true DDPM posterior (matching Meta's
            original DiT) with the same TFG guidance coefficients. While not
            rigorously derived for DDPM, this approach:
            - Preserves TFG hyperparameter (rho, mu) compatibility
            - Maintains consistent guidance magnitude with DDIM
            - Is numerically stable across all timesteps

            The guidance formula itself is identical to _dit_tfg_step (DDIM).
        """
        config = self.tfg_config
        device = z.device
        t = ts[t_idx]

        # Get alpha schedule values from precomputed schedules
        alpha_prod_t = alpha_prod_ts[t_idx]
        alpha_prod_t_prev = alpha_prod_t_prevs[t_idx]
        alpha_t = alpha_prod_t / alpha_prod_t_prev.clamp_min(1e-8)

        # Get time-dependent TFG parameters using DDPM schedule
        rho = self._get_dit_rho(t_idx, alpha_prod_ts, alpha_prod_t_prevs)
        mu = self._get_dit_mu(t_idx, alpha_prod_ts, alpha_prod_t_prevs)
        sigma = self._get_dit_sigma(t_idx, alpha_prod_ts)

        for _recur_step in range(config.recur_steps):
            mc_eps = self._get_mc_noise(z.shape, sigma)

            # Variance guidance: Δ_t = ρ × ∇_{z_t} log f̃(x_{0|t})
            if rho != 0.0:
                z_g = z.clone().detach().requires_grad_(True)
                eps_pred, model_var_values = self._dit_forward_with_cfg(z_g, t, labels, cfg_scale)
                x0 = self._dit_predict_x0(z_g, eps_pred, alpha_prod_t)

                logprobs = self._compute_smoothed_logprob(x0, mc_eps, guidance, tfg_targets=tfg_targets)
                delta_t = grad(logprobs.sum(), z_g)[0]
                delta_t = torch.nan_to_num(delta_t, nan=0.0, posinf=0.0, neginf=0.0)
                delta_t = rescale_grad(delta_t, clip_scale=config.clip_scale, mode=config.rescale_mode)
                delta_t = delta_t * rho
            else:
                delta_t = torch.zeros_like(z)
                eps_pred, model_var_values = self._dit_forward_with_cfg(z, t, labels, cfg_scale)
                x0 = self._dit_predict_x0(z, eps_pred, alpha_prod_t)

            # Mean guidance: Δ_0 = Σ_{i=1}^{N_iter} μ × ∇_{x_0} log f̃(x_0)
            new_x0 = x0.clone().detach()
            for _ in range(config.iter_steps):
                if mu != 0.0:
                    new_x0_grad = new_x0.detach().requires_grad_(True)
                    guidance_grad = self._compute_smoothed_grad(new_x0_grad, mc_eps, guidance, tfg_targets=tfg_targets)
                    new_x0 = new_x0 + mu * guidance_grad

            delta_0 = new_x0 - x0.detach()

            # DDPM posterior sampling (without guidance) using learned variance
            x_prev = self._dit_posterior_mean_and_sample(
                x0=x0.detach(),
                z=z,
                t=t,
                alpha_prod_t=alpha_prod_t,
                alpha_prod_t_prev=alpha_prod_t_prev,
                model_var_values=model_var_values.detach() if model_var_values is not None else None,
            )

            # Apply TFG guidance (TFG paper Algorithm 1, line 8)
            # x_{t-1} += Δ_t/√α_t + √ᾱ_{t-1} × Δ_0
            x_prev = x_prev + delta_t / alpha_t**0.5 + delta_0 * alpha_prod_t_prev**0.5

            # Recurrence: add noise back to get z_t from z_{t-1}
            if _recur_step < config.recur_steps - 1:
                z = self._dit_predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev)
            else:
                z = x_prev

        return x_prev

    def _dit_forward_with_cfg(
        self,
        z: torch.Tensor,
        t: torch.Tensor | int,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning (eps_guided, model_var_values).

        model_var_values are channels [in_channels:] from the CONDITIONAL model output.
        CFG is NOT applied to variance channels (matches original forward_with_cfg).

        Returns:
            Tuple of:
            - eps_guided: CFG-guided epsilon prediction (B, in_channels, H, W).
            - model_var_values: Variance channels from conditional output
              (B, in_channels, H, W), or None if model doesn't output variance.
        """
        batch_size = z.shape[0]
        device = z.device

        if isinstance(t, int):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        else:
            t_tensor = t.expand(batch_size) if t.ndim == 0 else t

        dit = self.denoiser.net.dit
        in_channels = self.denoiser.net.in_channels

        # Conditional
        model_out_cond = dit(z, t_tensor, labels)

        # Unconditional
        y_uncond = torch.full((batch_size,), self.num_classes, device=device, dtype=torch.long)
        model_out_uncond = dit(z, t_tensor, y_uncond)

        # Get CFG channel mode from denoiser (DiT-specific setting)
        cfg_channel_mode = getattr(self.denoiser, "cfg_channel_mode", "first3")

        # Apply CFG based on channel mode (to eps channels only)
        if cfg_channel_mode == "first3":
            eps_guided = model_out_cond[:, :in_channels].clone()
            eps_guided[:, :3] = model_out_uncond[:, :3] + cfg_scale * (model_out_cond[:, :3] - model_out_uncond[:, :3])
        else:
            eps_guided = model_out_uncond[:, :in_channels] + cfg_scale * (
                model_out_cond[:, :in_channels] - model_out_uncond[:, :in_channels]
            )

        # Extract variance channels from conditional output (no CFG applied)
        model_var_values = model_out_cond[:, in_channels:] if model_out_cond.shape[1] > in_channels else None

        return eps_guided, model_var_values

    def _dit_forward_epsilon_with_cfg(
        self,
        z: torch.Tensor,
        t: torch.Tensor | int,
        labels: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Get epsilon prediction with CFG for DiT.

        The CFG channel mode is controlled by denoiser.cfg_channel_mode:
        - "first3": Apply CFG only to first 3 channels (original DiT behavior).
        - "all": Apply CFG to all 4 channels.
        """
        eps, _ = self._dit_forward_with_cfg(z, t, labels, cfg_scale)
        return eps

    def _dit_predict_x0(
        self,
        z: torch.Tensor,
        eps: torch.Tensor,
        alpha_prod_t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x0 from z and epsilon for DiT."""
        sqrt_alpha = alpha_prod_t**0.5
        sqrt_one_minus_alpha = (1 - alpha_prod_t) ** 0.5

        x0 = (z - sqrt_one_minus_alpha * eps) / sqrt_alpha.clamp_min(1e-8)

        if self.tfg_config and self.tfg_config.clip_x0:
            clip_range = self.tfg_config.clip_sample_range_latent
            x0 = torch.clamp(x0, -clip_range, clip_range)

        return x0

    def _dit_ddim_step_from_x0(
        self,
        z: torch.Tensor,
        x0: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """DDIM step from x0 prediction."""
        sqrt_alpha = alpha_prod_t**0.5
        sqrt_one_minus_alpha = (1 - alpha_prod_t) ** 0.5

        # Compute new epsilon from x0
        new_epsilon = (z - sqrt_alpha * x0) / sqrt_one_minus_alpha.clamp_min(1e-8)

        # DDIM sigma
        eta = self.eta
        sigma = (
            eta
            * (
                (1 - alpha_prod_t_prev) / (1 - alpha_prod_t).clamp_min(1e-8) * (1 - alpha_prod_t / alpha_prod_t_prev)
            ).clamp_min(0)
            ** 0.5
        )

        # DDIM formula
        sqrt_alpha_prev = alpha_prod_t_prev**0.5
        sqrt_one_minus_alpha_prev_minus_sigma2 = (1 - alpha_prod_t_prev - sigma**2).clamp_min(0) ** 0.5

        pred_sample = sqrt_alpha_prev * x0 + sqrt_one_minus_alpha_prev_minus_sigma2 * new_epsilon

        # Add noise if eta > 0 and not final step
        if eta > 0 and t > 0:
            noise = torch.randn_like(pred_sample)
            if self.generator is not None:
                noise = torch.randn(pred_sample.shape, device=pred_sample.device, generator=self.generator)
            pred_sample = pred_sample + sigma * noise

        return pred_sample

    def _dit_predict_xt(
        self,
        x_prev: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_t from x_{t-1} for recurrence."""
        alpha_ratio = alpha_prod_t / alpha_prod_t_prev.clamp_min(1e-8)
        xt_mean = alpha_ratio**0.5 * x_prev
        noise_std = (1 - alpha_ratio).clamp_min(0) ** 0.5

        noise = torch.randn_like(x_prev)
        if self.generator is not None:
            noise = torch.randn(x_prev.shape, device=x_prev.device, generator=self.generator)

        return xt_mean + noise_std * noise

    def _get_dit_rho(self, t_idx: int, alpha_prod_ts: torch.Tensor, alpha_prod_t_prevs: torch.Tensor) -> float:
        """Get rho value for DiT following original TFG schedule.

        Note: DiT always normalizes schedules (ignores config.normalize_schedules)
        because DDPM's discrete timestep nature requires normalization for
        consistency with the original TFG paper's formula: rho * s[t] * len(s) / sum(s).
        """
        config = self.tfg_config
        schedule = config.rho_schedule
        alpha_ratio = alpha_prod_ts / alpha_prod_t_prevs

        if schedule == "decrease":
            scheduler = 1 - alpha_ratio
        elif schedule == "increase":
            scheduler = alpha_ratio
        elif schedule == "constant":
            scheduler = torch.ones_like(alpha_prod_ts)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        normalizer = len(scheduler) / scheduler.sum().clamp_min(1e-8)
        return float(config.rho * scheduler[t_idx].item() * normalizer.item())

    def _get_dit_mu(self, t_idx: int, alpha_prod_ts: torch.Tensor, alpha_prod_t_prevs: torch.Tensor) -> float:
        """Get mu value for DiT following original TFG schedule.

        Note: DiT always normalizes schedules (ignores config.normalize_schedules)
        because DDPM's discrete timestep nature requires normalization for
        consistency with the original TFG paper's formula: mu * s[t] * len(s) / sum(s).
        """
        config = self.tfg_config
        schedule = config.mu_schedule
        alpha_ratio = alpha_prod_ts / alpha_prod_t_prevs

        if schedule == "decrease":
            scheduler = 1 - alpha_ratio
        elif schedule == "increase":
            scheduler = alpha_ratio
        elif schedule == "constant":
            scheduler = torch.ones_like(alpha_prod_ts)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        normalizer = len(scheduler) / scheduler.sum().clamp_min(1e-8)
        return float(config.mu * scheduler[t_idx].item() * normalizer.item())

    def _get_dit_sigma(self, t_idx: int, alpha_prod_ts: torch.Tensor) -> float:
        """Get sigma value for DiT.

        Note: Sigma is NEVER normalized (unlike rho and mu). This matches the
        original TFG implementation where sigma controls Monte Carlo smoothing
        kernel width (a precision parameter), not guidance strength.
        Formula: sigma * scheduler[t] (no len(s)/sum(s) normalization).
        """
        config = self.tfg_config
        schedule = config.sigma_schedule

        if schedule == "decrease":
            scheduler = (1 - alpha_prod_ts) ** 0.5
        elif schedule == "constant":
            scheduler = torch.ones_like(alpha_prod_ts)
        else:
            scheduler = alpha_prod_ts**0.5

        return float(config.sigma * scheduler[t_idx].item())

    # =========================================================================
    # PixelFlow Multi-stage Generation
    # =========================================================================

    def _generate_pixelflow(
        self,
        guidance: BaseGuider | None,
        tfg_targets: torch.Tensor | None,
        cfg_labels: torch.Tensor | None,
        cfg_scale: float,
        bsz: int,
        device: torch.device,
        num_steps: int,
        show_progress: bool,
    ) -> torch.Tensor:
        """Generate images using multi-stage pyramid sampling for PixelFlow."""
        # Initialize at lowest resolution
        init_factor = 2 ** (self.num_stages - 1)
        h = w = self.img_size // init_factor
        latents = torch.randn(bsz, 3, h, w, device=device)

        # Labels for CFG
        labels = cfg_labels if cfg_labels is not None else torch.full((bsz,), self.num_classes, device=device)
        labels_cfg = torch.cat([torch.full_like(labels, self.num_classes), labels], dim=0)

        total_steps = num_steps * self.num_stages
        pbar = tqdm(total=total_steps, desc="PixelFlow", disable=not show_progress)
        if guidance is not None:
            pbar.set_description("PixelFlow + TFG")

        for stage_idx in range(self.num_stages):
            if stage_idx > 0:
                # Upsample
                h, w = h * 2, w * 2
                latents = F.interpolate(latents, size=(h, w), mode="nearest")

                # Block noise injection
                orig_st = self.original_start_t[stage_idx]
                alpha = self._cal_rectify_ratio(orig_st)
                beta = alpha * (1 - orig_st) / math.sqrt(-self.gamma)
                noise = self._sample_block_noise(bsz, 3, h, w, device, latents.dtype)
                latents = alpha * latents + beta * noise

            # Stage-dependent CFG scale
            scale_dict = {0: 0, 1: 1 / 6, 2: 2 / 3, 3: 1}
            stage_cfg = (cfg_scale - 1) * scale_dict.get(stage_idx, 1) + 1

            # Timesteps — match original PixelFlowScheduler exactly
            # Original computes Timesteps_per_stage with linspace(..., 1001)[:-1] for non-last
            # stages, making stage_T_end slightly less than the integer endpoint. Then
            # t ∈ [0, 0.999] is linearly mapped to T ∈ [stage_T_start, stage_T_end].
            _NUM_TRAIN_TIMESTEPS = 1000
            start_ratio = 0.0 if stage_idx == 0 else sum(self.stage_distance[:stage_idx]) / self.total_stage_distance
            end_ratio = (
                1.0
                if stage_idx == self.num_stages - 1
                else sum(self.stage_distance[: stage_idx + 1]) / self.total_stage_distance
            )
            T_start_idx = int(_NUM_TRAIN_TIMESTEPS * start_ratio)
            T_end_idx = min(int(_NUM_TRAIN_TIMESTEPS * end_ratio), _NUM_TRAIN_TIMESTEPS - 1)

            if stage_idx == self.num_stages - 1:
                stage_T_start = float(T_start_idx)
                stage_T_end = float(T_end_idx)
            else:
                stage_T_start = float(T_start_idx)
                stage_T_end = T_end_idx - (T_end_idx - T_start_idx) / _NUM_TRAIN_TIMESTEPS

            # Within-stage time: linspace(0, 999/1000, num_steps) + append 1.0 (matching original)
            _t_end = (_NUM_TRAIN_TIMESTEPS - 1) / _NUM_TRAIN_TIMESTEPS  # 0.999
            t_schedule = np.linspace(0, _t_end, num_steps, dtype=np.float64)
            t_within_stage = np.append(t_schedule, 1.0)
            # Linear map: T = stage_T_start + (t / 0.999) * (stage_T_end - stage_T_start)
            Timesteps = stage_T_start + (t_schedule / _t_end) * (stage_T_end - stage_T_start)

            for i in range(num_steps):
                t = t_within_stage[i]
                t_next = t_within_stage[i + 1]
                T = Timesteps[i]

                if self.sampling_method == "euler":
                    latents = self._pixelflow_euler_step(
                        z=latents,
                        t=t,
                        t_next=t_next,
                        T=T,
                        h=h,
                        labels=labels,
                        labels_cfg=labels_cfg,
                        stage_cfg=stage_cfg,
                        guidance=guidance,
                        tfg_targets=tfg_targets,
                    )
                elif self.sampling_method == "heun":
                    # Use Euler for last step of last stage
                    if stage_idx == self.num_stages - 1 and i == num_steps - 1:
                        latents = self._pixelflow_euler_step(
                            z=latents,
                            t=t,
                            t_next=t_next,
                            T=T,
                            h=h,
                            labels=labels,
                            labels_cfg=labels_cfg,
                            stage_cfg=stage_cfg,
                            guidance=guidance,
                            tfg_targets=tfg_targets,
                        )
                    else:
                        T_next = Timesteps[i + 1] if i + 1 < num_steps else stage_T_end
                        latents = self._pixelflow_heun_step(
                            z=latents,
                            t=t,
                            t_next=t_next,
                            T=T,
                            T_next=T_next,
                            h=h,
                            labels=labels,
                            labels_cfg=labels_cfg,
                            stage_cfg=stage_cfg,
                            guidance=guidance,
                            tfg_targets=tfg_targets,
                        )

                pbar.update(1)

        pbar.close()
        return latents

    def _pixelflow_euler_step(
        self,
        z: torch.Tensor,
        t: float,
        t_next: float,
        T: float,
        h: int,
        labels: torch.Tensor,
        labels_cfg: torch.Tensor,
        stage_cfg: float,
        guidance: BaseGuider | None,
        tfg_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single Euler step for PixelFlow with optional TFG."""
        bsz = z.size(0)
        device = z.device
        dt = t_next - t

        # Forward with CFG
        z_cfg = torch.cat([z, z], dim=0)
        t_tensor = torch.full((bsz * 2,), T / 1000.0, device=device)
        v_pred = self.denoiser.net.forward_multires(z_cfg, t_tensor, labels_cfg, h)

        v_uncond, v_cond = v_pred[:bsz], v_pred[bsz:]
        v_guided = v_uncond + stage_cfg * (v_cond - v_uncond)

        # Euler step
        z_next = z + dt * v_guided

        # TFG guidance
        if guidance is not None and self.tfg_config is not None:
            if self.guidance_space == "x":
                # x-space: TFG/DDPM heritage - Position correction (NO dt)
                delta_t, delta_0 = self._pixelflow_compute_guidance_deltas(
                    z=z,
                    v_guided=v_guided,
                    t=t,
                    T=T,
                    h=h,
                    labels=labels,
                    labels_cfg=labels_cfg,
                    stage_cfg=stage_cfg,
                    guidance=guidance,
                    tfg_targets=tfg_targets,
                )
                t_clamped = max(t, self.t_eps)
                z_next = z_next + (t_next / t_clamped) * delta_t + t_next * delta_0
            elif self.guidance_space in ("v", "v2"):
                # v-space (and v2 Euler fallback on last Heun step):
                # Flow Guidance heritage - Velocity modification (dt implicit)
                delta_t, delta_0 = self._pixelflow_compute_guidance_deltas(
                    z=z,
                    v_guided=v_guided,
                    t=t,
                    T=T,
                    h=h,
                    labels=labels,
                    labels_cfg=labels_cfg,
                    stage_cfg=stage_cfg,
                    guidance=guidance,
                    tfg_targets=tfg_targets,
                )
                # Theoretical scaling: lambda_t based on lambda_mode config
                lambda_t = self._get_lambda_t(t)

                # Modify velocity directly and recompute z_next (dt implicit through ODE)
                v_with_guidance = v_guided + lambda_t * delta_t + delta_0
                z_next = z + dt * v_with_guidance

        return z_next

    def _pixelflow_heun_step(
        self,
        z: torch.Tensor,
        t: float,
        t_next: float,
        T: float,
        T_next: float,
        h: int,
        labels: torch.Tensor,
        labels_cfg: torch.Tensor,
        stage_cfg: float,
        guidance: BaseGuider | None,
        tfg_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single Heun step for PixelFlow with optional TFG."""
        bsz = z.size(0)
        device = z.device
        dt = t_next - t

        # First velocity at t
        z_cfg = torch.cat([z, z], dim=0)
        t_tensor = torch.full((bsz * 2,), T / 1000.0, device=device)
        v_pred1 = self.denoiser.net.forward_multires(z_cfg, t_tensor, labels_cfg, h)
        v_uncond1, v_cond1 = v_pred1[:bsz], v_pred1[bsz:]
        v1 = v_uncond1 + stage_cfg * (v_cond1 - v_uncond1)

        # v2-space with guidance: special path where corrector uses GUIDED Euler state.
        # Handled first (early return) to avoid computing an unguided v2 that would be discarded.
        if guidance is not None and self.tfg_config is not None and self.guidance_space == "v2":
            # v2-space: Per-NFE velocity modification
            #
            # Corrector evaluates at GUIDED Euler state, ensuring per-NFE
            # consistency: both v2 and δ_t_2 see the guided intermediate
            # point z + dt·v1_guided (not the unguided z + dt·v1).

            # Guidance at v1 (predictor step)
            delta_t_1, delta_0_1 = self._pixelflow_compute_guidance_deltas(
                z=z,
                v_guided=v1,
                t=t,
                T=T,
                h=h,
                labels=labels,
                labels_cfg=labels_cfg,
                stage_cfg=stage_cfg,
                guidance=guidance,
                tfg_targets=tfg_targets,
            )
            lambda_t = self._get_lambda_t(t)
            v1_guided = v1 + lambda_t * delta_t_1 + delta_0_1

            # Guided Euler step for corrector state
            z_euler_guided = z + dt * v1_guided

            # Second velocity at GUIDED intermediate point
            z_cfg_next = torch.cat([z_euler_guided, z_euler_guided], dim=0)
            t_tensor_next = torch.full((bsz * 2,), T_next / 1000.0, device=device)
            v_pred2 = self.denoiser.net.forward_multires(z_cfg_next, t_tensor_next, labels_cfg, h)
            v_uncond2, v_cond2 = v_pred2[:bsz], v_pred2[bsz:]
            v2 = v_uncond2 + stage_cfg * (v_cond2 - v_uncond2)

            # Guidance at v2 (corrector step) — evaluated at guided state
            delta_t_2, delta_0_2 = self._pixelflow_compute_guidance_deltas(
                z=z_euler_guided,
                v_guided=v2,
                t=t_next,
                T=T_next,
                h=h,
                labels=labels,
                labels_cfg=labels_cfg,
                stage_cfg=stage_cfg,
                guidance=guidance,
                tfg_targets=tfg_targets,
            )
            # NOTE: Use t_next for lambda, not t!
            lambda_t_next = self._get_lambda_t(t_next)
            v2_guided = v2 + lambda_t_next * delta_t_2 + delta_0_2

            # Heun with guided velocities (dt implicit through ODE)
            return z + dt * 0.5 * (v1_guided + v2_guided)

        # Standard path: unguided Euler step for corrector
        z_euler = z + dt * v1
        z_cfg_next = torch.cat([z_euler, z_euler], dim=0)
        t_tensor_next = torch.full((bsz * 2,), T_next / 1000.0, device=device)
        v_pred2 = self.denoiser.net.forward_multires(z_cfg_next, t_tensor_next, labels_cfg, h)
        v_uncond2, v_cond2 = v_pred2[:bsz], v_pred2[bsz:]
        v2 = v_uncond2 + stage_cfg * (v_cond2 - v_uncond2)

        z_next = z + dt * 0.5 * (v1 + v2)

        # TFG guidance (x-space and v-space)
        if guidance is not None and self.tfg_config is not None:
            if self.guidance_space == "x":
                # x-space: TFG/DDPM heritage - Position correction (NO dt)
                delta_t, delta_0 = self._pixelflow_compute_guidance_deltas(
                    z=z,
                    v_guided=v1,
                    t=t,
                    T=T,
                    h=h,
                    labels=labels,
                    labels_cfg=labels_cfg,
                    stage_cfg=stage_cfg,
                    guidance=guidance,
                    tfg_targets=tfg_targets,
                )
                t_clamped = max(t, self.t_eps)
                z_next = z_next + (t_next / t_clamped) * delta_t + t_next * delta_0
            elif self.guidance_space == "v":
                # v-space: Flow Guidance heritage - Velocity modification (dt implicit)
                delta_t, delta_0 = self._pixelflow_compute_guidance_deltas(
                    z=z,
                    v_guided=v1,
                    t=t,
                    T=T,
                    h=h,
                    labels=labels,
                    labels_cfg=labels_cfg,
                    stage_cfg=stage_cfg,
                    guidance=guidance,
                    tfg_targets=tfg_targets,
                )
                lambda_t = self._get_lambda_t(t)

                v_avg_guided = (v1 + v2) / 2 + lambda_t * delta_t + delta_0
                z_next = z + dt * v_avg_guided

        return z_next

    @torch.enable_grad()
    def _pixelflow_compute_guidance_deltas(
        self,
        z: torch.Tensor,
        v_guided: torch.Tensor,
        t: float,
        T: float,
        h: int,
        labels: torch.Tensor,
        labels_cfg: torch.Tensor,
        stage_cfg: float,
        guidance: BaseGuider,
        tfg_targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute TFG guidance deltas for PixelFlow."""
        config = self.tfg_config
        bsz = z.size(0)
        device = z.device
        full_size = self.img_size

        # Get time-dependent parameters
        rho = self._get_schedule_value(config.rho, config.rho_schedule, t, config.normalize_schedules)
        mu = self._get_schedule_value(config.mu, config.mu_schedule, t, config.normalize_schedules)
        # Sigma is NEVER normalized - it controls Monte Carlo smoothing kernel width,
        # not guidance strength. Original TFG: get_std() has no normalization.
        sigma = self._get_schedule_value(config.sigma, config.sigma_schedule, t, normalize=False)

        t_4d = torch.tensor(t, device=device, dtype=z.dtype).view(1, 1, 1, 1)

        # Fallback to CFG labels if tfg_targets not provided.
        # This is PixelFlow-specific: since PixelFlow operates in pixel space at
        # lower resolutions during early stages, using class labels provides a
        # reasonable default guidance signal even without explicit TFG targets.
        # For other models (JiT, SiT, DiT), None is passed to guidance, which
        # means the guider must handle target-less guidance (e.g., unconditional).
        effective_targets = tfg_targets if tfg_targets is not None else labels

        # Initialize deltas
        delta_t = torch.zeros_like(z)
        delta_0 = torch.zeros_like(z)

        z_current = z  # Local copy for recurrence (matches JiT/SiT pattern)

        for recur_idx in range(config.recur_steps):
            mc_eps = self._get_mc_noise((bsz, 3, full_size, full_size), sigma)

            # Variance guidance
            if rho != 0.0:
                z_grad = z_current.clone().detach().requires_grad_(True)

                # Forward with gradients
                z_cfg_grad = torch.cat([z_grad, z_grad], dim=0)
                t_tensor_grad = torch.full((bsz * 2,), T / 1000.0, device=device)
                v_pred_grad = self.denoiser.net.forward_multires(z_cfg_grad, t_tensor_grad, labels_cfg, h)
                v_uncond_grad, v_cond_grad = v_pred_grad[:bsz], v_pred_grad[bsz:]
                v_guided_grad = v_uncond_grad + stage_cfg * (v_cond_grad - v_uncond_grad)

                # x0 estimate
                x0_est = z_grad + (1 - t_4d) * v_guided_grad
                if config.clip_x0:
                    x0_est = torch.clamp(x0_est, -config.clip_sample_range, config.clip_sample_range)

                # Upsample to full resolution for guidance
                x0_full = self._upsample_if_needed(x0_est, h, full_size)

                logprobs = self._compute_smoothed_logprob(x0_full, mc_eps, guidance, tfg_targets=effective_targets)
                delta_t = grad(logprobs.sum(), z_grad)[0]
                delta_t = torch.nan_to_num(delta_t, nan=0.0, posinf=0.0, neginf=0.0)
                delta_t = rescale_grad(delta_t, clip_scale=config.clip_scale, mode=config.rescale_mode)
                delta_t = delta_t * rho
            else:
                # rho=0: need x0 estimate for mean guidance only
                if recur_idx > 0:
                    # z_current was re-sampled by recurrence — must recompute velocity
                    with torch.no_grad():
                        z_cfg_rho0 = torch.cat([z_current, z_current], dim=0)
                        t_tensor_rho0 = torch.full((bsz * 2,), T / 1000.0, device=device)
                        v_pred_rho0 = self.denoiser.net.forward_multires(z_cfg_rho0, t_tensor_rho0, labels_cfg, h)
                        v_unc_rho0, v_cond_rho0 = v_pred_rho0[:bsz], v_pred_rho0[bsz:]
                        v_current = v_unc_rho0 + stage_cfg * (v_cond_rho0 - v_unc_rho0)
                    x0_est = z_current + (1 - t_4d) * v_current
                else:
                    # First iteration: passed-in v_guided is still valid
                    x0_est = z_current + (1 - t_4d) * v_guided
                if config.clip_x0:
                    x0_est = torch.clamp(x0_est, -config.clip_sample_range, config.clip_sample_range)

            # Mean guidance
            x0_full = self._upsample_if_needed(x0_est.detach(), h, full_size)
            if h >= full_size:
                x0_full = x0_full.clone()

            for _ in range(config.iter_steps):
                if mu != 0.0:
                    x0_full_grad = x0_full.detach().requires_grad_(True)
                    guidance_grad = self._compute_smoothed_grad(
                        x0_full_grad, mc_eps, guidance, tfg_targets=effective_targets
                    )
                    x0_full = x0_full + mu * guidance_grad

            # Compute delta_0 at full resolution then resize back
            x0_est_full = self._upsample_if_needed(x0_est.detach(), h, full_size)
            delta_0_full = x0_full - x0_est_full
            delta_0 = self._downsample_if_needed(delta_0_full, h, full_size)

            # Recurrence: complete re-sample (TFG paper original intent)
            # At the same timestep t, re-sample z with new noise realization
            if recur_idx < config.recur_steps - 1:
                with torch.no_grad():
                    # Recompute velocity from z_current (may have changed in prior
                    # recurrence iterations — cannot reuse the passed-in v_guided).
                    z_cfg_recur = torch.cat([z_current, z_current], dim=0)
                    t_tensor_recur = torch.full((bsz * 2,), T / 1000.0, device=device)
                    v_pred_recur = self.denoiser.net.forward_multires(z_cfg_recur, t_tensor_recur, labels_cfg, h)
                    v_unc_r, v_cond_r = v_pred_recur[:bsz], v_pred_recur[bsz:]
                    v_recur = v_unc_r + stage_cfg * (v_cond_r - v_unc_r)
                    x_est = z_current + (1 - t) * v_recur
                    if config.clip_x0:
                        x_est = torch.clamp(x_est, -config.clip_sample_range, config.clip_sample_range)

                # Complete re-sample with new noise
                new_noise = torch.randn_like(z_current)
                z_current = t * x_est + (1 - t) * new_noise

        return delta_t, delta_0

    def _sample_block_noise(
        self, bs: int, ch: int, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sample correlated 2x2 block noise for PixelFlow stage transitions."""
        eps = 1e-6
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(4),
            torch.eye(4) * (1 - self.gamma) + torch.ones(4, 4) * self.gamma + eps * torch.eye(4),
        )
        block_number = bs * ch * (h // 2) * (w // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)])
        noise = rearrange(
            noise,
            "(b c h w) (p q) -> b c (h p) (w q)",
            b=bs,
            c=ch,
            h=h // 2,
            w=w // 2,
            p=2,
            q=2,
        )
        return noise.to(device=device, dtype=dtype)

    def _upsample_if_needed(self, x: torch.Tensor, h: int, full_size: int) -> torch.Tensor:
        """Upsample tensor to full resolution if needed."""
        if h < full_size:
            return F.interpolate(x, size=full_size, mode="bilinear", align_corners=False)
        return x

    def _downsample_if_needed(self, x: torch.Tensor, h: int, full_size: int) -> torch.Tensor:
        """Downsample tensor back to current resolution if needed."""
        if h < full_size:
            return F.interpolate(x, size=h, mode="bilinear", align_corners=False)
        return x

    # =========================================================================
    # Common TFG utilities
    # =========================================================================

    def _get_schedule_value(self, base_value: float, schedule: str, t: float, normalize: bool = False) -> float:
        """Get time-dependent value based on schedule.

        Args:
            base_value: Base value for the parameter (e.g., rho, mu, sigma)
            schedule: Schedule type ("constant", "increase", "decrease")
            t: Current timestep in [0, 1] range
            normalize: If True, normalize so that the average value over t in [0,1]
                equals base_value. This matches the original TFG implementation.
                Default: False.

        Returns:
            Scheduled value at timestep t.

        Note:
            Original TFG (DDPM) uses: value = base * schedule[t] * len(s) / sum(s)
            This normalization ensures that hyperparameters have consistent meaning
            across different schedules. Without it, "increase" with rho=2.0 would
            average to 1.0 instead of 2.0.

            For continuous time t in [0,1]:
            - "increase": integral of t from 0 to 1 = 0.5, so multiply by 2
            - "decrease": integral of (1-t) from 0 to 1 = 0.5, so multiply by 2
            - "constant": integral of 1 from 0 to 1 = 1.0, no normalization needed
        """
        if schedule == "constant":
            return base_value
        elif schedule == "increase":
            # Normalization: integral of t from 0 to 1 is 0.5
            # To make average = base_value, multiply by 2
            normalizer = 2.0 if normalize else 1.0
            return base_value * t * normalizer
        elif schedule == "decrease":
            # Normalization: integral of (1-t) from 0 to 1 is 0.5
            # To make average = base_value, multiply by 2
            normalizer = 2.0 if normalize else 1.0
            return base_value * (1 - t) * normalizer
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def _get_mc_noise(self, shape: tuple[int, ...], sigma: float) -> torch.Tensor:
        """Get Monte Carlo noise samples for gradient smoothing."""
        if sigma == 0.0:
            return torch.zeros((1, *shape), device=self.device)

        eps_samples = []
        for _ in range(self.tfg_config.eps_bsz):
            eps = torch.randn(shape, device=self.device, generator=self.generator) * sigma
            eps_samples.append(eps)

        return torch.stack(eps_samples)

    def _decode_latent_for_guidance(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent tensors to pixel space for guidance classifiers.

        For latent diffusion models (DiT, SiT), the x0 prediction lives in
        VAE latent space (B, 4, 32, 32). Pixel-space classifiers expect
        (B, 3, H, W) images. This method bridges the gap using differentiable
        VAE decoding so that gradients flow back to latent space.

        For pixel-space models (JiT, PixelFlow), this is a no-op.

        Args:
            x: Tensor of shape (B, C, H, W). Either latent (4-ch) or pixel (3-ch).

        Returns:
            Pixel-space tensor of shape (B, 3, H, W) in [-1, 1] range.
        """
        if not self.is_latent:
            return x
        return self.denoiser.vae.decode_with_grad(x)

    @torch.enable_grad()
    def _compute_smoothed_logprob(
        self,
        x0: torch.Tensor,
        mc_eps: torch.Tensor,
        guidance: BaseGuider,
        tfg_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute smoothed log probability using Monte Carlo estimation.

        For latent diffusion models (DiT, SiT), x0 is in VAE latent space.
        MC noise is added in latent space, then each perturbed sample is
        decoded to pixel space before being passed to the guidance classifier.

        Args:
            x0: Predicted clean sample. Latent (B, 4, 32, 32) or pixel (B, 3, H, W).
            mc_eps: Monte Carlo noise of shape (mc_bsz, B, C, H, W).
            guidance: Guidance function (classifier).
            tfg_targets: Per-sample target classes of shape (B,).

        Returns:
            Averaged log probability of shape (B,).
        """
        flat_x0 = (x0.unsqueeze(0) + mc_eps).reshape(-1, *x0.shape[1:])

        flat_x0 = self._decode_latent_for_guidance(flat_x0)

        expanded_targets = None
        if tfg_targets is not None:
            expanded_targets = tfg_targets.repeat(mc_eps.shape[0])

        logprobs = guidance.get_guidance(
            flat_x0,
            targets=expanded_targets,
            return_logp=True,
            check_grad=False,
        )

        logprobs = logprobs.reshape(mc_eps.shape[0], x0.shape[0])
        avg_logprobs = torch.logsumexp(logprobs, dim=0) - math.log(mc_eps.shape[0])

        return avg_logprobs

    @torch.enable_grad()
    def _compute_smoothed_grad(
        self,
        x0: torch.Tensor,
        mc_eps: torch.Tensor,
        guidance: BaseGuider,
        tfg_targets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute smoothed gradient using Monte Carlo estimation.

        Gradients flow through VAE decode for latent models:
        classifier → pixel image → VAE decode → latent x0 (via chain rule).

        Args:
            x0: Predicted clean sample requiring gradients.
            mc_eps: Monte Carlo noise.
            guidance: Guidance function (classifier).
            tfg_targets: Per-sample target classes.

        Returns:
            Gradient tensor of same shape as x0.
        """
        avg_logprobs = self._compute_smoothed_logprob(x0, mc_eps, guidance, tfg_targets)
        gradient = grad(avg_logprobs.sum(), x0)[0]
        return rescale_grad(gradient, clip_scale=self.tfg_config.clip_scale, mode=self.tfg_config.rescale_mode)
