"""PixelFlow Denoiser for TFG integration.

This module provides the PixelFlowDenoiser class that wraps PixelFlow with
multi-stage pyramid sampling and provides an interface compatible with TFG.

The denoiser handles:
1. Multi-stage pyramid sampling (32→64→128→256)
2. Within-stage time normalization (each stage: t=0→1)
3. Stage-dependent CFG scaling
4. Block noise injection between stages
5. Prediction target conversion (v -> x/e)

Key difference from SiTDenoiser:
- Operating space: PIXEL (3, 256, 256) not latent (4, 32, 32)
- NO VAE encode/decode needed!
- Uses pyramid sampling instead of single-stage ODE

This makes PixelFlow ideal for TFG experiments in pixel space.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from jit_tfg.models.pixelflow.wrapper import PixelFlowWrapper
from jit_tfg.models.sit.transport.path import LinearPath


class PixelFlowDenoiser(nn.Module):
    """Multi-stage Pyramid Denoiser for PixelFlow, compatible with TFG system.

    This class provides the same interface as JiT's Denoiser but uses
    PixelFlow's v-prediction framework with multi-stage pyramid sampling.
    It bridges PixelFlow with TFG guidance.

    Key characteristics:
    - Prediction target: v (velocity = x - epsilon)
    - Sampling: Multi-stage pyramid (32→64→128→256)
    - Time convention: Within-stage normalization (each stage t=0→1)
    - Operating space: PIXEL (3, 256, 256) - NO VAE!
    - Stage-dependent CFG: Increases from 1.0 to cfg_scale across stages

    Attributes:
        net: PixelFlowWrapper instance.
        path: LinearPath for flow matching interpolation.
        pred_target: Always "v" (velocity) for PixelFlow.
        cfg_scale: Final classifier-free guidance scale.
        num_sampling_steps: Steps per stage (default: 30, paper default, yields 120 NFE with Euler).
        num_stages: Number of pyramid stages (default: 4).
        img_size: Output image size (256).
        num_classes: Number of classes for conditioning.
        is_latent_diffusion: False (pixel space).
        gamma: Block noise correlation (-1/3).

    Example:
        >>> denoiser = load_pixelflow_denoiser(checkpoint_path="path/to/checkpoint")
        >>> labels = torch.tensor([207, 360], device="cuda")
        >>> images = denoiser.generate(labels, num_steps=10)
    """

    def __init__(
        self,
        net: PixelFlowWrapper,
        num_sampling_steps: int = 30,
        cfg_scale: float = 2.4,
        t_eps: float = 5e-2,
        img_size: int = 256,
        num_stages: int = 4,
        gamma: float = -1 / 3,
    ) -> None:
        """Initialize PixelFlow Denoiser.

        Args:
            net: PixelFlowWrapper instance wrapping the PixelFlow model.
            num_sampling_steps: Steps per stage (default: 30, paper default, 120 NFE with Euler).
            cfg_scale: Final classifier-free guidance scale (default: 2.4, paper optimal).
            t_eps: Small epsilon for numerical stability.
            img_size: Output image size.
            num_stages: Number of pyramid stages (default: 4).
            gamma: Block noise correlation coefficient (default: -1/3).
        """
        super().__init__()

        self.net = net
        self.path = LinearPath()
        self.pred_target: Literal["v"] = "v"  # PixelFlow always predicts velocity
        self.t_eps = t_eps
        self.cfg_scale = cfg_scale
        self.num_sampling_steps = num_sampling_steps
        self.img_size = img_size
        self.num_classes = net.num_classes
        self.num_stages = num_stages
        self.gamma = gamma

        # Precompute stage time ranges
        self._setup_stage_schedule()

        # TFG compatibility attributes (CRITICAL)
        self.noise_scale = 1.0  # Unit variance noise in pixel space
        self.steps = num_sampling_steps * num_stages  # Total steps
        self.is_latent_diffusion = False  # KEY: pixel space, no VAE!

    def _setup_stage_schedule(self) -> None:
        """Precompute stage-specific time schedules."""
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
        """Calculate rectification ratio for stage transition."""
        return 1 / (math.sqrt(1 - (1 / self.gamma)) * (1 - start_t) + start_t)

    def _get_stage_cfg_scale(self, stage_idx: int) -> float:
        """Get stage-dependent CFG scale."""
        scale_dict = {0: 0, 1: 1 / 6, 2: 2 / 3, 3: 1}
        return (self.cfg_scale - 1) * scale_dict.get(stage_idx, 1) + 1

    def _sample_block_noise(
        self, bs: int, ch: int, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sample correlated 2x2 block noise for stage transitions."""
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

    def _convert_prediction(
        self,
        v_pred: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert velocity prediction to x, v, e predictions.

        For flow matching velocity prediction:
        - v_pred = network output (given)
        - x_pred = z + (1-t) * v_pred
        - e_pred = z - t * v_pred

        Args:
            v_pred: Velocity prediction of shape (B, C, H, W).
            z: Noisy image of shape (B, C, H, W).
            t: Continuous timestep of shape (B,) or (B, 1, 1, 1) in [0, 1].

        Returns:
            Tuple of (x_pred, v_pred, e_pred), each of shape (B, C, H, W).
        """
        # Handle t shape
        if t.ndim == 1:
            t = t.view(-1, 1, 1, 1)

        # v -> x_0: x = z + (1-t) * v
        x_pred = self.path.get_x0_from_velocity(v_pred, z, t)

        # v -> epsilon: epsilon = z - t * v
        e_pred = self.path.get_noise_from_velocity(v_pred, z, t)

        return x_pred, v_pred, e_pred

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning velocity prediction.

        Args:
            z: Noisy pixel images of shape (B, 3, H, W).
            t: Continuous timestep of shape (B,) in [0, 1].
            labels: Class labels of shape (B,).

        Returns:
            Velocity prediction of shape (B, 3, H, W).
        """
        return self.net(z, t, labels)

    @torch.no_grad()
    def _forward_sample(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity prediction with classifier-free guidance.

        Used by UnifiedSampler for per-step velocity computation.
        Returns velocity for ODE stepping.

        Args:
            z: Current noisy image of shape (B, C, H, W).
            t: Current timestep of shape (B,) or (B, 1, 1, 1) in [0, 1].
            labels: Class labels of shape (B,).

        Returns:
            Velocity prediction of shape (B, C, H, W).
        """
        # Handle t shape
        t_flat = t.flatten() if t.ndim > 1 else t

        # Conditional prediction
        v_cond = self.net(z, t_flat, labels)

        # Unconditional prediction
        y_uncond = torch.full_like(labels, self.num_classes)
        v_uncond = self.net(z, t_flat, y_uncond)

        # CFG combination
        v_guided = v_uncond + self.cfg_scale * (v_cond - v_uncond)

        return v_guided

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        num_steps: int | None = None,
        method: Literal["euler", "dopri5"] = "euler",
        show_progress: bool = True,
        dopri5_atol: float = 1e-6,
        dopri5_rtol: float = 1e-3,
    ) -> torch.Tensor:
        """Generate images from class labels using multi-stage pyramid sampling.

        Returns pixel images directly - no VAE decode needed!

        Uses PixelFlow's cascade sampling:
        1. Start at 32x32 with random noise
        2. For each stage: run ODE steps, then upsample and inject block noise
        3. Final stage produces 256x256 images

        Args:
            labels: Class labels of shape (B,).
            num_steps: Steps per stage (default: self.num_sampling_steps).
            method: ODE solver - "euler" (default, TFG compatible) or "dopri5" (adaptive, paper optimal).
            show_progress: Whether to show progress bar.
            dopri5_atol: Absolute tolerance for Dopri5 solver (default: 1e-6).
            dopri5_rtol: Relative tolerance for Dopri5 solver (default: 1e-3).

        Returns:
            Generated images of shape (B, 3, 256, 256) in [-1, 1] range.

        Note:
            - "euler": Fixed-step Euler solver, compatible with TFG guidance.
            - "dopri5": Adaptive Runge-Kutta solver (paper optimal for baseline FID 1.98).
              NOT compatible with TFG due to adaptive stepping.
        """
        if method not in ("euler", "dopri5"):
            raise ValueError(
                f"PixelFlow only supports 'euler' and 'dopri5' solvers, got '{method}'. "
                "Heun sampling is available via UnifiedSampler, not through generate() directly."
            )

        device = labels.device
        batch_size = labels.size(0)
        num_steps = num_steps or self.num_sampling_steps

        # Initialize at lowest resolution (32x32 for 4 stages)
        init_factor = 2 ** (self.num_stages - 1)
        h = w = self.img_size // init_factor
        latents = torch.randn(batch_size, 3, h, w, device=device)

        # Prepare CFG labels
        prompt_embeds = labels.int()
        prompt_embeds_cfg = torch.cat([torch.full_like(prompt_embeds, self.num_classes), prompt_embeds], dim=0)

        total_steps = num_steps * self.num_stages
        pbar = tqdm(total=total_steps, desc="PixelFlow", disable=not show_progress)

        for stage_idx in range(self.num_stages):
            if stage_idx > 0:
                # Upsample
                h, w = h * 2, w * 2
                latents = F.interpolate(latents, size=(h, w), mode="nearest")

                # Inject block noise for stage transition
                orig_st = self.original_start_t[stage_idx]
                alpha = self._cal_rectify_ratio(orig_st)
                beta = alpha * (1 - orig_st) / math.sqrt(-self.gamma)
                noise = self._sample_block_noise(batch_size, 3, h, w, device, latents.dtype)
                latents = alpha * latents + beta * noise

            # Get stage-specific settings
            stage_cfg = self._get_stage_cfg_scale(stage_idx)

            # Timesteps to pass to model (in [0, 1000) range)
            # These are the global timesteps for model conditioning
            # Match original PixelFlowScheduler exactly:
            #   T_start_idx, T_end_idx: integer indices into Timesteps[0..999]
            #   For non-last stages: stage_T endpoints from linspace(T_start, T_end, 1001)[:-1]
            #   For last stage: stage_T endpoints from linspace(T_start, T_end, 1000)
            #   Then t ∈ [0, 0.999] is linearly mapped to T ∈ [stage_T_start, stage_T_end]
            _NUM_TRAIN_TIMESTEPS = 1000
            start_ratio = 0.0 if stage_idx == 0 else sum(self.stage_distance[:stage_idx]) / self.total_stage_distance
            end_ratio = (
                1.0
                if stage_idx == self.num_stages - 1
                else sum(self.stage_distance[: stage_idx + 1]) / self.total_stage_distance
            )
            T_start_idx = int(_NUM_TRAIN_TIMESTEPS * start_ratio)
            T_end_idx = min(int(_NUM_TRAIN_TIMESTEPS * end_ratio), _NUM_TRAIN_TIMESTEPS - 1)

            # Compute stage_T_start/end matching original's Timesteps_per_stage
            if stage_idx == self.num_stages - 1:
                # Last stage: linspace(T_start, T_end, 1000) → endpoints are exact
                stage_T_start = float(T_start_idx)
                stage_T_end = float(T_end_idx)
            else:
                # Non-last stages: linspace(T_start, T_end, 1001)[:-1]
                # First element = T_start, last element = T_end - (T_end - T_start)/1000
                stage_T_start = float(T_start_idx)
                stage_T_end = T_end_idx - (T_end_idx - T_start_idx) / _NUM_TRAIN_TIMESTEPS

            if method == "dopri5":
                # Dopri5 adaptive ODE solver (paper optimal for baseline)
                latents = self._dopri5_sample_stage(
                    latents=latents,
                    batch_size=batch_size,
                    h=h,
                    T_start=stage_T_start,
                    T_end=stage_T_end,
                    stage_cfg=stage_cfg,
                    prompt_embeds_cfg=prompt_embeds_cfg,
                    device=device,
                    atol=dopri5_atol,
                    rtol=dopri5_rtol,
                )
                pbar.update(num_steps)  # Approximate progress
            else:
                # Euler fixed-step solver (TFG compatible)
                # Match original PixelFlowScheduler: t ∈ linspace(0, 999/1000, num_steps)
                # with appended 1.0, then linearly mapped to Timesteps
                t_end = (_NUM_TRAIN_TIMESTEPS - 1) / _NUM_TRAIN_TIMESTEPS  # 0.999
                t_schedule = np.linspace(0, t_end, num_steps, dtype=np.float64)
                t_within_stage = np.append(t_schedule, 1.0)
                # Linear map: T = stage_T_start + (t / 0.999) * (stage_T_end - stage_T_start)
                Timesteps = stage_T_start + (t_schedule / t_end) * (stage_T_end - stage_T_start)

                for i in range(num_steps):
                    t = t_within_stage[i]
                    t_next = t_within_stage[i + 1]
                    T = Timesteps[i]

                    # Forward with CFG
                    latent_input = torch.cat([latents] * 2)
                    n_cfg = latent_input.shape[0]

                    # Call wrapper directly for multi-resolution support
                    # T is in [0, 1000), forward_multires expects t in [0, 1]
                    t_input = torch.full((n_cfg,), T / 1000.0, device=device)
                    v_pred = self.net.forward_multires(latent_input, t_input, prompt_embeds_cfg, h)

                    # CFG combination
                    v_uncond, v_cond = v_pred[:batch_size], v_pred[batch_size:]
                    v_guided = v_uncond + stage_cfg * (v_cond - v_uncond)

                    # Euler step with within-stage time
                    dt = t_next - t
                    latents = latents + dt * v_guided

                    pbar.update(1)

        pbar.close()
        return latents

    def _dopri5_sample_stage(
        self,
        latents: torch.Tensor,
        batch_size: int,
        h: int,
        T_start: float,
        T_end: float,
        stage_cfg: float,
        prompt_embeds_cfg: torch.Tensor,
        device: torch.device,
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ) -> torch.Tensor:
        """Sample a single stage using Dopri5 adaptive ODE solver.

        This implements the paper's optimal sampling strategy for baseline generation.
        Uses torchdiffeq's odeint with the Dopri5 (4th/5th order Runge-Kutta) method.

        Args:
            latents: Current latent state of shape (B, 3, H, W).
            batch_size: Batch size.
            h: Current resolution height.
            T_start: Start time in [0, 1000) range.
            T_end: End time in [0, 1000) range.
            stage_cfg: Stage-specific CFG scale.
            prompt_embeds_cfg: CFG-concatenated labels of shape (2*B,).
            device: Torch device.
            atol: Absolute tolerance for adaptive stepping.
            rtol: Relative tolerance for adaptive stepping.

        Returns:
            Final latent state after ODE integration.
        """
        from torchdiffeq import odeint

        # Define velocity function for ODE solver
        # t is within-stage time in [0, 1], but we need to map to global T
        def velocity_fn(t_within: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            # Map within-stage t [0, 1] to global T [T_start, T_end]
            t_scalar = t_within.item()
            T = T_start + t_scalar * (T_end - T_start)

            # Forward with CFG
            latent_input = torch.cat([z] * 2)
            n_cfg = latent_input.shape[0]

            # T is in [0, 1000), forward_multires expects t in [0, 1]
            t_input = torch.full((n_cfg,), T / 1000.0, device=device)
            v_pred = self.net.forward_multires(latent_input, t_input, prompt_embeds_cfg, h)

            # CFG combination
            v_uncond, v_cond = v_pred[:batch_size], v_pred[batch_size:]
            v_guided = v_uncond + stage_cfg * (v_cond - v_uncond)

            return v_guided

        # Integration times: within-stage [0, 1]
        t_span = torch.tensor([0.0, 1.0], device=device)

        # Solve ODE using Dopri5
        solution = odeint(
            velocity_fn,
            latents,
            t_span,
            method="dopri5",
            atol=atol,
            rtol=rtol,
        )

        # Return final state (solution[-1] is at t=1)
        return solution[-1]

    def _euler_sample_single_stage(
        self,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        iterator,
    ) -> torch.Tensor:
        """Single-stage Euler method (for compatibility/testing).

        Note: This does NOT produce good results for pretrained PixelFlow models.
        Use generate() with multi-stage sampling instead.

        Args:
            z: Initial noise of shape (B, C, H, W).
            timesteps: Timestep schedule of shape (num_steps + 1,).
            labels: Class labels of shape (B,).
            iterator: Progress iterator.

        Returns:
            Final sample at t=1.
        """
        batch_size = z.size(0)

        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_tensor = torch.full((batch_size,), t, device=z.device)

            # Get velocity prediction with CFG
            v_pred = self._forward_sample(z, t_tensor, labels)

            # Euler step: z_{t+dt} = z_t + dt * v(z_t, t)
            dt = t_next - t
            z = z + dt * v_pred

        return z

    @torch.no_grad()
    def generate_flow_matching_style(
        self,
        labels: torch.Tensor,
        num_steps: int | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate using standard flow matching (alias for generate).

        This method exists for interface compatibility with other denoisers.
        For PixelFlow, it's identical to generate() with Euler method.

        Args:
            labels: Class labels of shape (B,).
            num_steps: Number of sampling steps.
            show_progress: Whether to show progress bar.

        Returns:
            Generated images of shape (B, 3, 256, 256) in [-1, 1] range.
        """
        return self.generate(
            labels=labels,
            num_steps=num_steps,
            method="euler",
            show_progress=show_progress,
        )

    def to(self, device: torch.device | str) -> PixelFlowDenoiser:
        """Move denoiser to device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        super().to(device)
        return self


def load_pixelflow_denoiser(
    checkpoint_path: str | None = None,
    from_pretrained: str | None = None,
    device: str | torch.device = "cuda",
    cfg_scale: float = 2.4,
    num_sampling_steps: int = 30,
) -> PixelFlowDenoiser:
    """Load PixelFlow denoiser from checkpoint.

    Args:
        checkpoint_path: Path to local checkpoint folder containing:
            - config.yaml: Model configuration
            - model.pt: Model weights (may contain 'ema' or 'model' keys)
        from_pretrained: HuggingFace Hub model ID (e.g., "ShoufaChen/PixelFlow-Class2Image").
        device: Target device.
        cfg_scale: Classifier-free guidance scale (default: 2.4, paper optimal).
        num_sampling_steps: Steps per stage (default: 30, paper default, 120 NFE with Euler).

    Returns:
        Initialized PixelFlowDenoiser ready for inference.

    Example:
        >>> # From local checkpoint
        >>> denoiser = load_pixelflow_denoiser(checkpoint_path="checkpoints/pixelflow/")

        >>> # From HuggingFace Hub (if available)
        >>> denoiser = load_pixelflow_denoiser(from_pretrained="PixelFlow/PixelFlow-XL-256")

    Note:
        The checkpoint folder should contain:
        - config.yaml with model architecture parameters
        - model.pt with state_dict (optionally under 'ema' or 'model' keys)
    """
    from omegaconf import OmegaConf

    from jit_tfg.models.pixelflow.model import PixelFlowModel
    from jit_tfg.models.pixelflow.wrapper import PixelFlowWrapper

    device = torch.device(device)

    if checkpoint_path:
        # Load from local checkpoint folder
        import os

        config_path = os.path.join(checkpoint_path, "config.yaml")
        model_path = os.path.join(checkpoint_path, "model.pt")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        config = OmegaConf.load(config_path)

        # Create model from config
        model = PixelFlowModel(
            num_attention_heads=config.model.params.num_attention_heads,
            attention_head_dim=config.model.params.attention_head_dim,
            in_channels=3,
            out_channels=3,
            depth=config.model.params.depth,
            num_classes=config.model.params.get("num_classes", 1000),
            patch_size=config.model.params.get("patch_size", 4),
            attention_bias=config.model.params.get("attention_bias", True),
        ).to(device)

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=device, weights_only=True)

        # Handle different checkpoint formats (ema, model, or raw state_dict)
        if isinstance(ckpt, dict):
            if "ema" in ckpt:
                state_dict = ckpt["ema"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict)

    elif from_pretrained:
        # Load from HuggingFace Hub
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_id=from_pretrained)
        return load_pixelflow_denoiser(
            checkpoint_path=local_dir,
            device=device,
            cfg_scale=cfg_scale,
            num_sampling_steps=num_sampling_steps,
        )

    else:
        raise ValueError("Either checkpoint_path or from_pretrained must be provided")

    model.eval()

    # Create wrapper
    wrapper = PixelFlowWrapper(model, img_size=256)

    # Create denoiser
    denoiser = PixelFlowDenoiser(
        net=wrapper,
        cfg_scale=cfg_scale,
        num_sampling_steps=num_sampling_steps,
    ).to(device)

    return denoiser
