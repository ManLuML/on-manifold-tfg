"""DiT Denoiser for TFG integration.

This module provides the DiTDenoiser class that wraps DiT with DDPM
diffusion logic and provides an interface compatible with the TFG system.

The denoiser handles:
1. DDPM forward/reverse diffusion
2. Prediction target conversion (epsilon -> x/v/e)
3. Classifier-free guidance
4. DDPM/DDIM sampling
5. VAE encode/decode for latent space operations

Note:
    For TFG experiments with DiT, use UnifiedSampler with model_type="DiT"
    which strictly follows the original TFG paper's DDPM conventions.
    The flow-matching style methods in this class are deprecated.
"""

from __future__ import annotations

import warnings
from typing import Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from jit_tfg.models.dit.diffusion.sampling import ddim_sample, ddpm_sample
from jit_tfg.models.dit.diffusion.schedules import DDPMSchedule
from jit_tfg.models.dit.vae import VAEHandler
from jit_tfg.models.dit.wrapper import CFGChannelMode, DiTWrapper


class DiTDenoiser(nn.Module):
    """DDPM Denoiser for DiT, compatible with TFG system.

    This class provides the same interface as JiT's Denoiser but implements
    DDPM diffusion instead of flow matching. It bridges DiT with the TFG
    guidance system.

    Key differences from JiT Denoiser:
    - Prediction target: Always epsilon (noise)
    - Sampling: DDPM or DDIM (not ODE)
    - Operating space: VAE latent (4, 32, 32) not pixel (3, 256, 256)
    - Timestep: Discrete [0, T-1] internally, continuous [0, 1] externally

    Attributes:
        net: DiTWrapper instance.
        vae: VAEHandler for latent <-> pixel conversion.
        schedule: DDPM diffusion schedule.
        pred_target: Always "e" (epsilon) for DiT.
        t_eps: Small epsilon for numerical stability.
        cfg_scale: Classifier-free guidance scale.
        num_sampling_steps: Default number of sampling steps.
        img_size: Output image size (256 for DiT-XL/2).
        latent_size: Latent spatial size (32 for 256x256 images).
        num_classes: Number of classes for conditioning.

    Example:
        >>> denoiser = DiTDenoiser.from_pretrained("facebook/DiT-XL-2-256")
        >>> labels = torch.tensor([207, 360], device="cuda")
        >>> images = denoiser.generate(labels, num_steps=50)
    """

    def __init__(
        self,
        net: DiTWrapper,
        vae: VAEHandler,
        schedule: DDPMSchedule | None = None,
        num_timesteps: int = 1000,
        beta_schedule: Literal["linear", "cosine"] = "linear",
        num_sampling_steps: int = 250,
        cfg_scale: float = 1.5,
        t_eps: float = 5e-2,
        img_size: int = 256,
        cfg_channel_mode: CFGChannelMode = "first3",
    ) -> None:
        """Initialize DiT Denoiser.

        Args:
            net: DiTWrapper instance wrapping the DiT model.
            vae: VAEHandler for latent space operations.
            schedule: Optional precomputed DDPM schedule.
            num_timesteps: Total DDPM timesteps (default: 1000).
            beta_schedule: Type of beta schedule ("linear" or "cosine").
            num_sampling_steps: Default number of sampling steps.
            cfg_scale: Classifier-free guidance scale.
            t_eps: Small epsilon for numerical stability.
            img_size: Output image size.
            cfg_channel_mode: How to apply CFG across output channels.
                - 'first3': Apply CFG only to first 3 channels (Meta original, default)
                - 'all': Apply CFG to all 4 channels (alternative)
        """
        super().__init__()

        self.net = net
        self.vae = vae
        self.pred_target: Literal["e"] = "e"  # DiT always predicts epsilon
        self.t_eps = t_eps
        self.cfg_scale = cfg_scale
        self.cfg_channel_mode: Literal["all", "first3"] = cfg_channel_mode
        self.num_sampling_steps = num_sampling_steps
        self.img_size = img_size
        self.latent_size = img_size // 8
        self.num_classes = net.num_classes
        self.num_timesteps = num_timesteps

        # TFG compatibility attributes
        self.noise_scale = 1.0  # DiT uses unit variance noise in latent space
        self.steps = num_sampling_steps
        self.is_latent_diffusion = True  # Signals that we work in latent space

        # Initialize DDPM schedule
        if schedule is not None:
            self.schedule = schedule
        else:
            self.schedule = DDPMSchedule.from_beta_schedule(
                schedule_type=beta_schedule,
                num_timesteps=num_timesteps,
            )

    def _convert_prediction(
        self,
        eps_pred: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert epsilon prediction to x, v, e predictions.

        For DDPM epsilon prediction:
        - e_pred = network output (given)
        - x_pred = (z - sqrt(1-α̅) * e_pred) / sqrt(α̅)
        - v_pred = sqrt(α̅) * e_pred - sqrt(1-α̅) * x_pred / sqrt(α̅)

        Args:
            eps_pred: Epsilon prediction of shape (B, C, H, W).
            z: Noisy latent of shape (B, C, H, W).
            t: Continuous timestep of shape (B,) or (B, 1, 1, 1) in [0, 1].

        Returns:
            Tuple of (x_pred, v_pred, e_pred), each of shape (B, C, H, W).
        """
        # Handle t shape
        if t.ndim == 1:
            t = t.view(-1, 1, 1, 1)

        # Convert continuous t to discrete DDPM index
        t_discrete = self.net.t_continuous_to_discrete(t.flatten())

        # Get DDPM coefficients
        sqrt_alpha = self.schedule.extract(self.schedule.sqrt_alphas_cumprod.to(z.device), t_discrete, z.shape)
        sqrt_one_minus_alpha = self.schedule.extract(
            self.schedule.sqrt_one_minus_alphas_cumprod.to(z.device), t_discrete, z.shape
        )

        # Convert epsilon to x_0
        x_pred = (z - sqrt_one_minus_alpha * eps_pred) / sqrt_alpha.clamp_min(self.t_eps)

        # Convert to velocity (for compatibility)
        # In flow matching: v = x - e, but in DDPM context we approximate
        # v ≈ (x_0 - z) / (1 - t) for consistency with TFG
        one_minus_t = (1 - t).clamp_min(self.t_eps)
        v_pred = (x_pred - z) / one_minus_t

        return x_pred, v_pred, eps_pred

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning epsilon prediction.

        Args:
            z: Noisy latents of shape (B, 4, H, W).
            t: Continuous timestep of shape (B,) in [0, 1].
            labels: Class labels of shape (B,).

        Returns:
            Epsilon prediction of shape (B, 4, H, W).
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

        For compatibility with TFG sampler, this returns velocity-like
        predictions that can be used for ODE-style stepping.

        Args:
            z: Current noisy latent of shape (B, C, H, W).
            t: Current timestep of shape (B,) or (B, 1, 1, 1) in [0, 1].
            labels: Class labels of shape (B,).

        Returns:
            Velocity-like prediction of shape (B, C, H, W).
        """
        # Handle t shape
        t_flat = t.flatten() if t.ndim > 1 else t

        # Conditional prediction
        eps_cond = self.net(z, t_flat, labels)

        # Unconditional prediction
        y_uncond = torch.full_like(labels, self.num_classes)
        eps_uncond = self.net(z, t_flat, y_uncond)

        # CFG combination
        eps_guided = eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)

        # Convert to velocity-like for TFG compatibility
        _, v_pred, _ = self._convert_prediction(eps_guided, z, t_flat)

        return v_pred

    @torch.no_grad()
    def forward_epsilon_with_cfg(
        self,
        z: torch.Tensor,
        t_discrete: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float | None = None,
        cfg_channel_mode: CFGChannelMode | None = None,
    ) -> torch.Tensor:
        """Get epsilon prediction with CFG using discrete timesteps.

        This method is the preferred way to get epsilon predictions for
        DDPM-based TFG sampling. It uses discrete timesteps directly
        without any conversion.

        The CFG channel mode is controlled by self.cfg_channel_mode:
        - "first3": Apply CFG only to first 3 channels (original DiT behavior).
        - "all": Apply CFG to all 4 channels.

        Args:
            z: Noisy latent of shape (B, C, H, W).
            t_discrete: Discrete timestep of shape (B,) in [0, T-1].
            labels: Class labels of shape (B,).
            cfg_scale: CFG scale. Uses self.cfg_scale if None.
            cfg_channel_mode: Override instance cfg_channel_mode if provided.
                - 'all': Apply CFG to all channels (standard)
                - 'first3': Apply CFG only to first 3 channels (Meta behavior)

        Returns:
            CFG-guided epsilon prediction of shape (B, C, H, W).
        """
        cfg_scale = cfg_scale or self.cfg_scale
        mode = cfg_channel_mode or self.cfg_channel_mode

        # Conditional prediction
        eps_cond = self.net.dit(z, t_discrete, labels)[:, : self.net.in_channels]

        # Unconditional prediction
        y_uncond = torch.full_like(labels, self.num_classes)
        eps_uncond = self.net.dit(z, t_discrete, y_uncond)[:, : self.net.in_channels]

        if mode == "first3":
            # Apply CFG only to first 3 channels (original Meta behavior)
            eps_cfg = eps_uncond[:, :3] + cfg_scale * (eps_cond[:, :3] - eps_uncond[:, :3])
            # Use conditional prediction for remaining channels
            return torch.cat([eps_cfg, eps_cond[:, 3:]], dim=1)
        else:  # mode == "all"
            # Standard CFG: apply to all channels
            return eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        num_steps: int | None = None,
        method: Literal["ddpm", "ddim"] = "ddim",
        eta: float = 0.0,
        return_latents: bool = False,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate images from class labels.

        Args:
            labels: Class labels of shape (B,).
            num_steps: Number of sampling steps.
            method: Sampling method ("ddpm" or "ddim").
            eta: DDIM stochasticity (0 = deterministic).
            return_latents: If True, return latents instead of decoded images.
            show_progress: Whether to show progress bar.

        Returns:
            Generated images of shape (B, 3, 256, 256) in [-1, 1] range.
            Or latents of shape (B, 4, 32, 32) if return_latents=True.
        """
        device = labels.device
        batch_size = labels.size(0)
        num_steps = num_steps or self.num_sampling_steps

        # Define model function with CFG
        cfg_channel_mode = self.cfg_channel_mode

        def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # Conditional
            eps_cond = self.net.dit(x, t, labels)[:, : self.net.in_channels]
            # Unconditional
            y_uncond = torch.full_like(labels, self.num_classes)
            eps_uncond = self.net.dit(x, t, y_uncond)[:, : self.net.in_channels]
            # CFG with channel mode
            if self.cfg_channel_mode == "first3":
                eps_cfg = eps_uncond[:, :3] + self.cfg_scale * (eps_cond[:, :3] - eps_uncond[:, :3])
                return torch.cat([eps_cfg, eps_cond[:, 3:]], dim=1)
            else:  # "all"
                return eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)

        shape = (batch_size, 4, self.latent_size, self.latent_size)

        if method == "ddpm":
            latents = ddpm_sample(
                model_fn=model_fn,
                schedule=self.schedule,
                shape=shape,
                device=device,
                num_steps=num_steps,
                show_progress=show_progress,
            )
        else:
            latents = ddim_sample(
                model_fn=model_fn,
                schedule=self.schedule,
                shape=shape,
                device=device,
                num_steps=num_steps,
                eta=eta,
                show_progress=show_progress,
            )

        if return_latents:
            return latents

        # Decode latents to images
        images = self.vae.decode(latents)
        return images

    @torch.no_grad()
    def generate_flow_matching_style(
        self,
        labels: torch.Tensor,
        num_steps: int | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate using flow-matching style ODE integration.

        .. deprecated::
            This method applies flow-matching ODE stepping to a DDPM model,
            which is conceptually incorrect. For TFG experiments with DiT,
            use UnifiedSampler with model_type="DiT" which follows the original
            TFG paper's DDPM conventions. For non-TFG generation, use generate(method='ddim').

        This method provides compatibility with JiT's generate() interface
        for fair comparison in TFG experiments.

        Args:
            labels: Class labels of shape (B,).
            num_steps: Number of sampling steps.
            show_progress: Whether to show progress bar.

        Returns:
            Generated images of shape (B, 3, 256, 256) in [-1, 1] range.
        """
        warnings.warn(
            "generate_flow_matching_style() is deprecated. "
            "For TFG experiments with DiT, use UnifiedSampler with model_type='DiT'. "
            "For non-TFG generation, use generate(method='ddim').",
            DeprecationWarning,
            stacklevel=2,
        )
        device = labels.device
        batch_size = labels.size(0)
        num_steps = num_steps or self.num_sampling_steps

        # Initialize from noise
        z = torch.randn(batch_size, 4, self.latent_size, self.latent_size, device=device)

        # Timesteps from 0 to 1 (JiT convention: 0=noise, 1=clean)
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Flow-style Sampling", total=num_steps)

        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_tensor = torch.full((batch_size,), t, device=device)

            # Get velocity-like prediction with CFG
            v_pred = self._forward_sample(z, t_tensor, labels)

            # Euler step
            z = z + (t_next - t) * v_pred

        # Decode to images
        images = self.vae.decode(z)
        return images

    def to(self, device: torch.device | str) -> DiTDenoiser:
        """Move denoiser to device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        super().to(device)
        self.vae = self.vae.to(device)
        self.schedule = self.schedule.to(torch.device(device))
        return self


def load_dit_denoiser(
    checkpoint_path: str | None = None,
    from_pretrained: str | None = None,
    device: str | torch.device = "cuda",
    vae_type: str = "mse",
    cfg_scale: float = 1.5,
    num_sampling_steps: int = 250,
    cfg_channel_mode: CFGChannelMode = "first3",
) -> DiTDenoiser:
    """Load DiT denoiser from checkpoint or HuggingFace Hub.

    Args:
        checkpoint_path: Path to local checkpoint (.pt file).
        from_pretrained: HuggingFace Hub model ID (e.g., "facebook/DiT-XL-2-256").
        device: Target device.
        vae_type: VAE variant ("mse" or "ema").
        cfg_scale: Classifier-free guidance scale.
        num_sampling_steps: Default number of sampling steps.
        cfg_channel_mode: How to apply CFG across output channels.
            - 'first3': Apply CFG only to first 3 channels (Meta original, default)
            - 'all': Apply CFG to all 4 channels (for TFG research)

    Returns:
        Initialized DiTDenoiser ready for inference.

    Example:
        >>> # From HuggingFace Hub
        >>> denoiser = load_dit_denoiser(from_pretrained="facebook/DiT-XL-2-256")

        >>> # From local checkpoint
        >>> denoiser = load_dit_denoiser(checkpoint_path="checkpoints/DiT-XL-2-256x256.pt")

        >>> # For exact reproducibility with Meta FID scores:
        >>> denoiser = load_dit_denoiser(
        ...     from_pretrained="facebook/DiT-XL-2-256",
        ...     cfg_channel_mode="first3",
        ... )
    """
    from jit_tfg.models.dit.model import DiT_XL_2

    device = torch.device(device)

    # Create DiT model
    dit = DiT_XL_2(
        input_size=32,
        num_classes=1000,
        learn_sigma=True,
    ).to(device)

    if checkpoint_path:
        # Load from local checkpoint (Meta official format)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Handle different checkpoint formats
        if "ema" in ckpt:
            state_dict = ckpt["ema"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        dit.load_state_dict(state_dict)

    elif from_pretrained:
        # Load from original Meta checkpoint (not HuggingFace diffusers format)
        # The diffusers format uses different layer names, so we use the original
        # checkpoints from dl.fbaipublicfiles.com
        import hashlib
        import os
        import urllib.request

        # Map HuggingFace model IDs to original checkpoint names and SHA256 hashes
        # Hashes verified from official Meta DiT release (dl.fbaipublicfiles.com)
        model_info = {
            "facebook/DiT-XL-2-256": {
                "filename": "DiT-XL-2-256x256.pt",
                "sha256": "9ec1876e4c03471bca126663a30e2d1b20610b6d2f87850a39a36f25cc685521",
            },
            # Note: DiT-XL-2-512 is not officially supported yet due to different
            # input size requirements. Use DiT-XL-2-256 for TFG experiments.
            # "facebook/DiT-XL-2-512": {
            #     "filename": "DiT-XL-2-512x512.pt",
            #     "sha256": "...",  # SHA256 hash not verified
            # },
        }

        if from_pretrained not in model_info:
            raise ValueError(f"Unknown model: {from_pretrained}. Supported models: {list(model_info.keys())}")

        info = model_info[from_pretrained]
        model_filename = info["filename"]
        expected_sha256 = info["sha256"]
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "jit-tfg", "dit")
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, model_filename)

        if not os.path.exists(ckpt_path):
            url = f"https://dl.fbaipublicfiles.com/DiT/models/{model_filename}"
            print(f"Downloading DiT checkpoint from {url}...")
            urllib.request.urlretrieve(url, ckpt_path)  # noqa: S310 - Known safe URL
            print(f"Saved to {ckpt_path}")

        # Verify checksum for security
        with open(ckpt_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != expected_sha256:
            raise ValueError(
                f"Checkpoint integrity check failed for {model_filename}. "
                f"Expected SHA256: {expected_sha256}, got: {file_hash}. "
                "The file may be corrupted or tampered with."
            )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        # Handle different checkpoint formats (EMA or model key)
        if "ema" in ckpt:
            state_dict = ckpt["ema"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        dit.load_state_dict(state_dict)
    else:
        raise ValueError("Either checkpoint_path or from_pretrained must be provided")

    dit.eval()

    # Create wrapper
    wrapper = DiTWrapper(dit, num_timesteps=1000, cfg_channel_mode=cfg_channel_mode)

    # Load VAE
    vae = VAEHandler(vae_type=vae_type, device=device)

    # Create denoiser
    denoiser = DiTDenoiser(
        net=wrapper,
        vae=vae,
        cfg_scale=cfg_scale,
        num_sampling_steps=num_sampling_steps,
        cfg_channel_mode=cfg_channel_mode,
    ).to(device)

    return denoiser
