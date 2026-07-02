"""SiT Denoiser for TFG integration.

This module provides the SiTDenoiser class that wraps SiT with flow matching
sampling logic and provides an interface compatible with the TFG system.

The denoiser handles:
1. Flow matching forward/reverse (ODE integration)
2. Prediction target conversion (v -> x/e)
3. Classifier-free guidance
4. Euler/Heun ODE sampling
5. VAE encode/decode for latent space operations

Key difference from DiTDenoiser:
- Prediction target: v (velocity) instead of ε (epsilon)
- Sampling: ODE integration instead of DDPM/DDIM
- Time convention: Same as JiT (t=0: noise, t=1: clean) - no inversion!

Note on ODE solver choice:
    The original SiT uses torchdiffeq.odeint with adaptive solvers (dopri5).
    We use manual Euler/Heun stepping for the following reasons:

    1. TFG algorithm structure: TFG operates on discrete timesteps with
       recurrence (N_recur) and iteration (N_iter) at each step. These
       per-step interventions are incompatible with adaptive solvers'
       variable internal steps.
    2. Consistency: JiT uses Euler, DiT uses DDIM. All models in this
       codebase use fixed-step solvers for uniform treatment.
    3. Fair comparison: Fixed-step solvers allow exact NFE (Number of
       Function Evaluations) control for fair comparisons across models.
    4. Reference: Matches SiT paper Algorithm 1 (Appendix D) which presents
       the Heun sampler with explicit discrete steps.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from jit_tfg.models.dit.vae import VAEHandler
from jit_tfg.models.sit.transport.path import LinearPath
from jit_tfg.models.sit.wrapper import CFGChannelMode, SiTWrapper


class SiTDenoiser(nn.Module):
    """Flow Matching Denoiser for SiT, compatible with TFG system.

    This class provides the same interface as JiT's Denoiser but uses
    SiT's v-prediction framework. It bridges SiT with the TFG guidance system.

    Key characteristics:
    - Prediction target: v (velocity = x - ε)
    - Sampling: ODE integration (Euler/Heun)
    - Time convention: Same as JiT (t=0: noise, t=1: clean)
    - Operating space: VAE latent (4, 32, 32) not pixel (3, 256, 256)

    Attributes:
        net: SiTWrapper instance.
        vae: VAEHandler for latent <-> pixel conversion.
        path: LinearPath for flow matching interpolation.
        pred_target: Always "v" (velocity) for SiT.
        t_eps: Small epsilon for numerical stability.
        cfg_scale: Classifier-free guidance scale.
        cfg_channel_mode: How to apply CFG across output channels.
        num_sampling_steps: Default number of sampling steps.
        img_size: Output image size (256 for SiT-XL/2).
        latent_size: Latent spatial size (32 for 256x256 images).
        num_classes: Number of classes for conditioning.

    Example:
        >>> denoiser = SiTDenoiser.from_pretrained("sit-xl-2")
        >>> labels = torch.tensor([207, 360], device="cuda")
        >>> images = denoiser.generate(labels, num_steps=125)
    """

    def __init__(
        self,
        net: SiTWrapper,
        vae: VAEHandler,
        num_sampling_steps: int = 125,
        cfg_scale: float = 1.5,
        cfg_channel_mode: CFGChannelMode = "first3",
        t_eps: float = 5e-2,
        img_size: int = 256,
    ) -> None:
        """Initialize SiT Denoiser.

        Args:
            net: SiTWrapper instance wrapping the SiT model.
            vae: VAEHandler for latent space operations.
            num_sampling_steps: Default number of sampling steps.
            cfg_scale: Classifier-free guidance scale.
            cfg_channel_mode: How to apply CFG across channels.
                - 'first3': Apply CFG only to first 3 channels (original SiT, default)
                - 'all': Apply CFG to all 4 channels (for TFG research)
            t_eps: Small epsilon for numerical stability.
            img_size: Output image size.
        """
        super().__init__()

        self.net = net
        self.vae = vae
        self.path = LinearPath()
        self.pred_target: Literal["v"] = "v"  # SiT always predicts velocity
        self.t_eps = t_eps
        self.cfg_scale = cfg_scale
        self.cfg_channel_mode = cfg_channel_mode
        self.num_sampling_steps = num_sampling_steps
        self.img_size = img_size
        self.latent_size = img_size // 8
        self.num_classes = net.num_classes

        # TFG compatibility attributes
        self.noise_scale = 1.0  # SiT uses unit variance noise in latent space
        self.steps = num_sampling_steps
        self.is_latent_diffusion = True  # Signals that we work in latent space

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
            z: Noisy latent of shape (B, C, H, W).
            t: Continuous timestep of shape (B,) or (B, 1, 1, 1) in [0, 1].

        Returns:
            Tuple of (x_pred, v_pred, e_pred), each of shape (B, C, H, W).
        """
        # Handle t shape
        if t.ndim == 1:
            t = t.view(-1, 1, 1, 1)

        # v -> x_0: x = z + (1-t) * v
        x_pred = self.path.get_x0_from_velocity(v_pred, z, t)

        # v -> ε: ε = z - t * v
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
            z: Noisy latents of shape (B, 4, H, W).
            t: Continuous timestep of shape (B,) in [0, 1].
            labels: Class labels of shape (B,).

        Returns:
            Velocity prediction of shape (B, 4, H, W).
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

        This is the main sampling function used during generation.
        Delegates to SiTWrapper.forward_cfg() to avoid duplicating CFG
        channel mode logic.

        Args:
            z: Current noisy latent of shape (B, C, H, W).
            t: Current timestep of shape (B,) or (B, 1, 1, 1) in [0, 1].
            labels: Class labels of shape (B,).

        Returns:
            Velocity prediction of shape (B, C, H, W).
        """
        return self.net.forward_cfg(
            z,
            t,
            labels,
            cfg_scale=self.cfg_scale,
            cfg_channel_mode=self.cfg_channel_mode,
        )

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        num_steps: int | None = None,
        method: Literal["euler", "heun"] = "heun",
        return_latents: bool = False,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate images from class labels using ODE integration.

        Args:
            labels: Class labels of shape (B,).
            num_steps: Number of sampling steps.
            method: ODE solver ("euler" or "heun").
            return_latents: If True, return latents instead of decoded images.
            show_progress: Whether to show progress bar.

        Returns:
            Generated images of shape (B, 3, 256, 256) in [-1, 1] range.
            Or latents of shape (B, 4, 32, 32) if return_latents=True.
        """
        device = labels.device
        batch_size = labels.size(0)
        num_steps = num_steps or self.num_sampling_steps

        # Initialize from pure noise (t=0)
        z = torch.randn(batch_size, 4, self.latent_size, self.latent_size, device=device)

        # Timesteps from 0 to 1 (flow matching: 0=noise, 1=clean)
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc=f"SiT ({method})", total=num_steps)

        if method == "euler":
            z = self._euler_sample(z, timesteps, labels, iterator)
        elif method == "heun":
            z = self._heun_sample(z, timesteps, labels, iterator)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        if return_latents:
            return z

        # Decode latents to images
        images = self.vae.decode(z)
        return images

    def _euler_sample(
        self,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        iterator,
    ) -> torch.Tensor:
        """Euler method for ODE integration.

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

    def _heun_sample(
        self,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        iterator,
    ) -> torch.Tensor:
        """Heun method (improved Euler) for ODE integration.

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
            dt = t_next - t

            t_tensor = torch.full((batch_size,), t, device=z.device)
            t_next_tensor = torch.full((batch_size,), t_next, device=z.device)

            # First velocity estimate
            v1 = self._forward_sample(z, t_tensor, labels)

            # Euler step to get intermediate point
            z_euler = z + dt * v1

            # Second velocity estimate at next timestep
            v2 = self._forward_sample(z_euler, t_next_tensor, labels)

            # Heun step: average of two velocities
            z = z + dt * 0.5 * (v1 + v2)

        return z

    @torch.no_grad()
    def generate_flow_matching_style(
        self,
        labels: torch.Tensor,
        num_steps: int | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate using standard flow matching (alias for generate).

        This method exists for interface compatibility with DiTDenoiser.
        For SiT, it's identical to generate() with Euler method.

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
            return_latents=False,
            show_progress=show_progress,
        )

    def to(self, device: torch.device | str) -> SiTDenoiser:
        """Move denoiser to device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        super().to(device)
        self.vae = self.vae.to(device)
        return self


def load_sit_denoiser(
    checkpoint_path: str | None = None,
    from_pretrained: str | None = None,
    device: str | torch.device = "cuda",
    vae_type: str = "ema",
    cfg_scale: float = 1.5,
    cfg_channel_mode: CFGChannelMode = "first3",
    num_sampling_steps: int = 125,
    model_name: str = "SiT-XL/2",
) -> SiTDenoiser:
    """Load SiT denoiser from checkpoint.

    Args:
        checkpoint_path: Path to local checkpoint (.pt file).
        from_pretrained: HuggingFace Hub model ID (e.g., "nyu-visionx/SiT-collections").
        device: Target device.
        vae_type: VAE variant ("mse" or "ema").
        cfg_scale: Classifier-free guidance scale.
        cfg_channel_mode: How to apply CFG across channels.
            - 'first3': Apply CFG only to first 3 channels (original SiT, default)
            - 'all': Apply CFG to all 4 channels (for TFG research)
        num_sampling_steps: Default number of sampling steps.
        model_name: SiT model variant (e.g., "SiT-XL/2", "SiT-L/2", "SiT-B/2").
            Default is "SiT-XL/2" which matches pretrained checkpoints.

    Returns:
        Initialized SiTDenoiser ready for inference.

    Example:
        >>> # From local checkpoint (default: original paper settings)
        >>> denoiser = load_sit_denoiser(checkpoint_path="checkpoints/sit/SiT-XL-2-256.pt")

        >>> # All channels mode for TFG research
        >>> denoiser = load_sit_denoiser(
        ...     checkpoint_path="checkpoints/sit/SiT-XL-2-256.pt",
        ...     cfg_channel_mode="all",
        ... )
    """
    from jit_tfg.models.sit.model import SiT_models

    device = torch.device(device)

    # Validate model name
    if model_name not in SiT_models:
        raise ValueError(f"Unknown model: {model_name}. Supported models: {list(SiT_models.keys())}")

    # Create SiT model using the specified architecture
    sit = SiT_models[model_name](
        input_size=32,
        num_classes=1000,
        learn_sigma=True,
    ).to(device)

    if checkpoint_path:
        # Load from local checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Handle different checkpoint formats
        if "ema" in ckpt:
            state_dict = ckpt["ema"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        sit.load_state_dict(state_dict)

    elif from_pretrained:
        # Load from HuggingFace Hub or direct URL
        import hashlib
        import os
        import warnings

        # Map model IDs to checkpoint info.
        # Primary source: nyu-visionx/SiT-collections on the HuggingFace Hub —
        # the same repo used by scripts/download_checkpoints.py.
        model_info = {
            "nyu-visionx/SiT-collections": {
                "repo_id": "nyu-visionx/SiT-collections",
                "filename": "SiT-XL-2-256.pt",
                "sha256": "e645b5cd82d1d645cfd9b7c96041b56b2a0aa61faed90c0869e79be02ab55aca",
            },
            # Fallback to original SiT release (official willisma/SiT repo, Dropbox)
            "sihyun-yu/SiT-XL-2-256": {
                "url": "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0",
                "filename": "SiT-XL-2-256.pt",
                "sha256": None,  # No pinned hash; a loud warning is emitted at load time
            },
        }

        if from_pretrained not in model_info:
            raise ValueError(
                f"Unknown model: {from_pretrained}. "
                f"Supported models: {list(model_info.keys())}. "
                f"Alternatively, provide a direct checkpoint_path."
            )

        info = model_info[from_pretrained]
        model_filename = info["filename"]
        expected_sha256 = info.get("sha256")

        cache_dir = os.path.join("checkpoints", "sit")
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, model_filename)

        if not os.path.exists(ckpt_path):
            if "repo_id" in info:
                from huggingface_hub import hf_hub_download

                print(f"Downloading SiT checkpoint {info['repo_id']}:{model_filename} (HuggingFace)...")
                hf_hub_download(
                    repo_id=info["repo_id"],
                    filename=model_filename,
                    local_dir=cache_dir,
                )
            else:
                import urllib.request

                url = info["url"]
                print(f"Downloading SiT checkpoint from {url}...")
                urllib.request.urlretrieve(url, ckpt_path)  # noqa: S310
            print(f"Saved to {ckpt_path}")

        # Verify checksum (mandatory whenever a hash is pinned above)
        if expected_sha256:
            with open(ckpt_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash != expected_sha256:
                raise ValueError(
                    f"Checkpoint integrity check failed for {model_filename}. "
                    f"Expected SHA256: {expected_sha256}, got: {file_hash}. "
                    "The file may be corrupted or tampered with."
                )
        else:
            warnings.warn(
                f"No pinned SHA256 for '{from_pretrained}'; the integrity of "
                f"{ckpt_path} was NOT verified. torch.load(weights_only=True) "
                "restricts deserialization to tensors, but the weights themselves "
                "are unauthenticated.",
                stacklevel=2,
            )

        # SiT release checkpoints are plain state dicts, so the restricted
        # weights_only=True loader suffices (verified against the official
        # SiT-XL-2-256.pt checkpoint).
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        # Handle different checkpoint formats
        if "ema" in ckpt:
            state_dict = ckpt["ema"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        sit.load_state_dict(state_dict)
    else:
        raise ValueError("Either checkpoint_path or from_pretrained must be provided")

    sit.eval()

    # Create wrapper
    wrapper = SiTWrapper(sit, cfg_channel_mode=cfg_channel_mode)

    # Load VAE (reuse from DiT - same Stable Diffusion VAE)
    vae = VAEHandler(vae_type=vae_type, device=device)

    # Create denoiser
    denoiser = SiTDenoiser(
        net=wrapper,
        vae=vae,
        cfg_scale=cfg_scale,
        cfg_channel_mode=cfg_channel_mode,
        num_sampling_steps=num_sampling_steps,
    ).to(device)

    return denoiser
