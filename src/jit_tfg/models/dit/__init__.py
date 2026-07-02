"""DiT (Diffusion Transformer) integration module for JiT-TFG.

This module provides DiT model integration for Train-Free Guidance (TFG)
experiments, enabling fair comparison between epsilon-prediction (DiT),
v-prediction (SiT), and x-prediction (JiT) models.

Main Components:
    - DiT: The Diffusion Transformer model architecture
    - DiTWrapper: Adapter for JiT-TFG interface compatibility
    - DiTDenoiser: DDPM-based denoiser for TFG integration
    - VAEHandler: Stable Diffusion VAE for latent <-> pixel conversion
    - DDPMSchedule: Precomputed diffusion schedule coefficients

Quick Start:
    >>> from jit_tfg.models.dit import load_dit_denoiser
    >>> denoiser = load_dit_denoiser(from_pretrained="facebook/DiT-XL-2-256")
    >>> labels = torch.tensor([207, 360], device="cuda")
    >>> images = denoiser.generate(labels)

For TFG experiments (using UnifiedSampler):
    >>> from jit_tfg.models.dit import DiTDenoiser
    >>> from jit_tfg.tfg import UnifiedSampler, TFGConfig
    >>> from jit_tfg.tfg.guiders import LatentClassifierGuider
    >>>
    >>> denoiser = load_dit_denoiser(...)
    >>> guider = LatentClassifierGuider(classifier, denoiser.vae, targets=[207])
    >>> config = TFGConfig(rho=1.0, mu=0.5)
    >>> sampler = UnifiedSampler("DiT", denoiser, config)
    >>> images = sampler.generate(cfg_labels=labels, guidance=guider, tfg_targets=labels)
"""

from jit_tfg.models.dit.denoiser import DiTDenoiser, load_dit_denoiser
from jit_tfg.models.dit.diffusion import DDPMSchedule
from jit_tfg.models.dit.model import DiT, DiT_models
from jit_tfg.models.dit.vae import VAEHandler
from jit_tfg.models.dit.wrapper import DiTWrapper

__all__ = [
    "DDPMSchedule",
    # Models
    "DiT",
    "DiTDenoiser",
    # Wrappers
    "DiTWrapper",
    "DiT_models",
    # Utilities
    "VAEHandler",
    # Loading
    "load_dit_denoiser",
]
