"""SiT (Scalable Interpolant Transformers) integration module for JiT-TFG.

This module provides SiT model integration for Train-Free Guidance (TFG)
experiments, enabling fair comparison between v-prediction (SiT),
epsilon-prediction (DiT), and x-prediction (JiT) models.

SiT is the v-prediction baseline for the paper's central hypothesis:
- x-prediction (JiT): Direct, bounded error → Best for TFG
- v-prediction (SiT): Bounded error like x → Second best
- ε-prediction (DiT): O(1/t) error amplification → Worst for TFG

Key Insight: SiT Uses Same Time Convention as JiT
    Unlike DiT (which uses DDPM convention), SiT's time convention
    already matches JiT:
    - t=0: noise
    - t=1: clean data
    - Forward: x_t = t*x + (1-t)*ε (identical to JiT's flow matching)

    This simplifies integration significantly - no timestep inversion needed!

Main Components:
    - SiT: The Scalable Interpolant Transformer model architecture
    - SiTWrapper: Adapter for JiT-TFG interface compatibility
    - SiTDenoiser: Flow matching denoiser for TFG integration
    - VAEHandler: Stable Diffusion VAE for latent <-> pixel conversion
    - LinearPath: Flow matching interpolation path

Quick Start:
    >>> from jit_tfg.models.sit import load_sit_denoiser
    >>> denoiser = load_sit_denoiser(checkpoint_path="path/to/SiT-XL-2.pt")
    >>> labels = torch.tensor([207, 360], device="cuda")
    >>> images = denoiser.generate(labels)

For TFG experiments (using UnifiedSampler):
    >>> from jit_tfg.models.sit import SiTDenoiser, load_sit_denoiser
    >>> from jit_tfg.tfg import UnifiedSampler, TFGConfig
    >>> from jit_tfg.tfg.guiders import LatentClassifierGuider
    >>>
    >>> denoiser = load_sit_denoiser(...)
    >>> guider = LatentClassifierGuider(classifier, denoiser.vae, targets=[207])
    >>> config = TFGConfig(rho=1.0, mu=0.5)
    >>> sampler = UnifiedSampler("SiT", denoiser, config)
    >>> images = sampler.generate(cfg_labels=labels, guidance=guider, tfg_targets=labels)

Key Differences from DiT:
    | Aspect              | DiT                    | SiT                    |
    |---------------------|------------------------|------------------------|
    | Time convention     | Opposite (inversion)   | Same as JiT            |
    | Prediction target   | ε (epsilon)            | v (velocity)           |
    | Diffusion framework | DDPM (discrete)        | Flow matching (cont.)  |
    | Sampler             | DDPM/DDIM              | ODE/SDE (Euler/Heun)   |
    | x₀ recovery         | (x_t-√(1-α̅)ε)/√α̅       | x_t + (1-t)*v          |
    | Schedule needed     | Yes (DDPMSchedule)     | No (direct integration)|
"""

# Re-export VAEHandler for convenience (shared with DiT)
from jit_tfg.models.dit.vae import VAEHandler
from jit_tfg.models.sit.denoiser import SiTDenoiser, load_sit_denoiser
from jit_tfg.models.sit.model import SiT, SiT_models
from jit_tfg.models.sit.transport import LinearPath
from jit_tfg.models.sit.wrapper import SiTWrapper

__all__ = [
    "LinearPath",
    "SiT",
    "SiTDenoiser",
    "SiTWrapper",
    "SiT_models",
    "VAEHandler",
    "load_sit_denoiser",
]
