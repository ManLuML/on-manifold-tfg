"""PixelFlow integration module for JiT-TFG.

This module provides PixelFlow model integration for Train-Free Guidance (TFG)
experiments, enabling comparison between v-prediction models in both latent
and pixel space.

PixelFlow is the v-prediction, pixel-space baseline for the paper's hypothesis:
- x-prediction (JiT): Direct, bounded error -> Best for TFG
- v-prediction (SiT/PixelFlow): Bounded error like x -> Second best
- epsilon-prediction (DiT): O(1/t) error amplification -> Worst for TFG

Experimental matrix coverage:
    |                | epsilon-pred | v-pred    | x-pred |
    |----------------|-------------|-----------|--------|
    | Latent space   | DiT-XL/2    | SiT-XL/2  | -      |
    | Pixel space    | ADM-G       | PixelFlow | JiT    |

Key Insight: PixelFlow Uses Same Time Convention as JiT
    Like SiT (and unlike DiT), PixelFlow's time convention matches JiT:
    - t=0: noise
    - t=1: clean data
    - Forward: x_t = t*x + (1-t)*epsilon (identical to JiT's flow matching)

    This simplifies integration - no timestep inversion needed!

Key Difference from SiT: Pixel Space
    Unlike SiT which operates in VAE latent space (4, 32, 32),
    PixelFlow operates directly in pixel space (3, 256, 256).
    This means:
    - No VAE encoder/decoder needed
    - Direct classifier guidance (no decode_with_grad)
    - Use ImageClassifierGuider instead of LatentClassifierGuider

Main Components:
    - PixelFlowModel: The PixelFlow transformer architecture (re-exported)
    - PixelFlowWrapper: Adapter for JiT-TFG interface compatibility
    - PixelFlowDenoiser: Flow matching denoiser for TFG integration
    - LinearPath: Flow matching interpolation path (shared with SiT)

Quick Start:
    >>> from jit_tfg.models.pixelflow import load_pixelflow_denoiser
    >>> denoiser = load_pixelflow_denoiser(checkpoint_path="path/to/checkpoint")
    >>> labels = torch.tensor([207, 360], device="cuda")
    >>> images = denoiser.generate(labels)  # Direct pixel images!

For TFG experiments (using UnifiedSampler):
    >>> from jit_tfg.models.pixelflow import PixelFlowDenoiser, load_pixelflow_denoiser
    >>> from jit_tfg.tfg import UnifiedSampler, TFGConfig
    >>> from jit_tfg.tfg.guiders import ImageClassifierGuider
    >>>
    >>> denoiser = load_pixelflow_denoiser(...)
    >>> guider = ImageClassifierGuider(classifier, targets=[207])
    >>> config = TFGConfig(rho=1.0, mu=0.5)
    >>> sampler = UnifiedSampler("PixelFlow", denoiser, config)
    >>> images = sampler.generate(cfg_labels=labels, guidance=guider, tfg_targets=labels)

Comparison with SiT and DiT:
    | Aspect              | DiT                    | SiT              | PixelFlow        |
    |---------------------|------------------------|------------------|------------------|
    | Time convention     | Opposite (inversion)   | Same as JiT      | Same as JiT      |
    | Prediction target   | epsilon                | v (velocity)     | v (velocity)     |
    | Operating space     | Latent (4, 32, 32)     | Latent (4, 32, 32)| Pixel (3, 256, 256)|
    | VAE needed          | Yes                    | Yes              | No               |
    | Guider type         | LatentClassifierGuider | LatentClassifierGuider | ImageClassifierGuider |
    | x0 recovery         | (x_t-sqrt(1-a)*e)/sqrt(a)| x_t + (1-t)*v   | x_t + (1-t)*v   |
"""

from jit_tfg.models.pixelflow.denoiser import PixelFlowDenoiser, load_pixelflow_denoiser
from jit_tfg.models.pixelflow.model import PixelFlowModel
from jit_tfg.models.pixelflow.wrapper import PixelFlowWrapper

# Re-export LinearPath for convenience (shared with SiT)
from jit_tfg.models.sit.transport import LinearPath

__all__ = [
    "LinearPath",
    "PixelFlowDenoiser",
    "PixelFlowModel",
    "PixelFlowWrapper",
    "load_pixelflow_denoiser",
]
