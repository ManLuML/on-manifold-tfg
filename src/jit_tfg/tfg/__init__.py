"""Train-Free Guidance (TFG) module for diffusion models.

This module provides the UnifiedSampler for integrating TFG guidance with
all supported diffusion models: JiT, DiT, SiT, and PixelFlow.

Main Components:
    - UnifiedSampler: Unified sampler supporting all models and guidance modes
    - TFGConfig: Configuration dataclass for TFG hyperparameters
    - BaseGuider: Abstract base class for guidance energy functions
    - create_classifier_guider: Factory function for classifier guiders

Usage:
    >>> from jit_tfg.tfg import UnifiedSampler, TFGConfig
    >>> from jit_tfg.tfg.guiders import create_classifier_guider
    >>>
    >>> # Create sampler
    >>> sampler = UnifiedSampler(
    ...     model_type="JiT",
    ...     denoiser=denoiser,
    ...     tfg_config=TFGConfig(rho=1.0, mu=0.5),
    ... )
    >>>
    >>> # Create guider
    >>> guider = create_classifier_guider(classifier, targets=[207], denoiser=denoiser)
    >>>
    >>> # Generate with TFG
    >>> images = sampler.generate(cfg_labels=labels, guidance=guider)
"""

from jit_tfg.tfg.config import TFGConfig
from jit_tfg.tfg.guiders import BaseGuider, create_classifier_guider
from jit_tfg.tfg.unified_sampler import UnifiedSampler

__all__ = [
    "BaseGuider",
    "TFGConfig",
    "UnifiedSampler",
    "create_classifier_guider",
]
