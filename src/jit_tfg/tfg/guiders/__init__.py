"""Guider modules for TFG guidance.

This module provides various guider implementations for steering
diffusion sampling toward desired properties.

Available Guiders:
    ImageNet Classification:
        - ImageClassifierGuider: Pixel-space classifier guidance
        - LatentClassifierGuider: Latent-space classifier guidance (with VAE decode)
        - LatentMultiTargetGuider: Latent-space with per-sample targets

    Fine-grained Classification:
        - HuggingFaceClassifierGuider: HuggingFace-based fine-grained guidance

    Inverse Problems:
        - InverseProblemGuider: Measurement consistency guidance (deblur, super-resolution)
        - GaussianBlurOperator: Gaussian blur degradation operator
        - SuperResolutionOperator: 4x bicubic downsampling operator

Factory Functions:
    - create_classifier_guider: Auto-selects guider based on denoiser type
    - create_inverse_guider: Creates degradation operator + guider from task name

Utility Functions:
    - load_imagenet_classifier: Load pretrained ImageNet classifier
"""

from jit_tfg.tfg.guiders.base import BaseGuider
from jit_tfg.tfg.guiders.finegrained import HuggingFaceClassifierGuider
from jit_tfg.tfg.guiders.imagenet import (
    ImageClassifierGuider,
    LatentClassifierGuider,
    LatentMultiTargetGuider,
    create_classifier_guider,
    load_imagenet_classifier,
)
from jit_tfg.tfg.guiders.inverse import (
    GaussianBlurOperator,
    InverseProblemGuider,
    SuperResolutionOperator,
    create_inverse_guider,
)

__all__ = [
    "BaseGuider",
    "GaussianBlurOperator",
    "HuggingFaceClassifierGuider",
    "ImageClassifierGuider",
    "InverseProblemGuider",
    "LatentClassifierGuider",
    "LatentMultiTargetGuider",
    "SuperResolutionOperator",
    "create_classifier_guider",
    "create_inverse_guider",
    "load_imagenet_classifier",
]
