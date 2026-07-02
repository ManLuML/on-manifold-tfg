"""Training-free guidance across diffusion prediction targets.

Evaluation-only code release for the ECCV 2026 paper "Not All Prediction
Targets Keep Training-Free Diffusion Guidance on the Manifold". The package
studies how a pretrained model's prediction target (x / v / epsilon) affects
training-free guidance (TFG), using four pretrained Diffusion Transformers:
JiT (x, pixel), DiT (epsilon, latent), SiT (v, latent), and PixelFlow
(v, pixel). No training code is included; checkpoints are downloaded via
``scripts/download_checkpoints.py``.

Package Structure:
    jit_tfg/
    ├── models/       - Pretrained-model wrappers with a shared denoiser API
    │   ├── jit/          - JiT (x-prediction, pixel space)
    │   ├── dit/          - DiT (epsilon-prediction, latent space) + VAE
    │   ├── sit/          - SiT (v-prediction, latent space)
    │   └── pixelflow/    - PixelFlow (v-prediction, pixel space, multi-stage)
    ├── tfg/          - Training-free guidance
    │   ├── config.py         - TFGConfig hyperparameters
    │   ├── unified_sampler.py - UnifiedSampler for all four model families
    │   ├── calibration.py    - x-space <-> v-space guidance calibration
    │   └── guiders/          - Energy functions (classifier, inverse problems)
    └── evaluation/   - Metrics
        ├── generation/   - FID / Inception Score
        └── guidance/     - Guidance validity (classifier accuracy) evaluators

TFG Integration:
    The tfg/ module provides Training-Free Guidance adapted for Flow Matching:
    - TFGConfig: 7 hyperparameters controlling guidance behavior
    - UnifiedSampler: Unified sampler supporting JiT, DiT, SiT, PixelFlow
    - BaseGuider: Abstract class for energy functions

References:
    - JiT Paper: "Back to Basics: Let Denoising Generative Models Denoise"
    - TFG Paper: "TFG: Unified Training-Free Guidance for Diffusion Models"
    - DiT: https://github.com/facebookresearch/DiT
    - SiT: https://github.com/willisma/SiT
    - PixelFlow: https://github.com/ShoufaChen/PixelFlow
"""

from jit_tfg.tfg import BaseGuider, TFGConfig, UnifiedSampler

__all__ = [
    "BaseGuider",
    "TFGConfig",
    "UnifiedSampler",
]
