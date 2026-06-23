"""JiT-TFG: Train-Free Guidance experiments on the JiT diffusion model.

This package provides an implementation of the JiT (Just image Transformer)
model from "Back to Basics: Let Denoising Generative Models Denoise" paper,
extended with experimental support for Train-Free Guidance (TFG).

The codebase is structured to investigate how different formulations of
diffusion models (prediction targets and loss functions) affect their
extensibility for guidance techniques.

Package Structure:
    jit_tfg/
    ├── jit/
    │   ├── model.py      - JiT transformer architecture
    │   ├── denoiser.py   - Flow-matching denoiser wrapper
    │   ├── engine.py     - Training and evaluation functions
    │   ├── prepare_ref.py - Reference dataset preparation
    │   └── utils/        - Utility functions
    │       ├── crop.py       - Image cropping utilities
    │       ├── lr_sched.py   - Learning rate scheduling
    │       ├── misc.py       - Distributed training utilities
    │       └── model_util.py - Model utilities (RoPE, embeddings)
    └── tfg/
        ├── __init__.py   - TFG module exports
        ├── config.py     - TFG hyperparameter configuration
        ├── unified_sampler.py - Unified sampler for all models
        ├── utils.py      - Utility functions (gradient rescaling)
        └── guiders/      - Energy function implementations
            ├── __init__.py
            └── base.py   - Base guider class

Research Focus:
    Evaluating 9 combinations from a 3x3 matrix of:
    - Prediction targets: x-prediction, v-prediction, epsilon-prediction
    - Loss functions: x-loss, v-loss, epsilon-loss

TFG Integration:
    The tfg/ module provides Training-Free Guidance adapted for Flow Matching:
    - TFGConfig: 7 hyperparameters controlling guidance behavior
    - UnifiedSampler: Unified sampler supporting JiT, DiT, SiT, PixelFlow
    - BaseGuider: Abstract class for energy functions

References:
    - JiT Paper: "Back to Basics: Let Denoising Generative Models Denoise"
    - TFG Paper: "TFG: Unified Training-Free Guidance for Diffusion Models"
    - SiT: https://github.com/willisma/SiT
    - Lightning-DiT: https://github.com/hustvl/LightningDiT
"""

from jit_tfg.tfg import BaseGuider, TFGConfig, UnifiedSampler

__all__ = [
    "BaseGuider",
    "TFGConfig",
    "UnifiedSampler",
]
