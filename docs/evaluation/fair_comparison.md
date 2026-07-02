# Evaluation: Fair Comparison

This document outlines the "Fair Comparison" protocol used to evaluate different diffusion model architectures (JiT, DiT, SiT, PixelFlow) under comparable computational budgets.

## Principles

To ensure fair comparison across different architectures and sampling methods, we normalize by **Number of Function Evaluations (NFE)** rather than raw sampling steps.

- **Heun (2nd-order ODE)**: 2 function evaluations per step (predictor + corrector).
- **Euler (1st-order ODE)**: 1 function evaluation per step.
- **DDPM / DDIM**: 1 function evaluation per step.

## Guided experiments (paper protocol, NFE ≈ 100)

All guided (TFG) runs in the paper target a computational budget of **≈ 100 NFE**:

| Model | Steps | Sampler | NFE | CFG | VAE | Notes |
|-------|-------|---------|-----|-----|-----|-------|
| **JiT-H/16** | 50 | Heun | **100** | 2.2 | - | Pixel space, 2 NFE/step |
| **SiT-XL/2** | 50 | Heun | **100** | 1.5 | ema | Latent space, 2 NFE/step |
| **DiT-XL/2** | **100** | DDPM | **100** | 1.5 | ema | Latent space, 1 NFE/step |
| **PixelFlow** | 30 × 4 stages | Euler | **120** | 2.4 | - | Pixel space, per-stage steps |

> [!NOTE]
> **Why 100 steps for DiT?**
> Since DDPM is a 1st-order sampler (1 NFE/step) and Heun is a 2nd-order solver (2 NFE/step), running DiT for 50 steps would give only 50 NFE — a computational disadvantage compared to JiT/SiT (100 NFE). DiT therefore uses 100 DDPM steps in guided runs (pass `--nfe 100`) to ensure parity in total model evaluations. PixelFlow's cascaded 4-stage design has a fixed per-stage step count, so its closest match is 30 steps × 4 stages = 120 NFE.

## CFG-only published-optimum defaults

The per-model defaults in `experiments/model_configs.json` reproduce each
model's published-optimum CFG-only protocol (used for the CFG-only FID
baselines), not the NFE-matched guided budget above:

| Model | Published-optimum settings | FID |
|-------|----------------------------|-----|
| **DiT-XL/2** | DDPM, 250 steps (NFE 250), CFG=1.5, VAE=ema | 2.27 |
| **SiT-XL/2** | Heun, 125 steps (NFE 250), CFG=1.5, VAE=ema | 2.06 |
| **JiT-H/16** | Heun, 50 steps (NFE 100), CFG=2.2 | 1.86 |
| **PixelFlow** | Euler, 30 steps × 4 stages (NFE 120), CFG=2.4 | 1.98 |

For JiT-H and PixelFlow the two configurations coincide; for DiT and SiT,
guided paper-protocol runs must override the defaults with `--nfe 100`.

## Inverse Problem Evaluation

For inverse problems (Gaussian deblur, 4× super-resolution), the same model configurations from `model_configs.json` are used but with **different metrics and guidance presets**.

| Aspect | Classification Guidance | Inverse Problem Guidance |
|--------|------------------------|--------------------------|
| **Script** | `finegrained_bird_tfg.py` | `deblur_sr.py` |
| **Guider** | Classifier (log p(y\|x)) | InverseProblemGuider (-\|\|y - A(x)\|\|₂) |
| **Metrics** | FID, IS, Validity | LPIPS, PSNR, SSIM |
| **Reference** | None (generate from scratch) | ImageNet val images (degraded + noisy) |
| **Models** | 256×256 only | 256×256 only |

> [!NOTE]
> Inverse-problem TFG presets differ per task (deblur vs super-resolution). They
> follow the conventions of the upstream Training-Free-Guidance inverse-problem
> tasks; the shipped presets live in `experiments/deblur_sr.py`. Run that script
> with `--help` for the full set of task-specific flags.
