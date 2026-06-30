# Evaluation: Fair Comparison

This document outlines the "Fair Comparison" protocol used to evaluate different diffusion model architectures (JiT, SiT, DiT) under comparable computational budgets.

## Principles

To ensure fair comparison across different architectures and sampling methods, we normalize by **Number of Function Evaluations (NFE)** rather than raw sampling steps.

- **ODEs (Heun)**: 2 function evaluations per step (Predictor + Corrector).
- **DDIM**: 1 function evaluation per step.

## Default Settings

We target a computational budget of **100 NFE** for all models.

| Model | Steps | Sampler | NFE | CFG | VAE | Notes |
|-------|-------|---------|-----|-----|-----|-------|
| **JiT** | 50 | Heun | **100** | 3.0 | - | Pixel space, 2 NFE/step |
| **SiT** | 50 | Heun | **100** | 4.0 | ema | Latent space, 2 NFE/step |
| **DiT** | **100** | DDIM | **100** | 4.0 | ema | Latent space, 1 NFE/step |

> [!NOTE]
> **Why 100 steps for DiT?**
> Originally, DiT baselines often used 50 steps. However, since DDIM is a 1st-order solver (1 NFE/step) and Heun is a 2nd-order solver (2 NFE/step), running DiT for 50 steps resulted in only 50 NFE, giving it a computational disadvantage compared to JiT/SiT (100 NFE). We now default DiT to 100 steps to ensure parity in total model evaluations.

## Reproducing original-paper numbers

The fair-comparison settings above are used for cross-model analysis. To instead
match the protocol of each model's original publication, override the per-model
sampler defaults with `--nfe` / `--sampling_method` (see the per-model defaults
in `experiments/model_configs.json`):

| Model | Original-paper settings | Target FID |
|-------|-------------------------|------------|
| **DiT-XL/2** | Steps=**250**, CFG=1.0, VAE=ema | 2.27 |
| **SiT-XL/2** | Steps=**250**, CFG=1.0, VAE=ema | 2.06 |

These reproduce the original publication's protocol rather than the NFE-matched
fair-comparison budget.

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
