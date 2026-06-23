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

## Paper Reproduction Mode

While the "Fair Comparison" settings are used for our internal research and cross-model analysis, the `--reproduce_paper` flag allows strictly reproducing valid numbers from the original papers.

| Model | `--reproduce_paper` Settings | Target FID |
|-------|------------------------------|------------|
| **DiT-XL/2** | Steps=**250**, CFG=1.0, VAE=ema | 2.27 |
| **SiT-XL/2** | Steps=**250**, CFG=1.0, VAE=ema | 2.06 |

When `--reproduce_paper` is used, the default "Fair Comparison" settings are overridden to match the original publication's protocol.

## Inverse Problem Evaluation

For inverse problems (Gaussian deblur, 4× super-resolution), the same model configurations from `model_configs.json` are used but with **different metrics and guidance presets**.

| Aspect | Classification Guidance | Inverse Problem Guidance |
|--------|------------------------|--------------------------|
| **Script** | `imagenet_tfg.py` / `finegrained_bird_tfg.py` | `deblur_sr.py` |
| **Guider** | Classifier (log p(y\|x)) | InverseProblemGuider (-\|\|y - A(x)\|\|₂) |
| **Metrics** | FID, IS, Validity | LPIPS, PSNR, SSIM |
| **Reference** | None (generate from scratch) | ImageNet val images (degraded + noisy) |
| **Models** | 256×256 only | 256×256 only |

> [!NOTE]
> Inverse problem TFG presets differ per task (deblur vs super-resolution) and come from the original TFG paper's scripts (`gaussian_deblur.sh`, `super_resolution.sh`). See `experiments/AGENTS.md` Section 6 for details.
