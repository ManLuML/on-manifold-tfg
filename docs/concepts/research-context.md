# Research Context

This document provides background on the research goals and methodology behind
*Not All Prediction Targets Keep Training-Free Diffusion Guidance on the
Manifold* (ECCV 2026).

> This release is **evaluation / inference only**. The four diffusion models are
> used from their authors' pretrained checkpoints; no model training is shipped.
> The discussion of training objectives below is conceptual background — it
> explains *what the pretrained models learned*, not a training pipeline you run
> here.

## Background

Training-Free Guidance (TFG) steers a pretrained diffusion model toward desired
properties (a target class, a measurement constraint) **without retraining**.
All TFG methods compute their guidance from a clean-data estimate \\(\hat{x}\\).
This work asks how the model's **prediction target** — what the network outputs:
clean data \\(x\\), velocity \\(v\\), or noise \\(\epsilon\\) — affects whether
guidance keeps samples on the data manifold.

## Research Objective

We study how the prediction target governs manifold preservation under TFG. The
central claim is a strict **error-amplification hierarchy** in recovering
\\(\hat{x}\\):

- **x-prediction**: the model outputs \\(\hat{x}\\) directly — no amplification.
- **v-prediction**: \\(\hat{x}\\) recovered from velocity — bounded amplification.
- **ε-prediction**: \\(\hat{x}\\) recovered by dividing by \\(t\\) — amplification
  grows unboundedly at high noise.

Because TFG's correction is computed from \\(\hat{x}\\), the fidelity of that
estimate governs whether guidance fails *gracefully* (target missed, image still
realistic) or *catastrophically* (collapsed, off-manifold artifacts).

## The Four Pretrained Models

To compare the three prediction targets at comparable unconditional quality, we
evaluate four pretrained Diffusion Transformers (CFG-only FID ≈ 2 on ImageNet
256×256):

| Model | Prediction target | Space |
|-------|:-----------------:|:-----:|
| **JiT-H/16** | x | Pixel |
| **PixelFlow-XL** | v | Pixel |
| **SiT-XL/2** | v | Latent |
| **DiT-XL/2** | ε | Latent |

These span all three targets across both pixel and latent space, so the
prediction-target effect can be isolated from the choice of representation.

## Prediction Targets (Conceptual Background)

The three targets parameterize the same denoising problem differently:

1. **x-prediction** (clean data \\(x_0\\))
   - Network directly outputs the estimated clean image.
   - No recovery formula needed — \\(\hat{x}\\) is the raw output.

2. **v-prediction** (velocity \\(v = x_0 - \epsilon\\))
   - \\(\hat{x}\\) recovered as \\(z_t + (1-t)\,\hat{v}\\) (flow-matching convention).
   - Bounded error amplification across timesteps.

3. **ε-prediction** (noise \\(\epsilon\\))
   - Traditional DDPM parameterization.
   - \\(\hat{x}\\) recovered by dividing out the noise scale, which blows up at
     high noise.

## Flow Matching Framework

The flow-matching models (JiT, SiT, PixelFlow) define a path from data \\(x\\) to
noise \\(\epsilon\\):

\\[
z_t = t \cdot x + (1-t) \cdot \epsilon
\\]

- At \\(t=0\\): \\(z_0 = \epsilon\\) (pure noise)
- At \\(t=1\\): \\(z_1 = x\\) (pure data)

The velocity field is the derivative with respect to \\(t\\):

\\[
v = \frac{dz_t}{dt} = x - \epsilon = \frac{x - z_t}{1-t}
\\]

Sampling solves \\(\frac{dz_t}{dt} = v_\theta(z_t, t)\\) from noise to data with a
numerical integrator (Euler or Heun). DiT instead uses the discrete DDPM
formulation with the reversed time convention — see
[Timestep Conventions](timestep-conventions.md).

## Classifier-Free Guidance (CFG)

All four models support CFG for conditional generation. In velocity form:

\\[
\hat{v} = v_\text{uncond} + s \cdot (v_\text{cond} - v_\text{uncond})
\\]

where \\(s\\) is the guidance scale. CFG can be applied selectively within a
timestep interval:

```python
# Only apply CFG when t in [interval_min, interval_max]
cfg_scale_effective = cfg_scale if (interval_min < t < interval_max) else 1.0
```

CFG is the baseline ("CFG-only") against which guided (TFG) runs are compared.

## Training-Free Guidance (TFG)

On top of a frozen pretrained model, TFG injects a gradient correction toward a
target. The research questions this release lets you investigate:

1. **Which prediction target keeps guided samples on the manifold?** The headline
   result is a 5.2-point Child-FID gap between x- and ε-prediction (**32.9** vs.
   **38.1**) at matched classifier accuracy — manifold damage invisible to
   standard FID/accuracy reporting.
2. **Does the hierarchy hold across guidance tasks?** The same ordering appears in
   class guidance, style transfer, and inverse problems (deblur / super-resolution).

## Evaluation Metrics

### FID (Fréchet Inception Distance)

\\[
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})
\\]

Lower is better. We report **Child-FID (C-FID)** — FID against the guided child
class — as the headline metric, since it exposes off-manifold damage that
whole-distribution FID and classifier accuracy miss.

### IS (Inception Score)

\\[
\text{IS} = \exp\left(\mathbb{E}_x \left[ D_{KL}(p(y|x) \| p(y)) \right]\right)
\\]

Higher indicates better class-conditional generation.

### Manifold-Aware Metrics

FID and classifier accuracy alone cannot distinguish realistic on-manifold
samples from adversarial-like off-manifold samples that fool classifiers. We
additionally consider **Precision** (Kynkäänniemi et al., NeurIPS 2019) and
**Density** (Naeem et al., ICML 2020) in DINOv2 feature space. See
[Manifold-Aware Evaluation Metrics](../evaluation/manifold-metrics.md) and the
[Metric Literature Survey](../evaluation/metric-survey.md).

## References

### Core Papers

- **JiT**: "Back to Basics: Let Denoising Generative Models Denoise" (Li & He, 2025)
- **DiT**: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
- **SiT**: "SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers" (Ma et al., 2024)
- **PixelFlow**: "PixelFlow: Pixel-Space Generative Models with Flow" (Chen et al., 2025)
- **Flow Matching**: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- **CFG**: "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
- **TFG**: "Training-Free Guidance" (upstream codebase: YWolfeee/Training-Free-Guidance)

## Next Steps

- [Timestep Conventions](timestep-conventions.md): **Critical** — DiT/SiT/JiT time conventions.
- [Architecture](architecture.md): Technical details of the model.
- [Guidance Spaces](guidance-spaces.md): How TFG corrections are applied to the trajectory.
