# Research Context

This document provides background on the research goals and methodology of the JiT-TFG project.

## Background

This codebase is built upon the official implementation of the JiT paper: **"Back to Basics: Let Denoising Generative Models Denoise"**.

The primary goal is to apply and experiment with **Train-Free Guidance (TFG)** on the JiT model.

## Research Objective

We aim to conduct a systematic study to understand how different formulations of diffusion models affect their extensibility, particularly for techniques like Train-Free Guidance.

## The 3×3 Experiment Matrix

We evaluate 9 specific combinations formed by a 3×3 matrix of prediction targets and loss functions:

### Prediction Targets

1. **x-prediction** (predicting clean data \\(x_0\\))
   - Model directly outputs estimated clean image
   - Intuitive interpretation
   - May have numerical stability issues near \\(t=1\\)

2. **v-prediction** (predicting velocity \\(v\\))
   - Model outputs \\(v = x_0 - \epsilon\\)
   - Balanced noise levels across timesteps
   - Good numerical properties

3. **epsilon-prediction** (predicting noise \\(\epsilon\\))
   - Model outputs the noise to be removed
   - Traditional DDPM formulation
   - Well-studied in literature

### Loss Functions

1. **x-loss** (loss computed in \\(x\\)-space)
   \\[
   \mathcal{L}_x = \mathbb{E}_{t,x,\epsilon} \left[ \| x - \hat{x} \|^2 \right]
   \\]

2. **v-loss** (loss computed in \\(v\\)-space)
   \\[
   \mathcal{L}_v = \mathbb{E}_{t,x,\epsilon} \left[ \| v - \hat{v} \|^2 \right]
   \\]

3. **epsilon-loss** (loss computed in \\(\epsilon\\)-space)
   \\[
   \mathcal{L}_\epsilon = \mathbb{E}_{t,x,\epsilon} \left[ \| \epsilon - \hat{\epsilon} \|^2 \right]
   \\]

### The 9 Combinations

| Prediction Target | x-loss | v-loss | epsilon-loss |
|-------------------|--------|--------|--------------|
| **x-prediction** | x-pred / x-loss | x-pred / v-loss | x-pred / eps-loss |
| **v-prediction** | v-pred / x-loss | v-pred / v-loss | v-pred / eps-loss |
| **epsilon-prediction** | eps-pred / x-loss | eps-pred / v-loss | eps-pred / eps-loss |

## Flow Matching Framework

The codebase uses flow matching as the training framework:

### Forward Process

The forward process defines a path from data \\(x\\) to noise \\(\epsilon\\):

\\[
z_t = t \cdot x + (1-t) \cdot \epsilon
\\]

- At \\(t=0\\): \\(z_0 = \epsilon\\) (pure noise)
- At \\(t=1\\): \\(z_1 = x\\) (pure data)

### Velocity Field

The velocity field is the derivative with respect to \\(t\\):

\\[
v = \frac{dz_t}{dt} = x - \epsilon = \frac{x - z_t}{1-t}
\\]

### ODE Sampling

Starting from \\(z_0 \sim \mathcal{N}(0, \sigma^2 I)\\), we solve:

\\[
\frac{dz_t}{dt} = v_\theta(z_t, t)
\\]

using numerical integration (Euler or Heun) up to \\(t=1\\).

## Current Implementation

The current codebase implements **x-prediction with v-loss**:

```python
# Forward process
z = t * x + (1 - t) * epsilon

# Target velocity
v = (x - z) / (1 - t)

# Model prediction (x-prediction)
x_pred = model(z, t, labels)

# Derive predicted velocity
v_pred = (x_pred - z) / (1 - t)

# Loss in v-space
loss = ((v - v_pred) ** 2).mean()
```

This combination was chosen as the baseline because:

1. **x-prediction**: Direct interpretation of output
2. **v-loss**: Good numerical stability across all timesteps

## Classifier-Free Guidance (CFG)

The model supports CFG for conditional generation:

\\[
\hat{v} = v_\text{uncond} + s \cdot (v_\text{cond} - v_\text{uncond})
\\]

where \\(s\\) is the guidance scale.

### CFG Training

During training, labels are randomly dropped with probability `label_drop_prob`:

```python
def drop_labels(labels):
    drop = torch.rand(labels.shape[0]) < label_drop_prob
    return torch.where(drop, num_classes, labels)  # num_classes = null class
```

### CFG Interval

CFG can be applied selectively within a timestep interval:

```python
# Only apply CFG when t in [interval_min, interval_max]
cfg_scale_effective = cfg_scale if (t < interval_max) and (t > interval_min) else 1.0
```

## Train-Free Guidance (TFG)

The research goal is to extend the model with Train-Free Guidance techniques. TFG aims to:

1. Guide generation toward desired properties without retraining
2. Work across different conditioning modalities
3. Maintain image quality while improving controllability

### TFG Research Questions

This codebase is designed to investigate:

1. **Which prediction target is most amenable to TFG?**
   - Does x-prediction offer better gradient flow?
   - Is v-prediction more numerically stable for guidance?

2. **Which loss function produces the best representation for guidance?**
   - Does training loss affect the smoothness of the learned manifold?
   - Are some combinations more sensitive to guidance perturbations?

3. **What is the optimal combination for TFG extensibility?**
   - Balancing generation quality and guidability
   - Trade-offs between different configurations

## Evaluation Metrics

### FID (Fréchet Inception Distance)

Measures the similarity between generated and real image distributions:

\\[
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})
\\]

Lower FID indicates better image quality and diversity.

### IS (Inception Score)

Measures quality and diversity of generated images:

\\[
\text{IS} = \exp\left(\mathbb{E}_x \left[ D_{KL}(p(y|x) \| p(y)) \right]\right)
\\]

Higher IS indicates better class-conditional generation.

### Manifold-Aware Metrics (Precision, Density)

For TFG evaluation, FID and classifier accuracy alone are insufficient --- they cannot distinguish realistic on-manifold samples from adversarial-like off-manifold samples that fool classifiers. We additionally use:

- **Precision** (Kynkäänniemi et al., NeurIPS 2019): Fraction of generated samples within the support of the real data distribution, computed via k-NN in DINOv2 feature space.
- **Density** (Naeem et al., ICML 2020): Continuous measure of how deeply samples reside within the real manifold support.

These metrics are critical for demonstrating x-prediction's advantage: maintaining manifold fidelity under TFG guidance. See [Manifold-Aware Evaluation Metrics](../evaluation/manifold-metrics.md) for details and [Metric Literature Survey](../evaluation/metric-survey.md) for the full survey.

## Experimental Protocol

### Training Configuration

| Parameter | Default Value | Notes |
|-----------|---------------|-------|
| Model | JiT-L/16 | 307M parameters |
| Image size | 256×256 | ImageNet resolution |
| Batch size | 1024 | Across 8 GPUs |
| Learning rate | 5e-5 × batch/256 | Scaled base LR |
| Epochs | 400 | Full training |
| EMA decay | 0.9999 | For stable sampling |

### Evaluation Protocol

1. Generate 50,000 images (50 per class for ImageNet)
2. Compute FID against ImageNet training set
3. Compute Inception Score
4. Compare across all 9 configurations

## References

### Core Papers

- **JiT**: "Back to Basics: Let Denoising Generative Models Denoise"
- **Flow Matching**: "Flow Matching for Generative Modeling"
- **CFG**: "Classifier-Free Diffusion Guidance"
- **DiT**: "Scalable Diffusion Models with Transformers"

### Related Work

- **SiT**: [https://github.com/willisma/SiT](https://github.com/willisma/SiT)
- **Lightning-DiT**: [https://github.com/hustvl/LightningDiT](https://github.com/hustvl/LightningDiT)
- **ADM**: "Diffusion Models Beat GANs on Image Synthesis"

## Next Steps

- [Timestep Conventions](timestep-conventions.md): **Critical** - Understanding DiT/SiT/JiT time conventions
- [Architecture](architecture.md): Technical details of the model
- [API Reference](../api/index.md): Implementation documentation
- [Testing](../development/testing.md): How to run experiments
