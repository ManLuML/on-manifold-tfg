# Timestep Conventions and Sampling Methods

This document explains the critical differences in timestep conventions and sampling methods across DiT, SiT, and JiT models. Understanding these differences is essential for correct sampling and guidance across the three prediction targets.

## Quick Reference Table

| Property | DiT | SiT | JiT |
|----------|-----|-----|-----|
| **Framework** | DDPM (Discrete) | Flow Matching (Continuous) | Flow Matching (Continuous) |
| **Time Domain** | t ∈ {0, 1, ..., 999} (discrete) | t ∈ [0, 1] (continuous) | t ∈ [0, 1] (continuous) |
| **t = 0** | Clean data (x₀) | **NOISE (ε)** | **NOISE (ε)** |
| **t = 1 (or T-1)** | Pure noise (xₜ) | Clean data (x) | Clean data (x) |
| **Prediction Target** | ε (noise) | v (velocity) preferred | x (clean data) |
| **Latent/Pixel** | Latent (32×32×4) | Latent (32×32×4) | Pixel (256×256×3) |
| **Sampling Steps** | 250 (DDPM), 50 (DDIM) | 250 (Euler/Heun) | 50 (Heun) |
| **Default CFG Scale** | 1.5 | 1.5 | 2.9 |

## Critical Note

> **SiT and JiT use the SAME time convention (implementation)!**
>
> - **SiT (implementation)**: t=0 → noise, t=1 → clean (same as JiT)
> - **JiT**: t=0 → noise, t=1 → clean
> - **DiT**: t=0 → clean, t=999 → noise (opposite direction)
>
> **Note**: The SiT paper describes the convention as t=0 clean, t=1 noise,
> but the upstream SiT implementation (willisma/SiT, `transport/path.py`; vendored
> here under `src/jit_tfg/models/sit/transport/`) uses α_t = t and σ_t = 1-t,
> resulting in z_t = t·x + (1-t)·ε, which is the JiT convention.

---

## DiT (Diffusion Transformer)

### Framework: DDPM (Denoising Diffusion Probabilistic Models)

DiT uses the standard DDPM formulation with discrete timesteps.

### Forward Process (Diffusion)

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Where:
- $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$ is the cumulative product of $(1-\beta)$
- $\beta_t$ follows a linear schedule from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$
- $T = 1000$ discrete timesteps

### Prediction Target: ε (Noise)

The network learns to predict the noise added to the clean image:

$$\hat{\epsilon} = \epsilon_\theta(x_t, t, c)$$

Clean data recovery:

$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

### DDPM Sampling Algorithm

```python
# Start from pure noise
x_T ~ N(0, I)

for t in reversed(range(T)):  # T-1, T-2, ..., 0
    eps = model(x_t, t, class_label)

    # Apply CFG if scale > 1
    if cfg_scale > 1:
        eps_uncond = model(x_t, t, null_label)
        eps = eps_uncond + cfg_scale * (eps - eps_uncond)

    # Predict x_0
    x0_pred = (x_t - sqrt(1 - alpha_bar[t]) * eps) / sqrt(alpha_bar[t])

    # Compute posterior mean
    if t > 0:
        posterior_mean = (sqrt(alpha_bar[t-1]) * beta[t] / (1 - alpha_bar[t])) * x0_pred \
                       + (sqrt(alpha[t]) * (1 - alpha_bar[t-1]) / (1 - alpha_bar[t])) * x_t
        posterior_var = beta[t] * (1 - alpha_bar[t-1]) / (1 - alpha_bar[t])
        x_{t-1} = posterior_mean + sqrt(posterior_var) * z  # z ~ N(0, I) if t > 0
    else:
        x_0 = x0_pred
```

### DDIM Sampling (Deterministic)

DDIM allows fewer steps with deterministic sampling:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \hat{\epsilon} + \sigma_t \cdot z$$

With $\sigma_t = 0$ for deterministic sampling (η=0).

### Code Reference

```python
# src/jit_tfg/models/dit/diffusion/ddpm_schedule.py
class DDPMSchedule:
    def __init__(self, betas: Tensor, ...):
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # t=0: alpha_bar ≈ 1 (clean), t=999: alpha_bar ≈ 0 (noise)
```

---

## SiT (Scalable Interpolant Transformers)

### Framework: Flow Matching / Stochastic Interpolants

SiT uses the interpolant framework. **In the implementation**, t=0 is noise and t=1 is clean data (same as JiT).

> **Note on Paper vs Implementation Discrepancy**:
> The SiT paper describes the convention as α_t = 1-t, σ_t = t (t=0 clean, t=1 noise).
> However, the upstream SiT implementation (willisma/SiT, `transport/path.py`;
> vendored here under `src/jit_tfg/models/sit/transport/`) uses α_t = t, σ_t = 1-t,
> which means **t=0 is noise and t=1 is clean** (same as JiT).

### Forward Process (Interpolation)

$$z_t = \alpha_t \cdot x + \sigma_t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

For **Linear Interpolant** (as implemented):
- $\alpha_t = t$ (increases from 0 to 1)
- $\sigma_t = 1 - t$ (decreases from 1 to 0)

Therefore:
$$z_t = t \cdot x + (1-t) \cdot \epsilon$$

At boundaries:
- **t = 0**: $z_0 = \epsilon$ (pure noise)
- **t = 1**: $z_1 = x$ (clean data)

**This is the same convention as JiT.**

### Prediction Target: v (Velocity)

The velocity is defined as the time derivative of the interpolation path:

$$v_t = \frac{d z_t}{d t} = x - \epsilon$$

The network predicts:
$$\hat{v} = v_\theta(z_t, t, c)$$

### Clean Data Recovery from Velocity

From $z_t = t \cdot x + (1-t) \cdot \epsilon$ and $v = x - \epsilon$:

We can derive:
$$\hat{x} = z_t + (1-t) \cdot \hat{v}$$

**Derivation:**
- $z_t = t \cdot x + (1-t) \cdot \epsilon$
- $v = x - \epsilon$, so $\epsilon = x - v$
- $z_t = t \cdot x + (1-t)(x - v) = x - (1-t)v$
- Therefore: $x = z_t + (1-t)v$

**This is the same formula as JiT.**

### Euler ODE Sampling

```python
# Start from noise at t=0
z_0 ~ N(0, I)
dt = 1/num_steps  # Positive: moving from t=0 to t=1

for i, t in enumerate(linspace(0, 1, num_steps)):
    v = model(z_t, t, class_label)

    # Apply CFG
    if cfg_scale > 1:
        v_uncond = model(z_t, t, null_label)
        v = v_uncond + cfg_scale * (v - v_uncond)

    # Euler step (moving toward t=1, i.e., clean data)
    z_{t+dt} = z_t + dt * v
```

### Code Reference

```python
# src/jit_tfg/models/sit/denoiser.py
# v -> x_0: x = z + (1-t) * v  (SiT uses the same convention as JiT)
x_pred = z + (1.0 - t) * v_pred

# src/jit_tfg/models/sit/transport/path.py - ICPlan (vendored from willisma/SiT)
def compute_alpha_t(self, t):
    return t, 1  # α_t = t (data coefficient)

def compute_sigma_t(self, t):
    return 1 - t, -1  # σ_t = 1 - t (noise coefficient)
```

---

## JiT (Just Image Transformer)

### Framework: Flow Matching (Reversed Time Convention)

JiT uses flow matching but with the **opposite** time direction from SiT.

### Forward Process (Interpolation)

$$z_t = t \cdot x + (1-t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

At boundaries:
- **t = 0**: $z_0 = \epsilon$ (pure noise)
- **t = 1**: $z_1 = x$ (clean data)

### Prediction Target: x (Clean Data)

JiT directly predicts the clean data:

$$\hat{x} = x_\theta(z_t, t, c)$$

This is argued to be more stable because:
1. Direct mapping to data manifold
2. No division by small values near boundaries
3. Semantically meaningful gradients for guidance

### Velocity and Other Targets

JiT can also output v or ε, computed from x̂:

$$\hat{v} = \hat{x} - z_t / (1-t) \cdot t = \hat{x} - \frac{t \cdot z_t}{1-t}$$

Actually, for JiT's convention:
- $z_t = t \cdot x + (1-t) \cdot \epsilon$
- $\epsilon = (z_t - t \cdot x) / (1-t)$
- $v = dx/dt$ direction... Let me be more careful.

The velocity field for JiT's ODE is:
$$\frac{d z_t}{d t} = x - \epsilon$$

So:
$$\hat{v} = \hat{x} - \hat{\epsilon}$$

Where:
$$\hat{\epsilon} = \frac{z_t - t \cdot \hat{x}}{1-t}$$

### Clean Data Recovery

From x-prediction: **Direct** (no conversion needed)
$$\hat{x} = x_\theta(z_t, t, c)$$

From ε-prediction:
$$\hat{x} = \frac{z_t - (1-t) \cdot \hat{\epsilon}}{t}$$

⚠️ **Numerical instability** as t → 0 due to division by small t.

From v-prediction:
$$\hat{x} = z_t + (1-t) \cdot \hat{v}$$

### Euler/Heun ODE Sampling

```python
# Start from noise at t=0
z_0 ~ N(0, I)
dt = 1/num_steps  # Positive: moving from t=0 to t=1

for i, t in enumerate(linspace(0, 1, num_steps)):
    x_pred = model(z_t, t, class_label)

    # Apply CFG
    if cfg_scale > 1:
        x_uncond = model(z_t, t, null_label)
        x_pred = x_uncond + cfg_scale * (x_pred - x_uncond)

    # Compute velocity: v = x - epsilon
    v = x_pred - (z_t - t * x_pred) / (1 - t).clamp_min(eps)

    # Euler step (moving toward t=1, i.e., clean data)
    z_{t+dt} = z_t + dt * v

    # Heun correction (optional)
    # ...
```

### Code Reference

```python
# src/jit_tfg/models/jit/denoiser.py
class Denoiser:
    def _predict_x0(self, z_t: Tensor, t: Tensor, ...) -> Tensor:
        if self.pred_target == "x":
            return self.model(z_t, t, y, ...)  # Direct
        elif self.pred_target == "e":
            eps = self.model(z_t, t, y, ...)
            return (z_t - (1-t) * eps) / t.clamp_min(self.t_eps)  # Unstable!
        elif self.pred_target == "v":
            v = self.model(z_t, t, y, ...)
            return z_t + (1-t) * v
```

---

## Time Convention Comparison Diagram

```
DiT (DDPM):     t=0 (clean) ──────────────────────────> t=999 (noise)
                    ↑                                        ↑
                  x_0                                      x_T
                (sample)                               (pure noise)

SiT (Flow):     t=0 (noise) ──────────────────────────> t=1 (clean)
                    ↑                                        ↑
                  z_0 = ε                               z_1 = x
               (sampling start)                          (data)

JiT (Flow):     t=0 (noise) ──────────────────────────> t=1 (clean)
                    ↑                                        ↑
                  z_0 = ε                               z_1 = x
               (sampling start)                          (data)
```

**Note**: SiT and JiT use the same convention (implementation-wise). Both sample from t=0 to t=1.

---

## TFG Guidance: Time Schedule Implications

When applying TFG (Training-Free Guidance), the schedule weighting depends on the time convention:

### "Increase" Schedule

```python
def get_schedule_weight(t: float, schedule: str) -> float:
    if schedule == "increase":
        return t  # More guidance as t increases
    elif schedule == "decrease":
        return 1 - t
    else:  # "constant"
        return 1.0
```

**Interpretation differs by model:**

| Schedule | DiT | SiT | JiT |
|----------|-----|-----|-----|
| `increase` | More guidance near noise (high t) | More guidance near **clean data** (high t) | More guidance near **clean data** (high t) |
| `decrease` | More guidance near clean (low t) | More guidance near **noise** (low t) | More guidance near **noise** (low t) |

⚠️ **For both SiT and JiT, "increase" means more guidance near the end of sampling (near clean data), which is typically desired for TFG.**

### Code Reference

```python
# src/jit_tfg/tfg/unified_sampler.py (SiT and JiT)
# Both use t=1 as clean, so 'increase' = more weight at high t (clean).
def _get_schedule_weight(self, t: float, schedule: str) -> float:
    if schedule == "increase":
        return t
    elif schedule == "decrease":
        return 1.0 - t
    return 1.0
```

**Note**: SiT and JiT share one schedule-weight function because they share the
same time convention; the `UnifiedSampler` handles both (and DiT's reversed DDPM
convention) in one place.

---

## Summary: Key Formulas for Each Model

### Forward Diffusion / Interpolation

| Model | Formula | t=0 | t=1 (or T) |
|-------|---------|-----|------------|
| DiT | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ | $x_0$ (clean) | $\approx \epsilon$ (noise) |
| SiT | $z_t = t x + (1-t) \epsilon$ | $z_0 = \epsilon$ (noise) | $z_1 = x$ (clean) |
| JiT | $z_t = t x + (1-t) \epsilon$ | $z_0 = \epsilon$ (noise) | $z_1 = x$ (clean) |

**Note**: SiT and JiT use the same forward process formula.

### Clean Data Recovery from Predictions

| Model | Target | Formula |
|-------|--------|---------|
| DiT | ε | $\hat{x}_0 = (x_t - \sqrt{1-\bar{\alpha}_t} \hat{\epsilon}) / \sqrt{\bar{\alpha}_t}$ |
| SiT | v | $\hat{x} = z_t + (1-t) \hat{v}$ |
| JiT | x | $\hat{x} = x_\theta(z_t, t)$ (direct) |
| JiT | v | $\hat{x} = z_t + (1-t) \hat{v}$ |
| JiT | ε | $\hat{x} = (z_t - (1-t) \hat{\epsilon}) / t$ (⚠️ unstable) |

**Note**: SiT (v-prediction) and JiT (v-prediction) use the same x₀ recovery formula.

### ODE / SDE Step Direction

| Model | Sampling Start | Sampling Direction | dt sign |
|-------|----------------|-------------------|---------|
| DiT | t = T (noise) | T → 0 | negative |
| SiT | t = 0 (noise) | 0 → 1 | positive |
| JiT | t = 0 (noise) | 0 → 1 | positive |

**Note**: SiT and JiT sample in the same direction (positive dt).

---

## Common Pitfalls

### 1. Confusing SiT Paper vs Implementation

```python
# WRONG: Assuming SiT paper convention (t=0 clean, t=1 noise)
x0_pred = z_t - t * v  # Paper convention, NOT what the code does!

# CORRECT: SiT implementation uses same formula as JiT
x0_pred = z_t + (1-t) * v  # Both SiT and JiT use this
```

### 2. Incorrect Schedule Direction

```python
# For TFG with JiT or SiT, if you want more guidance near the END of sampling:
rho_schedule = "increase"  # t high = near clean data = end of sampling

# For TFG with DiT, if you want more guidance near the END of sampling:
rho_schedule = "decrease"  # t low = near clean data = end of DDPM sampling
```

### 3. Time Conversion Between Frameworks

```python
# Converting DiT discrete t to JiT/SiT continuous t
def dit_to_flow_time(t_discrete: int, T: int = 1000) -> float:
    # DiT t=0 is clean, JiT/SiT t=1 is clean
    # DiT t=999 is noise, JiT/SiT t=0 is noise
    return 1.0 - t_discrete / (T - 1)

# SiT and JiT use the same time convention - no conversion needed!
def sit_to_jit_time(t_sit: float) -> float:
    return t_sit  # Same convention
```

### 4. Numerical Stability Near Boundaries

```python
# JiT/SiT: Division by (1-t) near t=1
eps_from_x = (z_t - t * x_pred) / (1 - t).clamp_min(1e-5)  # ✓ Safe

# JiT/SiT: Division by t for x from eps (near t=0)
x_from_eps = (z_t - (1-t) * eps_pred) / t.clamp_min(1e-5)  # ⚠️ Use t_eps=0.05

# JiT/SiT: x from velocity (safe, no division)
x_from_v = z_t + (1-t) * v_pred  # ✓ No division needed, same for both
```

---

## Next Steps

- [Guidance Spaces](guidance-spaces.md): How TFG corrections are applied to the trajectory
- [Schedule Normalization Deep Dive](schedule-normalization-deep-dive.md): DDPM→flow schedule conversion
- [Research Context](research-context.md): Research goals and methodology

The unified TFG sampler is implemented in `src/jit_tfg/tfg/unified_sampler.py`;
read the source for the per-model sampling and guidance API.
