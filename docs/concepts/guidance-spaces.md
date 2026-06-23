# Guidance Spaces: Design Philosophy

## Overview

Three guidance spaces control how TFG corrections are applied to the ODE trajectory. Each has a distinct theoretical heritage and practical behavior.

| Space | Heritage | dt Handling | Scaling | Philosophy |
|-------|----------|-------------|---------|------------|
| **x** (default) | TFG/DDPM | None | `t_next/t`, `t_next` | Position correction |
| **v** | Flow Guidance | Implicit | `(1-t)/t` | Velocity modification |
| **v2** | Our work | Implicit | `(1-t)/t` per NFE | Per-NFE velocity modification |

---

## x-space: TFG/DDPM Heritage

### Theory

Original TFG (Algorithm 1, Line 9, DDPM formulation):
```
x_prev += Δ_t / √α_t + √ᾱ_{t-1} · Δ_0
```

Original implementation (`edm/methods/tfg.py:129`):
```python
x_prev += Delta_t / alpha_t ** 0.5 + Delta_0 * alpha_prod_t_prev ** 0.5
```

Flow matching adaptation:
```
z_next += (t_next / t) · δ_t + t_next · δ_0
```

### Key Properties

- **No dt in guidance**: DDPM uses discrete timesteps where dt doesn't naturally exist. Guidance is a direct position correction: "move the sample to position X".
- **Step-count dependent**: Because there's no dt, cumulative guidance scales with `num_steps`. Changing `num_steps` requires re-tuning `rho`/`mu`. This matches original TFG behavior.
- **DDPM correspondence**: `1/√α_t` → `t_next/t` (variance guidance), `√ᾱ_{t-1}` → `t_next` (mean guidance).

### When to Use

- Backward compatibility with published TFG hyperparameters
- Direct translation from DDPM-based experiments
- Default choice for initial experiments

---

## v-space: Flow Guidance Heritage

### Theory

From "On the Guidance of Flow Matching" (Feng et al., ICML 2025), the covariance-preconditioned guidance `g^{cov-G}`:

```
g_t = λ_t · ∇_{x_t} J(x̂_1)
λ_t = -(1-t)/t   (for affine path α=t, β=1-t)
```

This theoretical scaling is derived from Assumption 3.3 (Jacobian trick for affine Gaussian paths) of Feng et al.

### Formula

```
λ_t = (1 - t) / t_clamped
v_guided = v + λ_t · δ_t + δ_0
z_next = z + dt · v_guided
```

### Key Properties

- **dt implicit**: Flow Matching is continuous ODE `dz/dt = v(z, t)`. "Change velocity by V" → `dt × V = position change`. The dt normalization makes guidance more robust to step count changes.
- **lambda_t asymmetry**: Strong guidance at high noise (`t→0`: `λ_t → ∞`), weak at low noise (`t→1`: `λ_t → 0`). This matches intuition: easy to redirect in noise, near manifold at clean.
- **lambda_t on delta_t only**: `g^{cov-G}` corresponds to delta_t (variance guidance). delta_0 (TFG mean guidance) is a separate concept not in the Flow Guidance framework—it's an x_0-space direct shift, not requiring covariance preconditioning.
- **Heun: v1-only guidance**: Flow Guidance defines one guidance vector per step. Per-NFE guidance is v2-space's contribution. v-space faithfully reproduces the Flow Guidance framework.

### When to Use

- Theoretically grounded experiments
- Flow Guidance paper reproduction
- When step-count stability is desired

---

## v2-space: Per-NFE Velocity Modification (Our Contribution)

### Theory

In higher-order solvers like Heun, two velocity evaluations happen per step. Instead of applying guidance once, each velocity is guided individually before the solver averages them.

### Formula

```
# v1 guidance at (z, t)
λ_t = (1 - t) / t_clamped
δ_t_1, δ_0_1 = guidance_deltas(z, t)
v1_guided = v1 + λ_t · δ_t_1 + δ_0_1

# Guided Euler step for corrector state
z_euler_guided = z + dt · v1_guided

# v2 at GUIDED intermediate state (not unguided z_euler!)
v2 = v_theta(z_euler_guided, t_next)
λ_{t_next} = (1 - t_next) / t_next_clamped
δ_t_2, δ_0_2 = guidance_deltas(z_euler_guided, t_next)
v2_guided = v2 + λ_{t_next} · δ_t_2 + δ_0_2

# Heun with guided velocities
z_next = z + dt · 0.5 · (v1_guided + v2_guided)
```

This is equivalent to applying Heun directly to the guided ODE `dz/dt = v_theta(z,t) + g(z,t)`,
which is the theoretically correct discretization for second-order integration of the guided flow.

### Key Differences from v-space

| Aspect | v-space | v2-space |
|--------|---------|----------|
| Guidance computations | 1 (at t) | 2 (at t and t_next) |
| Lambda values | Single λ_t | λ_t ≠ λ_{t_next} |
| Guidance input state | z | z (v1) and z_euler_guided (v2) |
| ODE being solved | Unguided + post-hoc correction | Guided ODE directly |

### When to Use

- Heun sampling with theoretical scaling
- Research into higher-order guidance integration
- Requires `sampling_method="heun"`

---

## Step-Count Dependency Analysis

### Mathematical Analysis

**x-space** (no dt):
```
Cumulative guidance ≈ Σ_i [(t_{i+1}/t_i)·δ_t + t_{i+1}·δ_0]
```
Each step adds a guidance term with no dt factor, so total guidance ∝ `num_steps`.

**v-space** (dt implicit):
```
Cumulative guidance ≈ Σ_i [dt · (λ_{t_i}·δ_t + δ_0)]
```
Each step's contribution is scaled by `dt = 1/num_steps`, providing natural normalization.

### Numerical Example

For identical `rho=1.0`, `mu=0.0`:

| num_steps | x-space cumulative | v-space cumulative |
|-----------|-------------------|-------------------|
| 10 | ~10 × per-step | ~1.0 × integral |
| 50 | ~50 × per-step | ~1.0 × integral |
| 100 | ~100 × per-step | ~1.0 × integral |

This means:
- **x-space**: Changing `num_steps` from 50 to 100 roughly doubles cumulative guidance → must halve `rho`.
- **v-space**: Changing `num_steps` has little effect on cumulative guidance → `rho` is more portable.

### Practical Implications

1. **Hyperparameter portability**: v/v2-space hyperparameters are more transferable across step counts.
2. **Original TFG reproduction**: x-space is required for fair comparison with published TFG results.
3. **Ablation design**: When comparing guidance spaces, fix `num_steps` to avoid confounding.

---

## Analysis: Why v-Space Underperforms with Same Hyperparameters

### The Magnitude Gap

With identical `rho` and `mu`, v-space applies dramatically less cumulative guidance than x-space due to the `dt` normalization factor. This is **by design** (velocity modification vs position correction), not a bug.

**Per-step guidance displacement** (50 steps, t=0.5, dt=0.02):

| | x-space | v-space | Ratio |
|---|---------|---------|-------|
| delta_t | `(t_next/t)·δ_t = 1.04·δ_t` | `dt·(1-t)/t·δ_t = 0.02·δ_t` | 52x |
| delta_0 | `t_next·δ_0 = 0.52·δ_0` | `dt·δ_0 = 0.02·δ_0` | 26x |

**Cumulative over full trajectory** (N=50, linspace(0,1), t_eps=0.05):
- delta_t (flow_guidance): x-space ~52, v-space ~3.2 → **~16x ratio**
- delta_t (identity): x-space ~52, v-space ~1.0 → **~52x ratio**
- delta_0: x-space ~25.5, v-space ~1.0 → **~25x ratio**

Note: flow_guidance ratio is lower (~16x) than identity (~52x) because lambda_t amplification at clamped early steps (t<t_eps) boosts v-space delta_t.

Note: delta_t and delta_0 ratios differ because `lambda_t` only applies to delta_t in v-space. Therefore `rho` and `mu` need **separate** scaling factors when calibrating v-space.

Use `experiments/analyze_guidance_spaces.py` to compute exact equivalence values.

### lambda_t Theory Mismatch for x-Prediction

The `lambda_t = (1-t)/t` scaling was derived from the Flow Guidance paper (Feng et al., ICML 2025) assuming:

1. **v-prediction model**: `x̂_1 = z_t + (1-t)·v_θ(z_t, t)`
2. **Jacobian structure**: `∂x̂_1/∂z_t = I + (1-t)·J_v` (has identity component)

For **x-prediction** (JiT), the Jacobian is `∂x̂_1/∂z_t = J_x` (no identity component). The `(1-t)/t` scaling, which amplifies to 19x at t=0.05, may **overcorrect** x-prediction's already-stable gradients (Theorem 3.1: `‖∇ E(x̂^(x))‖ ≤ L·‖J_x‖`, bounded regardless of t).

The `lambda_mode` config option enables ablation:
- `"auto"` (default): x-prediction → identity, v/e-prediction → flow_guidance
- `"flow_guidance"`: Standard (1-t)/t (derived for v-prediction)
- `"identity"`: lambda=1.0 (step-count stable, no time-dependent amplification)

### Schedule Inversion: flow_guidance × rho_schedule="increase"

**Critical interaction**: `flow_guidance` lambda with `rho_schedule="increase"` (default) **inverts the schedule**:

```
rho(t) = base_rho × t              (increase schedule)
lambda_t = (1-t)/t                  (flow_guidance)
effective = rho(t) × lambda_t = base_rho × (1-t)    ← DECREASE!
```

| Configuration | t=0.1 | t=0.5 | t=0.9 | Effective Schedule |
|---|---|---|---|---|
| x-space + increase | 0.12 | 0.52 | 0.92 | increase ✓ |
| v-space + flow_guidance + increase | 0.018 | 0.010 | 0.002 | **DECREASE** ✗ |
| v-space + identity + increase | 0.002 | 0.010 | 0.018 | increase ✓ |

This inversion cannot be compensated by scaling rho — it changes the **shape** of the guidance profile. The `"auto"` default avoids this by using `"identity"` for x-prediction models.

### Jacobian Fix (v0.x.x)

The flow matching TFG delta computation previously used `_forward_sample` (which has `@torch.no_grad()`), causing the model Jacobian to be excluded from `delta_t`. This was fixed by introducing `_forward_flow_with_grad` which preserves gradient tracking through the model, matching the original TFG implementation. DiT was already correct (uses its own `_dit_forward_with_cfg` without `@torch.no_grad()`).

### Guidance Space Selection Guidelines

| Model | pred_target | Recommended space | Notes |
|-------|-------------|-------------------|-------|
| JiT | x | **x-space** | Stable gradients, published TFG hyperparams apply directly |
| SiT | v | v-space (experiment) | Theoretical match with Flow Guidance derivation |
| DiT | ε | x-space (only option) | DDPM discrete timesteps, v-space not supported |
| PixelFlow | v | x-space (default) | Multi-stage adds complexity |

---

## Calibration Guide: Fair v-Space Comparison

### Auto-Calibration

Use `--auto_calibrate` to automatically compute equivalent v-space rho/mu:

```bash
# v-space with auto-calibrated rho/mu (matches x-space cumulative guidance)
uv run python experiments/finegrained_bird_tfg.py --model jit-b-16 \
    --guidance_mode dps --guidance_space v --auto_calibrate
```

### Manual Calibration

```bash
# Compute equivalence values
uv run python experiments/analyze_guidance_spaces.py \
    --num_steps 50 --x_rho 0.5 --x_mu 0.5 --lambda_mode identity

# Apply manually
uv run python experiments/finegrained_bird_tfg.py --model jit-b-16 \
    --guidance_mode dps --guidance_space v \
    --rho_override 29.5 --mu_override 14.1
```

### Magnitude Ratios by Lambda Mode

| lambda_mode | delta_t ratio (x/v) | delta_0 ratio (x/v) |
|---|---|---|
| flow_guidance | ~16x | ~25x |
| identity | ~52x | ~25x |

Note: delta_0 ratio is identical because lambda_t only applies to delta_t.
flow_guidance ratio is lower because clamped lambda at t<t_eps boosts v-space.

---

## Research Questions

- How does guidance space interact with prediction target (x/v/ε)?
- Does v-space's theoretical scaling actually improve FID/IS?
- What is the optimal guidance space for each (model, sampler) pair?
- Does delta_0 scaling (with/without lambda_t) affect guidance quality?
- With calibrated rho/mu, does v-space match x-space FID?
- Does `lambda_mode="identity"` + v-space outperform standard v-space for x-prediction?
- Does schedule inversion (flow_guidance + increase) explain v-space underperformance?
