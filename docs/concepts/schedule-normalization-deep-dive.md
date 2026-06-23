# Schedule Normalization Deep Dive

This document provides a detailed analysis of TFG's Schedule Normalization: what it is, why it's needed, and the mathematical justification for converting from the original DDPM implementation to Flow Matching.

---

## 0. Schedule Definition in TFG Paper (Direct Quotes)

### 0.1 Paper Section 4 (Page 7) - Structure Analysis

The TFG paper decomposes time-dependent vector parameters **ρ**, **μ** into a scalar and a "structure":

> "Fortunately, below we demonstrate that, if we **decompose ρ into ρ̄ · sρ(t)** (same for μ) where ρ̄ is a scalar and sρ(t) is a "**structure**" (a non-negative function) **such that Σt sρ(t) = T**, then some structures are consistently better than others regardless of the other hyper-parameters."
>
> — TFG Paper, Section 4, Page 7

### 0.2 Paper Equation (8) - Three Structure Definitions

```
s(t) = αt / Σ(t=1 to T) αt           (increase)
s(t) = (1 - αt) / Σ(t=1 to T)(1-αt)  (decrease)
s(t) = 1                              (constant)
```

> "These structures are selected to be qualitatively different, while each is justified to be reasonable under certain conditions."
>
> — TFG Paper, Equation (8), Page 7

**Key Point**: Each structure is **normalized so that its sum equals T**. For example:
- `increase`: αt divided by sum of all αt → Σs(t) = T
- `decrease`: (1-αt) divided by sum of all (1-αt) → Σs(t) = T
- `constant`: s(t) = 1 → Σs(t) = T

### 0.3 Original TFG Code Implementation (edm/methods/tfg.py:56)

```python
def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
    if self.args.rho_schedule == 'decrease':
        scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
    elif self.args.rho_schedule == 'increase':
        scheduler = alpha_prod_ts / alpha_prod_t_prevs
    elif self.args.rho_schedule == 'constant':
        scheduler = torch.ones_like(alpha_prod_ts)

    # Normalization formula: base * s[t] * N / Σs[i]
    return self.args.rho * scheduler[t] * len(scheduler) / scheduler.sum()
```

In this code:
- `len(scheduler)` = N (total number of timesteps)
- `scheduler.sum()` = Σs[i] (sum of all schedule values)
- **Normalization factor** = `N / Σs[i]`

---

## 1. Original TFG Implementation Analysis

### 1.1 Original Code (edm/methods/tfg.py)

```python
def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
    if self.args.rho_schedule == 'decrease':    # β_t
        scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
    elif self.args.rho_schedule == 'increase':  # α_t
        scheduler = alpha_prod_ts / alpha_prod_t_prevs
    elif self.args.rho_schedule == 'constant':  # 1
        scheduler = torch.ones_like(alpha_prod_ts)

    return self.args.rho * scheduler[t] * len(scheduler) / scheduler.sum()
```

### 1.2 Variable Meanings

| Variable | Meaning | Description |
|----------|---------|-------------|
| `alpha_prod_ts` | ᾱ_t (alpha bar) | DDPM's cumulative product of alphas |
| `alpha_prod_t_prevs` | ᾱ_{t-1} | Alpha bar at previous timestep |
| `scheduler` | s[t] | Schedule value at time t (array) |
| `len(scheduler)` | N | Total number of timesteps |
| `scheduler.sum()` | Σs[i] | Sum of all schedule values |

### 1.3 Schedule Definitions

In DDPM convention (t=0: clean, t=T-1: noisy):

| Schedule | Formula | Meaning |
|----------|---------|---------|
| `increase` | α_t = ᾱ_t / ᾱ_{t-1} | Increases as denoising progresses (high near clean) |
| `decrease` | β_t = 1 - α_t | Decreases as denoising progresses (high near noisy) |
| `constant` | 1 | Always the same |

---

## 2. Mathematical Meaning of `scheduler.sum()`

### 2.1 Normalization Formula Breakdown

Original TFG's normalization formula:
```
value = base_value × s[t] × (N / Σs[i])
                           ^^^^^^^^^^^^
                           normalizer
```

Here, `N / Σs[i]` is the **normalization factor (normalizer)**.

### 2.2 Why is This Normalization Needed?

**Goal**: Ensure that **average guidance strength equals `base_value`** regardless of which schedule is used.

**Proof**:
```
Average value over all timesteps
= (1/N) × Σ_t value[t]
= (1/N) × Σ_t (base_value × s[t] × N / Σs[i])
= base_value × (1/N) × N × (Σ_t s[t]) / (Σ_i s[i])
= base_value × (Σ_t s[t]) / (Σ_i s[i])
= base_value × 1
= base_value ✓
```

### 2.3 Concrete Example (N=100 timesteps)

**Constant schedule** (s[t] = 1 for all t):
```
scheduler.sum() = 100
normalizer = 100 / 100 = 1.0
value = rho × 1 × 1.0 = rho
average = rho ✓
```

**Increase schedule** (s[t] = α_t, approximately 0.99 ~ 0.5 range):
```
scheduler.sum() ≈ 75 (assumed)
normalizer = 100 / 75 ≈ 1.33
value[t] = rho × α_t × 1.33
average = rho × (1.33 × 75 / 100) = rho ✓
```

---

## 3. Conversion to Flow Matching

### 3.1 Key Differences

| Item | Original TFG (DDPM) | Flow Matching (JiT/SiT) |
|------|---------------------|-------------------------|
| Time range | t ∈ {0, 1, ..., N-1} (discrete) | t ∈ [0, 1] (continuous) |
| Time convention | t=0: clean, t=N-1: noisy | t=0: noisy, t=1: clean |
| Schedule function | s[t] = α_t (non-linear) | s(t) = t (linear) |
| Sum/Average | Σ (summation) | ∫ (integral) |

### 3.2 Conversion to Integration

**Discrete → Continuous correspondence**:

| Discrete (DDPM) | Continuous (Flow Matching) |
|-----------------|---------------------------|
| `len(scheduler)` = N | `∫₀¹ 1 dt` = 1 (integration range) |
| `scheduler.sum()` = Σs[i] | `∫₀¹ s(t) dt` (integral value) |
| `N / Σs[i]` | `1 / ∫₀¹ s(t) dt` |

### 3.3 Integral Values for Each Schedule

**Constant schedule**: s(t) = 1
```
∫₀¹ 1 dt = [t]₀¹ = 1 - 0 = 1
normalizer = 1 / 1 = 1
```

**Increase schedule**: s(t) = t
```
∫₀¹ t dt = [t²/2]₀¹ = 1/2 - 0 = 0.5
normalizer = 1 / 0.5 = 2
```

**Decrease schedule**: s(t) = 1-t
```
∫₀¹ (1-t) dt = [t - t²/2]₀¹ = (1 - 1/2) - 0 = 0.5
normalizer = 1 / 0.5 = 2
```

### 3.4 Why "×2"?

**Key Insight**:
- For the `increase` schedule, s(t) = t has an average value of 0.5
- To make the average equal to 1, multiply by 2
- Therefore `normalizer = 2`

**Visual Understanding**:
```
Strength
    ^
2.0 |         * (t=1)
    |       /
1.0 |-----/------ Average should be 1.0
    |   /
0.0 |--*---------> t
    0           1

Area under s(t) = t graph = 0.5
To make area 1.0, double the height → s(t) = 2t
```

---

## 4. Verification of Our Implementation

### 4.1 Flow Matching Implementation (unified_sampler.py)

```python
def _get_schedule_value(self, base_value, schedule, t, normalize=True):
    if schedule == "constant":
        return base_value
    elif schedule == "increase":
        normalizer = 2.0 if normalize else 1.0
        return base_value * t * normalizer
    elif schedule == "decrease":
        normalizer = 2.0 if normalize else 1.0
        return base_value * (1 - t) * normalizer
```

### 4.2 DiT Implementation (Identical to Original TFG)

```python
def _get_dit_rho(self, t_idx, alpha_prod_ts, alpha_prod_t_prevs):
    alpha_ratio = alpha_prod_ts / alpha_prod_t_prevs

    if schedule == "decrease":
        scheduler = 1 - alpha_ratio
    elif schedule == "increase":
        scheduler = alpha_ratio
    elif schedule == "constant":
        scheduler = torch.ones_like(alpha_prod_ts)

    # Same normalization as original TFG!
    normalizer = len(scheduler) / scheduler.sum().clamp_min(1e-8)
    return config.rho * scheduler[t_idx] * normalizer
```

### 4.3 Test Verification

```python
def test_schedule_normalization_average():
    """Verify that average of 101 samples equals base_value"""
    timesteps = [i / 100 for i in range(101)]

    # Increase: average should be 2.0
    values = [get_schedule_value(2.0, "increase", t) for t in timesteps]
    avg = sum(values) / len(values)
    assert avg ≈ 2.0  # ✓ Passes

    # Decrease: average should be 2.0
    values = [get_schedule_value(2.0, "decrease", t) for t in timesteps]
    avg = sum(values) / len(values)
    assert avg ≈ 2.0  # ✓ Passes
```

---

## 5. Correctness Verification: Original vs Our Implementation

### 5.1 Mathematical Equivalence

**Original TFG (discrete)**:
```
value = base × s[t] × N / Σs[i]
```
Average:
```
avg = (1/N) × Σ(base × s[t] × N / Σs[i])
    = base × (N/N) × (Σs[t] / Σs[i])
    = base × 1
    = base ✓
```

**Our Implementation (continuous)**:
```
value = base × s(t) × (1 / ∫s(t)dt)
```
Average:
```
avg = ∫₀¹ (base × s(t) × 1/∫s(t)dt) dt
    = base × (1/∫s(t)dt) × ∫s(t)dt
    = base × 1
    = base ✓
```

**Conclusion**: Mathematically equivalent.

### 5.2 Semantic Equivalence

| Original TFG | Our Flow Matching Version | Match |
|--------------|---------------------------|-------|
| "increase" = strong near clean | s(t) high at t→1 (clean) | ✓ |
| "decrease" = strong near noisy | s(t) high at t→0 (noisy) | ✓ |
| average = base_value | average = base_value | ✓ |

### 5.3 Differences (Intentional)

| Item | Original | Our Implementation | Reason |
|------|----------|-------------------|--------|
| Schedule function | α_t (non-linear) | t (linear) | Matches flow matching's linear interpolation |
| Time convention | t=0 clean | t=0 noise | Flow matching standard |

These differences are **intentional** and tailored to each framework's characteristics.

---

## 6. Potential Mistake Checklist

### 6.1 Checklist

| Check Item | Status | Description |
|------------|--------|-------------|
| Is normalization formula correct? | ✓ | `len/sum` → `1/∫` conversion accurate |
| Are integral calculations correct? | ✓ | ∫t dt = 0.5, ∫(1-t) dt = 0.5 |
| Is time convention mapping correct? | ✓ | "increase" is strong in clean direction |
| Is DiT identical to original? | ✓ | Uses same `len/sum` formula |
| Do tests verify this? | ✓ | `test_schedule_normalization_average` |

### 6.2 Issues Found: None

Our implementation accurately preserves the original TFG's intent.

---

## 7. Summary

### 7.1 Meaning of `scheduler.sum()`
- **Total sum** of schedule values across all timesteps
- Used as denominator for normalization
- Purpose: Ensure average guidance equals `base_value` regardless of schedule

### 7.2 Why Does the Integral Become "×2"?
```
Discrete:   normalizer = N / Σs[i] = N / (N × average) = 1 / average
Continuous: normalizer = 1 / ∫s(t)dt = 1 / 0.5 = 2
```
- Integral value is 0.5, so multiply by its reciprocal (2)
- This is the continuous version of `N / Σs[i]`

### 7.3 Implementation Correctness
- **DiT**: Uses **exactly the same** formula as original TFG
- **JiT/SiT**: **Mathematically equivalent** transformation for continuous time
- **Tests**: Empirically verify that average equals base_value

### 7.4 Conclusion
Our Schedule Normalization implementation **accurately** transforms the original TFG's intent to the flow matching framework. No mistakes were found.

---

## Appendix A: Detailed Mathematical Proofs

### A.1 Discrete → Continuous Limit

Limit from N equally-spaced timesteps to continuous time:

```
lim_{N→∞} (N / Σ_{i=0}^{N-1} s(i/N))

= lim_{N→∞} (N / (N × (1/N) × Σ_{i=0}^{N-1} s(i/N)))

= lim_{N→∞} (1 / ((1/N) × Σ_{i=0}^{N-1} s(i/N)))

As Riemann sum:
= 1 / ∫₀¹ s(t) dt
```

### A.2 Increase Schedule Verification

```
s(t) = t
∫₀¹ t dt = [t²/2]₀¹ = 0.5
normalizer = 1/0.5 = 2

Normalized schedule: s̃(t) = 2t
∫₀¹ 2t dt = [t²]₀¹ = 1 ✓

Average: ∫₀¹ base × 2t dt = base × 1 = base ✓
```

### A.3 Decrease Schedule Verification

```
s(t) = 1-t
∫₀¹ (1-t) dt = [t - t²/2]₀¹ = 0.5
normalizer = 1/0.5 = 2

Normalized schedule: s̃(t) = 2(1-t)
∫₀¹ 2(1-t) dt = [2t - t²]₀¹ = 1 ✓

Average: ∫₀¹ base × 2(1-t) dt = base × 1 = base ✓
```

---

## Appendix B: Complete DDPM → Flow Matching Conversion Analysis

Beyond Schedule Normalization (rho/mu), other time-dependent operations in TFG need to be converted from DDPM to Flow Matching. This appendix analyzes all conversion points.

### B.1 List of Time-Dependent Operations in TFG

| Operation | Original TFG (DDPM) | Our Implementation (Flow Matching) | Status |
|-----------|--------------------|------------------------------------|--------|
| rho schedule | `s[t] × N / Σs[i]` | `s(t) × 2` | ✓ Converted |
| mu schedule | `s[t] × N / Σs[i]` | `s(t) × 2` | ✓ Converted |
| sigma schedule | `(1-ᾱ_t)^0.5` (no normalization) | `(1-t)` | ⚠️ Different (intentional) |
| Guidance scaling | `Δ_t/√α_t + Δ_0·√ᾱ_{t-1}` | `Δ_t·(t_next/t) + Δ_0·t_next` | ⚠️ Heuristic |
| Recurrence | `_predict_xt()` (DDPM posterior) | noise injection | ⚠️ Different formula |

### B.2 Sigma Schedule Differences

**Original TFG (edm/methods/tfg.py:68-74)**:
```python
def get_std(self, t, alpha_prod_ts, alpha_prod_t_prevs):
    if self.args.sigma_schedule == 'decrease':
        scheduler = (1 - alpha_prod_ts) ** 0.5  # DDPM noise std
    elif self.args.sigma_schedule == 'constant':
        scheduler = torch.ones_like(alpha_prod_ts)

    return self.args.sigma * scheduler[t]  # No normalization!
```

**Our Implementation (unified_sampler.py)**:
- Flow Matching: `sigma * (1 - t)` (linear)
- DiT: `sigma * (1 - alpha_prod_ts)^0.5` (same as original)

**Key Differences**:
| Item | DDPM | Flow Matching |
|------|------|---------------|
| Decrease schedule | `√(1-ᾱ_t)` (non-linear, actual noise std) | `(1-t)` (linear) |
| Normalization | None | None (for consistency) |

**Analysis**:
- DDPM's `√(1-ᾱ_t)` represents actual noise level
- Flow Matching's `(1-t)` linearly approximates noise ratio
- They differ mathematically, but **relative magnitude** matters more for Monte Carlo smoothing
- This difference is **intentional**, matching flow matching's linear characteristics

### B.3 Guidance Scaling Formula (Most Important Difference)

**Original TFG (edm/methods/tfg.py:129)**:
```python
x_prev += Delta_t / alpha_t ** 0.5 + Delta_0 * alpha_prod_t_prev ** 0.5
```
where `alpha_t = alpha_prod_t / alpha_prod_t_prev`

**Our Implementation (unified_sampler.py:441, 491)**:
```python
z_next = z_next + (t_next / t_clamped) * delta_t + t_next * delta_0
```

**Mathematical Correspondence**:
| DDPM | Flow Matching | Meaning |
|------|---------------|---------|
| `1 / √α_t` | `t_next / t` | Variance guidance scaling |
| `√ᾱ_{t-1}` | `t_next` | Mean guidance scaling |

**Why This Transformation?**

1. **Variance guidance** (`delta_t` scaling):
   - DDPM: `Δ_t / √α_t` ≈ divide by noise step size
   - Flow: `Δ_t × (t_next/t)` ≈ multiply by time ratio
   - Both adjust "guidance strength relative to current noise level"

2. **Mean guidance** (`delta_0` scaling):
   - DDPM: `Δ_0 × √ᾱ_{t-1}` ≈ signal level at next timestep
   - Flow: `Δ_0 × t_next` ≈ signal ratio at next timestep
   - Both reflect "how close to clean data"

**Note**: This transformation is **heuristic**.
From `paper/core_references/mds/theoretical_verification_report.md`:
> "Gap identified: The paper does not derive these scaling factors from first principles for flow matching. The adaptation is heuristic."

### B.4 Recurrence Mechanism

**Original TFG (edm/methods/base.py:112-128)**:
```python
def _predict_xt(self, x_prev, alpha_prod_t, alpha_prod_t_prev):
    xt_mean = (alpha_prod_t / alpha_prod_t_prev) ** 0.5 * x_prev
    return xt_mean + (1 - alpha_prod_t / alpha_prod_t_prev) ** 0.5 * noise
```
This reverses x_{t-1} → x_t from DDPM posterior.

**Our Implementation (unified_sampler.py:586-588)**:
```python
noise_scale = math.sqrt(max(0, (1 - t) ** 2 - (1 - (t + 0.01)) ** 2))
if noise_scale > 0:
    z_current = z_current + noise_scale * torch.randn_like(z_current)
```

**Differences**:
| Item | DDPM | Flow Matching |
|------|------|---------------|
| Noise scale | `√(1 - α_t/α_{t-1})` | `√((1-t)² - (1-t')²)` |
| Meaning | Posterior variance | Continuous time difference |

### B.5 DiT Implementation is Identical to Original

DiT is DDPM-based, so it uses **exactly the same** formulas as original TFG:

```python
# unified_sampler.py:771
x_prev = x_prev + delta_t / alpha_t**0.5 + delta_0 * alpha_prod_t_prev**0.5

# unified_sampler.py:934-946 (sigma)
scheduler = (1 - alpha_prod_ts) ** 0.5  # Same as original

# unified_sampler.py:889-927 (rho/mu)
normalizer = len(scheduler) / scheduler.sum()  # Same as original
```

### B.6 Missing Conversion Checklist

| Item | Needs Conversion? | Current Status | Risk Level |
|------|------------------|----------------|------------|
| rho scheduling | ✓ | ✓ Complete | Low |
| mu scheduling | ✓ | ✓ Complete | Low |
| sigma scheduling | ✓ | ⚠️ Simplified | Medium |
| Guidance scaling | ✓ | ⚠️ Heuristic | **High** |
| Recurrence | ✓ | ⚠️ Different formula | Medium |
| x₀ clipping | ✓ | ✓ Complete | Low |
| Gradient rescaling | ✗ (framework-agnostic) | ✓ Same | Low |

### B.7 Conclusions and Recommendations

1. **rho/mu Schedule Normalization**: ✅ **Correctly converted**
   - `N/Σs[i]` → `1/∫s(t)dt = 2` is mathematically correct

2. **sigma Schedule**: ⚠️ **Intentionally simplified**
   - DDPM's non-linear noise std → Flow's linear (1-t)
   - Experimental verification recommended

3. **Guidance Scaling**: ⚠️ **Heuristic adaptation**
   - Empirical conversion, not mathematical equivalence
   - Needs FID/IS comparison for validation

4. **Recurrence**: ⚠️ **Uses different formula**
   - DDPM posterior vs continuous time noise injection
   - Functionally same purpose (revert one step)

**Final Assessment**: Schedule Normalization (rho/mu) has been **perfectly converted**.
Other time-dependent operations have been **heuristically adapted** due to fundamental mathematical differences between flow matching and DDPM. Experimental verification is needed to confirm these adaptations work in practice.

---

## Appendix C: "On the Guidance of Flow Matching" Paper Analysis

This appendix analyzes the "On the Guidance of Flow Matching" paper (ICML 2025) and its official implementation to evaluate the validity of our implementation direction.

### C.1 Core Theory from the Paper

**Affine Gaussian Path Definition** (Eq. 3):
```
x_t = α_t · x_1 + β_t · x_0
```
where α_t=t, β_t=1-t (uncoupled affine path)

**x̂_1 Estimation** (1-step Euler):
```
x̂_1 = x_t + (1-t) · v_t
```

**Covariance-Preconditioned Guidance** (Theorem 3.1, Eq. 6):
```
g^cov_t = -(α̇β - β̇α)/β · Σ_{1|t} · ∇J(x̂_1)
```

**Jacobian Trick (g^cov-G)** (Proposition 3.4):
```
g^cov-G_t = λ_t · ∇_{x_t} J(x̂_1)
where λ_t = -β(α̇β - β̇α)/α
```

For affine path (α=t, β=1-t):
```
α̇ = 1, β̇ = -1
α̇β - β̇α = (1)(1-t) - (-1)(t) = 1-t+t = 1

λ_t = -(1-t)(1) / t = -(1-t)/t
```

**Key Result**: The theoretical scaling factor for covariance-preconditioned gradient guidance in flow matching is **`(1-t)/t`** (ignoring sign).

### C.2 Official Implementation Analysis (original_implementations/flow_guidance)

**Schedule Definitions** (`image/gflow_img/utils/grad_fn.py`):

```python
def get_scheduler(name, eps=1e-1):
    if name == "const":
        return lambda t: 1 + 0 * t
    elif name == "linear_decay":
        return lambda t: 1 - t
    elif name == "linear_ramp":
        return lambda t: t
    elif name == "as_score":
        return lambda t: 1 / (t + eps) - 1  # ≈ (1-t)/t for small eps
    elif name == "pigdm":
        return lambda t: (1 - t) ** 2 / ((1 - t) ** 2 + t ** 2)
    elif name == "pigdm_gamma":
        return lambda t: torch.sqrt(t / (t ** 2 + (1 - t) ** 2))
    ...
```

**Notable Points**:
- `as_score` schedule: `1/(t+eps) - 1 ≈ (1-t)/t` — This is the **theoretical scaling factor**
- **No normalization**: No ×2 normalization factor is used

**Guidance Application** (`inverse/inverse_problems.py`):

1. **Simple Gradient** (line 176):
```python
return dx_dt + grad  # No time-dependent scaling!
```

2. **PiGDM (Covariance-Preconditioned)** (lines 288-292):
```python
ratio = (1 - t) / (t + eps)  # Uses theoretical λ_t!
v_adapted = vt + ratio * gamma * g * scale
```

### C.3 Comparison with Our Implementation

| Item | Flow Guidance Paper | Flow Guidance Implementation | Our Implementation | Notes |
|------|--------------------|-----------------------------|-------------------|-------|
| **Delta_t scaling** | `(1-t)/t` (theory) | `(1-t)/t` (PiGDM) or none (simple) | `t_next/t` | **Different** |
| **Delta_0 scaling** | implicit in ∇J(x̂_1) | — | `t_next` | **Added** |
| **Schedule normalization** | None | None | ×2 (increase/decrease) | **Only ours uses this** |
| **Schedule types** | linear_decay, as_score, pigdm, etc. | Same | increase, decrease, constant | Similar |

### C.4 Key Difference Analysis

#### C.4.1 Guidance Scaling: `(1-t)/t` vs `t_next/t`

**Theoretical (g^cov-G)**:
```
gradient_scaling = (1-t)/t
```
- t→0 (noisy): very large (stronger guidance when more noisy)
- t→1 (clean): approaches 0

**Our Implementation**:
```
gradient_scaling = t_next/t
```
- t_next ≈ t + dt, so approximately 1 + dt/t
- t→0: large (stronger guidance when more noisy) — similar trend
- t→1: approximately 1 — difference exists

**Analysis**:
- Both have stronger guidance at t→0
- Exact values differ, but **qualitative trend is similar**
- Our implementation is a heuristic mapping from DDPM's `1/√α_t`

#### C.4.2 Mean Guidance (Delta_0)

**Flow Guidance Paper**: No separate scaling factor defined for mean guidance
- Simply uses ∇J(x̂_1) directly or iterative update

**Our Implementation**: Applies `t_next` scaling to Delta_0
- Derived from DDPM's `√ᾱ_{t-1}` mapping
- Additional heuristic

#### C.4.3 Schedule Normalization (×2 factor)

**Flow Guidance Implementation**: No normalization
```python
return scale * schedule(t) * grad_fn(t, x, dx_dt, model)
```

**Our Implementation**: Applies ×2 to increase/decrease
```python
normalizer = 2.0 if normalize else 1.0
return base_value * t * normalizer
```

**Analysis**:
- TFG's normalization ensures "average guidance strength = base_value"
- Flow Guidance doesn't use this normalization
- **Different design philosophies**: TFG prioritizes hyperparameter interpretability, Flow Guidance prioritizes theoretical scaling

### C.5 Implementation Validity Assessment

#### C.5.1 Valid Points

1. **Schedule Normalization (×2)**: ✅
   - Explicit requirement from TFG paper (Σs(t) = T)
   - Ensures average guidance strength consistency
   - Different from Flow Guidance, but correct within TFG framework

2. **Time Direction Consistency**: ✅
   - "increase" is strong at clean (t→1) — same meaning in both implementations
   - Flow Guidance's `linear_ramp` = our `increase` (s(t) = t)

3. **Numerical Stability**: ✅
   - t_eps clamp prevents division by zero
   - Flow Guidance uses same approach (`eps=1e-1`)

#### C.5.2 Areas for Improvement

1. **Variance Guidance Scaling**: ⚠️
   - Theoretically, `(1-t)/t` is recommended
   - Current `t_next/t` is heuristic mapping
   - **Recommendation**: Experimentally compare both scalings

2. **Mean Guidance Scaling**: ⚠️
   - `t_next` scaling lacks theoretical justification
   - Flow Guidance uses direct update without separate scaling
   - **Recommendation**: Compare with direct update approach

### C.6 Recommended Experiment Plan

| Experiment | Description | Purpose |
|------------|-------------|---------|
| A. Scaling comparison | `t_next/t` vs `(1-t)/t` | Verify real-world performance of theoretical scaling |
| B. Mean guidance | `t_next * delta_0` vs direct `delta_0` | Validate necessity of Delta_0 scaling |
| C. Schedule normalization | normalize=True vs False | Measure impact of ×2 factor |

### C.7 Conclusions

**Our implementation correctly transforms TFG framework's intent to flow matching.** However:

1. **Guidance Scaling**: Our `t_next/t` differs from theoretical `(1-t)/t`, but shows qualitatively similar trend (increases at t→0)
2. **Schedule Normalization**: Different from Flow Guidance, but maintained for TFG's hyperparameter interpretability
3. **Experimental Verification Needed**: Compare performance with theoretical scaling to find optimal settings

**Validity Assessment**: Our implementation direction is **valid**.
- Preserves TFG paper's core ideas (variance/mean guidance, scheduling, smoothing)
- Uses reasonable heuristics for DDPM → Flow Matching conversion
- May differ from theoretical optimum, but expected to work practically

---

## Appendix D: Theoretical Derivation Comparison of Guidance Scaling

This appendix provides detailed comparison of theoretical derivations for guidance scaling in DDPM and Flow Matching.

### D.1 Guidance Scaling in DDPM

**DDPM Forward Process**:
```
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
```

**DDPM Reverse Step (DDIM)**:
```
x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}) · ε̂
```

**TFG Guidance Application** (edm/methods/tfg.py:129):
```
x_prev += Δ_t / √α_t + Δ_0 · √ᾱ_{t-1}
```

Where:
- `α_t = ᾱ_t / ᾱ_{t-1}`: step-wise alpha
- `√ᾱ_{t-1}`: signal coefficient of x̂_0

**Interpretation**:
- `Δ_t / √α_t`: propagate gradient from x_t space to x_{t-1} space
- `Δ_0 · √ᾱ_{t-1}`: scale x̂_0 space delta by signal coefficient

### D.2 Guidance Scaling in Flow Matching

**Flow Matching ODE**:
```
dz_t/dt = v_θ(z_t, t)
z_t = t · x_1 + (1-t) · x_0
```

**x̂_1 Estimation**:
```
x̂_1 = z_t + (1-t) · v_t
```

**Theoretical Scaling from Flow Guidance** (g^cov-G):
```
∇_{z_t} J(x̂_1) · λ_t
where λ_t = (1-t)/t
```

**Our Implementation**:
```
z_next += (t_next/t) · Δ_t + t_next · Δ_0
```

### D.3 Mapping Analysis

| DDPM Item | DDPM Value | Flow Matching Correspondence | Our Implementation | Theoretical Value |
|-----------|------------|-----------------------------|--------------------|-------------------|
| Signal coeff | `√ᾱ_t` | `t` | `t` | `t` |
| Noise coeff | `√(1-ᾱ_t)` | `(1-t)` | `(1-t)` | `(1-t)` |
| Δ_t scaling | `1/√α_t` | ? | `t_next/t` | `(1-t)/t` |
| Δ_0 scaling | `√ᾱ_{t-1}` | `t_next` | `t_next` | — |

**Observations**:
- Signal/Noise coefficient mapping is natural
- Δ_t scaling differs between theory (`(1-t)/t`) and implementation (`t_next/t`)

### D.4 Numerical Comparison (based on t=0.5, dt=0.02)

| Scaling | t=0.1 | t=0.5 | t=0.9 |
|---------|-------|-------|-------|
| Our implementation: `t_next/t` | 1.20 | 1.04 | 1.02 |
| Theoretical: `(1-t)/t` | 9.0 | 1.0 | 0.11 |

**Difference Analysis**:
- t=0.1 (noisy): Theory (9.0) >> Implementation (1.2) — theory is much more aggressive
- t=0.5 (middle): Similar (1.0 vs 1.04)
- t=0.9 (clean): Theory (0.11) << Implementation (1.02) — theory almost none

### D.5 Guidance Behavior Comparison

**Theoretical Scaling `(1-t)/t` Meaning**:
- t→0 (noisy): very strong guidance — easy to redirect in noise
- t→1 (clean): almost no guidance — already near manifold

**Our Scaling `t_next/t` Meaning**:
- Relatively uniform guidance across all timesteps
- Similar scaling pattern to DDPM's `1/√α_t`

### D.6 Conclusions and Recommendations

**Limitations of Current Implementation**:
- Differs from theoretically derived `(1-t)/t` scaling
- Especially large difference in t→0 (high noise) region

**Recommendations**:
1. **Option A**: Keep current implementation
   - Expected behavior similar to TFG's DDPM results
   - Mapping of validated DDPM heuristics

2. **Option B**: Experiment with theoretical scaling
   - Separate experiments with `(1-t)/t` scaling
   - Closer to Flow Guidance paper's theoretical optimum

3. **Option C**: Configurable scaling
   - Add option `guidance_scaling_mode: "heuristic" | "theoretical"`
   - Choose optimal setting experimentally

**Experiment Priority**: B > C > A (theoretical verification is important)

---

## Appendix E: Guidance Space Design

This appendix documents the three guidance spaces implemented in UnifiedSampler and their philosophical differences.

### E.1 Overview

| Space | Heritage | dt Handling | Scaling | Philosophy |
|-------|----------|-------------|---------|------------|
| **x** | TFG/DDPM | None | `t_next/t`, `t_next` | Position correction |
| **v** | Flow Guidance | Implicit | `(1-t)/t` | Velocity modification |
| **v2** | Our work | Implicit | `(1-t)/t` per NFE | Per-NFE velocity modification |

### E.2 x-space: TFG/DDPM Heritage (Position Correction)

**Design Philosophy**: DDPM uses discrete timesteps, where dt doesn't naturally exist. Guidance is applied as direct position correction: "move the sample to position X".

**Formula**:
```
z_next = z + dt * v                                    # ODE step
z_next = z_next + (t_next/t) * Δ_t + t_next * Δ_0     # Position correction (NO dt!)
```

**Rationale**:
- DDPM operates with discrete timesteps {0, 1, ..., T-1}
- TFG paper applies guidance as `x_prev += Δ_t/√α_t + Δ_0·√ᾱ_{t-1}` (no dt)
- Guidance magnitude is **step-size independent**

**When to use**: Backward compatibility with TFG hyperparameters, direct translation from DDPM experiments.

### E.3 v-space: Flow Guidance Heritage (Velocity Modification)

**Design Philosophy**: Flow Matching is a continuous ODE `dz/dt = v(z, t)`. Guidance modifies the velocity field: "change the velocity by V". The position change comes from integrating the guided velocity.

**Formula**:
```
λ_t = (1 - t) / t_clamped                 # Theoretical scaling
v_guided = v + λ_t * Δ_t + Δ_0            # Velocity modification
z_next = z + dt * v_guided                # ODE integration (dt implicit)
```

**Rationale**:
- Flow Matching ODE naturally includes dt in integration
- `λ_t = (1-t)/t` is theoretically derived (covariance-preconditioned gradient)
- Stronger guidance at high noise (t→0), weaker at low noise (t→1)

**When to use**: Theoretically grounded experiments, Flow Guidance paper reproduction.

### E.4 v2-space: Per-NFE Velocity Modification (Our Contribution)

**Design Philosophy**: In higher-order solvers like Heun, multiple velocity evaluations happen per step. Instead of applying guidance once, we guide each velocity individually before the solver averages them.

**Formula**:
```
# Guidance at v1 (predictor step)
λ_t = (1 - t) / t_clamped
v1_guided = v1 + λ_t * Δ_t_1 + Δ_0_1

# Guidance at v2 (corrector step)
λ_{t_next} = (1 - t_next) / t_next_clamped     # NOTE: t_next, not t!
v2_guided = v2 + λ_{t_next} * Δ_t_2 + Δ_0_2

# Heun with guided velocities
z_next = z + dt * 0.5 * (v1_guided + v2_guided)
```

**Key Differences from v-space**:
- **v-space**: Single guidance at t, then apply to averaged velocity
- **v2-space**: Separate guidance at t and t_next, then average guided velocities

**Rationale**:
- Each NFE sees different state (z vs z_euler) and different time (t vs t_next)
- More fine-grained guidance control for higher-order solvers
- Different lambda values at each NFE: `λ_t ≠ λ_{t_next}`

**When to use**: Heun sampling with theoretical scaling, research into higher-order guidance integration.

### E.5 Numerical Comparison

For t=0.5, t_next=0.6, dt=0.1:

| Metric | x-space | v-space | v2-space |
|--------|---------|---------|----------|
| Δ_t scaling | `0.6/0.5 = 1.2` | `0.5/0.5 = 1.0` | `1.0` at v1, `0.67` at v2 |
| dt in guidance | No | Yes (implicit) | Yes (implicit) |
| # guidance computations | 1 | 1 | 2 |
| Heun integration | After | After | Within |

### E.6 Implementation Notes

1. **v2-space requires Heun**: The per-NFE philosophy only makes sense for multi-evaluation solvers.

2. **t_next clamping in v2**: At the corrector step, use `t_next_clamped = max(t_next, t_eps)`, not `t_clamped`. This was a bug fix identified during implementation.

3. **DiT always uses x-space**: DDPM's discrete timestep nature aligns with position correction philosophy.

### E.7 Expected Experimental Differences

| Comparison | Expected Observation | Reason |
|------------|---------------------|--------|
| x vs v | Different magnitudes at t→0 and t→1 | λ_t = (1-t)/t amplifies at high noise |
| v vs v2 | Different for Heun, same for Euler | v2 has separate guidance at each NFE |
| x (50 steps) vs x (100 steps) | Similar per-step guidance | x-space is step-size independent |
| v (50 steps) vs v (100 steps) | Different per-step guidance | v-space guidance scales with dt |

### E.8 Research Positioning

The guidance spaces are presented as an **ablation study** comparing:
1. **x-space**: What TFG originally does (DDPM heritage)
2. **v-space**: What theoretical analysis suggests (Flow Guidance)
3. **v2-space**: Natural extension for higher-order solvers

This is **not** our main contribution—the main contribution is the analysis of which prediction target (x/v/ε) is optimal for TFG. The guidance spaces provide a framework for fair comparison across different guidance application strategies.

### E.9 Step-Count Dependency Analysis

A key practical difference between guidance spaces is how cumulative guidance magnitude varies with `num_steps`.

#### E.9.1 Mathematical Analysis

**x-space** cumulative guidance (over full trajectory t: 0 → 1):

With `num_steps = N`, each step contributes guidance *without* dt:
```
Total_x = Σ_{i=0}^{N-1} [(t_{i+1}/t_i) · δ_t_i + t_{i+1} · δ_0_i]
```

Since each term is independent of `dt = 1/N`, doubling N adds approximately twice as many guidance corrections.

**v-space** cumulative guidance (over full trajectory t: 0 → 1):

Each step contributes guidance *with* dt:
```
Total_v = Σ_{i=0}^{N-1} [dt · (λ_{t_i} · δ_t_i + δ_0_i)]
        = Σ_{i=0}^{N-1} [(1/N) · (λ_{t_i} · δ_t_i + δ_0_i)]
```

As `N → ∞`, this converges to the integral:
```
Total_v → ∫_0^1 (λ_t · δ_t(t) + δ_0(t)) dt
```

This integral is independent of N (Riemann sum convergence).

#### E.9.2 Numerical Example

Assume constant `δ_t = δ_0 = 0.1` for simplicity:

| num_steps | dt | x-space per-step | x-space total | v-space per-step | v-space total |
|-----------|-----|-----------------|---------------|-----------------|---------------|
| 10 | 0.1 | ~0.22 | ~2.2 | ~0.022 | ~0.22 |
| 50 | 0.02 | ~0.22 | ~11.0 | ~0.0044 | ~0.22 |
| 100 | 0.01 | ~0.22 | ~22.0 | ~0.0022 | ~0.22 |

(Values are approximate; actual δ_t depends on the state trajectory.)

**Conclusion**: x-space total guidance grows linearly with N. v-space total guidance converges to a constant.

#### E.9.3 Practical Implications

1. **Hyperparameter portability**: v/v2-space hyperparameters (rho, mu) are more transferable across step counts. x-space requires re-tuning when changing num_steps.
2. **Comparison fairness**: When comparing x-space and v-space, always fix num_steps.
3. **Original TFG reproduction**: Use x-space to match published TFG results (DDPM-based).

See also: `docs/concepts/guidance-spaces.md` for the full design philosophy document.
