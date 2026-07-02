"""Guidance space calibration utilities.

Computes equivalent rho/mu values when switching between x-space and v-space
guidance. v-space applies significantly less cumulative guidance than x-space
with identical hyperparameters due to dt normalization.

The magnitude gap depends on lambda_mode (N=50, t_eps=0.05):
- flow_guidance: ~16x delta_t ratio (clamped lambda boosts v-space at early steps)
- identity: ~52x delta_t ratio (no lambda amplification)
- delta_0: ~25x ratio for both (lambda not applied to delta_0)

Additionally, flow_guidance lambda with rho_schedule='increase' causes
schedule inversion: rho(t)*lambda_t = base_rho*t*(1-t)/t = base_rho*(1-t),
which is a decrease profile. Identity lambda preserves the intended schedule.
"""

from __future__ import annotations


def compute_guidance_ratio_profile(
    num_steps: int = 50,
    t_eps: float = 0.05,
    lambda_mode: str = "flow_guidance",
) -> dict:
    """Compute per-step and cumulative x/v guidance ratio.

    Args:
        num_steps: Number of ODE steps (Heun steps, not NFE).
        t_eps: Minimum time value for clamping (matches unified_sampler t_eps).
        lambda_mode: Lambda mode ('flow_guidance' or 'identity').

    Returns:
        Dictionary with per-step and cumulative analysis results.
    """
    # Match actual sampler: linspace(0, 1, num_steps+1) with t_eps clamping
    timesteps = [i / num_steps for i in range(num_steps + 1)]

    steps_data = []
    cum_x_delta_t = 0.0
    cum_v_delta_t = 0.0
    cum_x_delta_0 = 0.0
    cum_v_delta_0 = 0.0

    for i in range(num_steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t

        if lambda_mode == "flow_guidance":
            t_clamped = max(t, t_eps)
            lambda_t = (1 - t) / t_clamped
        else:  # identity
            lambda_t = 1.0

        # x-space: z += (t_next/t) * delta_t + t_next * delta_0
        x_delta_t_scale = t_next / max(t, t_eps)
        x_delta_0_scale = t_next

        # v-space: z += dt * (lambda_t * delta_t + delta_0)
        v_delta_t_scale = dt * lambda_t
        v_delta_0_scale = dt

        cum_x_delta_t += x_delta_t_scale
        cum_v_delta_t += v_delta_t_scale
        cum_x_delta_0 += x_delta_0_scale
        cum_v_delta_0 += v_delta_0_scale

        ratio_delta_t = x_delta_t_scale / v_delta_t_scale if v_delta_t_scale > 0 else float("inf")
        ratio_delta_0 = x_delta_0_scale / v_delta_0_scale if v_delta_0_scale > 0 else float("inf")

        steps_data.append({
            "step": i,
            "t": t,
            "t_next": t_next,
            "dt": dt,
            "lambda_t": lambda_t,
            "x_delta_t_scale": x_delta_t_scale,
            "v_delta_t_scale": v_delta_t_scale,
            "ratio_delta_t": ratio_delta_t,
            "x_delta_0_scale": x_delta_0_scale,
            "v_delta_0_scale": v_delta_0_scale,
            "ratio_delta_0": ratio_delta_0,
        })

    cum_ratio_delta_t = cum_x_delta_t / cum_v_delta_t if cum_v_delta_t > 0 else float("inf")
    cum_ratio_delta_0 = cum_x_delta_0 / cum_v_delta_0 if cum_v_delta_0 > 0 else float("inf")

    return {
        "num_steps": num_steps,
        "t_eps": t_eps,
        "lambda_mode": lambda_mode,
        "steps": steps_data,
        "cumulative": {
            "x_delta_t_total": cum_x_delta_t,
            "v_delta_t_total": cum_v_delta_t,
            "ratio_delta_t": cum_ratio_delta_t,
            "x_delta_0_total": cum_x_delta_0,
            "v_delta_0_total": cum_v_delta_0,
            "ratio_delta_0": cum_ratio_delta_0,
        },
    }


def compute_equivalent_rho_mu(
    x_rho: float,
    x_mu: float,
    num_steps: int = 50,
    t_eps: float = 0.05,
    lambda_mode: str = "flow_guidance",
) -> dict:
    """Compute v-space rho/mu equivalent to given x-space values.

    Args:
        x_rho: x-space rho value.
        x_mu: x-space mu value.
        num_steps: Number of ODE steps.
        t_eps: Minimum time value.
        lambda_mode: Lambda mode for v-space.

    Returns:
        Dictionary with equivalent v-space rho and mu values.
    """
    result = compute_guidance_ratio_profile(num_steps, t_eps, lambda_mode)
    cum = result["cumulative"]

    return {
        "x_rho": x_rho,
        "x_mu": x_mu,
        "v_rho_equivalent": x_rho * cum["ratio_delta_t"],
        "v_mu_equivalent": x_mu * cum["ratio_delta_0"],
        "ratio_delta_t": cum["ratio_delta_t"],
        "ratio_delta_0": cum["ratio_delta_0"],
    }
