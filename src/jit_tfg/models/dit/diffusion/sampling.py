"""DDPM and DDIM sampling utilities for DiT.

This module provides sampling functions for the DiT diffusion model,
including standard DDPM sampling and accelerated DDIM sampling.

References:
    - "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    - "Denoising Diffusion Implicit Models" (Song et al., 2021)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from tqdm import tqdm

from jit_tfg.models.dit.diffusion.schedules import DDPMSchedule


def get_ddim_timestep_sequence(
    num_timesteps: int,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a DDIM timestep sequence using uniform integer stride.

    Matches original DiT's SpacedDiffusion with "ddimN" respacing:
    - range(0, 1000, step) = [0, step, 2*step, ..., ] (ascending)
    - Reversed for sampling: descending (noisy to clean)

    Args:
        num_timesteps: Total DDPM timesteps (e.g., 1000).
        num_steps: Number of sampling steps.
        device: Target device.

    Returns:
        Timestep sequence of shape (num_steps,) in descending order.
    """
    step_size = num_timesteps // num_steps
    ts_ascending = torch.arange(0, num_timesteps, step_size, device=device)[:num_steps]
    return ts_ascending.flip(0)


def get_ddpm_timestep_sequence(
    num_timesteps: int,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a DDPM timestep sequence using fractional stride.

    Matches original DiT's space_timesteps(num_timesteps, "N") from respace.py.
    Unlike DDIM's uniform integer stride, DDPM uses fractional striding that
    always includes both endpoints (0 and num_timesteps-1=999).

    Args:
        num_timesteps: Total DDPM timesteps (e.g., 1000).
        num_steps: Number of sampling steps.
        device: Target device.

    Returns:
        Timestep sequence of shape (num_steps,) in descending order.
    """
    if num_steps == 1:
        return torch.tensor([0], device=device, dtype=torch.long)
    frac_stride = (num_timesteps - 1) / (num_steps - 1)
    ts = [round(i * frac_stride) for i in range(num_steps)]
    return torch.tensor(sorted(ts, reverse=True), device=device, dtype=torch.long)


@torch.no_grad()
def ddpm_sample(
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    schedule: DDPMSchedule,
    shape: tuple[int, ...],
    device: torch.device,
    num_steps: int | None = None,
    clip_denoised: bool = False,
    show_progress: bool = True,
) -> torch.Tensor:
    """Standard DDPM sampling.

    Samples from the learned distribution by iteratively denoising
    from pure noise using the reverse diffusion process.

    When num_steps < num_timesteps, posterior coefficients are computed
    on-the-fly from rebased alpha values (matching SpacedDiffusion behavior).

    Args:
        model_fn: Function that takes (x_t, t) and returns epsilon prediction.
            t should be in discrete DDPM format [0, T-1].
        schedule: DDPM schedule with precomputed coefficients.
        shape: Output shape (B, C, H, W).
        device: Target device.
        num_steps: Number of sampling steps (default: all timesteps).
        clip_denoised: Whether to clip x_0 predictions to [-1, 1].
        show_progress: Whether to show progress bar.

    Returns:
        Generated samples of shape (B, C, H, W).
    """
    schedule = schedule.to(device)
    num_steps = num_steps or schedule.num_timesteps

    # Get timestep sequence (fractional stride for DDPM, descending)
    timesteps = get_ddpm_timestep_sequence(schedule.num_timesteps, num_steps, device)

    # Precompute alpha_bar at each selected timestep
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    # Start from pure noise
    x = torch.randn(shape, device=device)

    iterator = tqdm(timesteps, desc="DDPM Sampling") if show_progress else timesteps

    for i, t in enumerate(iterator):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict epsilon
        eps_pred = model_fn(x, t_batch)

        # Predict x_0
        x_0_pred = schedule.predict_x0_from_eps(x, t_batch, eps_pred)
        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)

        # Compute rebased posterior on-the-fly (matches SpacedDiffusion).
        # alpha_prod_t_prev = alpha_bar at the NEXT selected timestep (one step
        # closer to clean), or 1.0 at the final step.
        alpha_prod_t = alphas_cumprod[t]
        if i + 1 < len(timesteps):
            alpha_prod_t_prev = alphas_cumprod[timesteps[i + 1]]
        else:
            alpha_prod_t_prev = torch.ones(1, device=device, dtype=alphas_cumprod.dtype)

        alpha_t = alpha_prod_t / alpha_prod_t_prev.clamp_min(1e-8)
        beta_t = 1 - alpha_t

        # Posterior mean: coef1 * x_0 + coef2 * x_t
        coef1 = beta_t * alpha_prod_t_prev**0.5 / (1 - alpha_prod_t).clamp_min(1e-8)
        coef2 = (1 - alpha_prod_t_prev) * alpha_t**0.5 / (1 - alpha_prod_t).clamp_min(1e-8)
        mean = coef1 * x_0_pred + coef2 * x

        # Posterior log-variance (FIXED_SMALL)
        posterior_variance = beta_t * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t).clamp_min(1e-8)
        log_variance = torch.log(posterior_variance.clamp_min(1e-20))

        # Sample x_{t-1}
        noise = torch.randn_like(x) if t > 0 else 0
        x = mean + (0.5 * log_variance).exp() * noise

    return x


@torch.no_grad()
def ddim_sample(
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    schedule: DDPMSchedule,
    shape: tuple[int, ...],
    device: torch.device,
    num_steps: int = 50,
    eta: float = 0.0,
    clip_denoised: bool = False,
    show_progress: bool = True,
) -> torch.Tensor:
    """DDIM sampling for accelerated generation.

    DDIM allows for deterministic sampling (eta=0) and can use
    far fewer steps than DDPM while maintaining quality.

    Args:
        model_fn: Function that takes (x_t, t) and returns epsilon prediction.
        schedule: DDPM schedule with precomputed coefficients.
        shape: Output shape (B, C, H, W).
        device: Target device.
        num_steps: Number of sampling steps.
        eta: Controls stochasticity (0 = deterministic, 1 = DDPM).
        clip_denoised: Whether to clip x_0 predictions to [-1, 1].
        show_progress: Whether to show progress bar.

    Returns:
        Generated samples of shape (B, C, H, W).
    """
    schedule = schedule.to(device)

    # Get timestep sequence (uniform integer stride for DDIM)
    timesteps = get_ddim_timestep_sequence(schedule.num_timesteps, num_steps, device)

    # Start from pure noise
    x = torch.randn(shape, device=device)

    iterator = tqdm(timesteps, desc="DDIM Sampling") if show_progress else timesteps

    for i, t in enumerate(iterator):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict epsilon
        eps_pred = model_fn(x, t_batch)

        # Predict x_0
        x_0_pred = schedule.predict_x0_from_eps(x, t_batch, eps_pred)
        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)

        # Get alpha values
        alpha_t = schedule.extract(schedule.alphas_cumprod, t_batch, x.shape)

        # Get next timestep
        if i + 1 < len(timesteps):
            t_next = timesteps[i + 1]
            t_next_batch = torch.full((shape[0],), t_next, device=device, dtype=torch.long)
            alpha_t_next = schedule.extract(schedule.alphas_cumprod, t_next_batch, x.shape)
        else:
            alpha_t_next = torch.ones_like(alpha_t)

        # DDIM update
        sigma_t = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)

        # Predicted direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_t_next - sigma_t**2) * eps_pred

        # x_{t-1}
        noise = torch.randn_like(x) if eta > 0 and i + 1 < len(timesteps) else 0
        x = torch.sqrt(alpha_t_next) * x_0_pred + pred_dir + sigma_t * noise

    return x


@torch.no_grad()
def ddim_sample_with_intermediate(
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    schedule: DDPMSchedule,
    shape: tuple[int, ...],
    device: torch.device,
    num_steps: int = 50,
    eta: float = 0.0,
    clip_denoised: bool = False,
    save_intermediates: int = 0,
    show_progress: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """DDIM sampling with optional intermediate outputs.

    Useful for visualization and analysis of the denoising process.

    Args:
        model_fn: Function that takes (x_t, t) and returns epsilon prediction.
        schedule: DDPM schedule with precomputed coefficients.
        shape: Output shape (B, C, H, W).
        device: Target device.
        num_steps: Number of sampling steps.
        eta: Controls stochasticity.
        clip_denoised: Whether to clip x_0 predictions.
        save_intermediates: Number of intermediate states to save (0 = none).
        show_progress: Whether to show progress bar.

    Returns:
        Tuple of (final_samples, intermediate_samples).
    """
    schedule = schedule.to(device)
    timesteps = get_ddim_timestep_sequence(schedule.num_timesteps, num_steps, device)

    x = torch.randn(shape, device=device)
    intermediates = []

    # Determine which steps to save
    if save_intermediates > 0:
        save_indices = {int(i) for i in torch.linspace(0, len(timesteps) - 1, save_intermediates)}
    else:
        save_indices = set()

    iterator = tqdm(timesteps, desc="DDIM Sampling") if show_progress else timesteps

    for i, t in enumerate(iterator):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_pred = model_fn(x, t_batch)

        x_0_pred = schedule.predict_x0_from_eps(x, t_batch, eps_pred)
        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)

        # Save intermediate if requested
        if i in save_indices:
            intermediates.append(x_0_pred.clone())

        alpha_t = schedule.extract(schedule.alphas_cumprod, t_batch, x.shape)

        if i + 1 < len(timesteps):
            t_next = timesteps[i + 1]
            t_next_batch = torch.full((shape[0],), t_next, device=device, dtype=torch.long)
            alpha_t_next = schedule.extract(schedule.alphas_cumprod, t_next_batch, x.shape)
        else:
            alpha_t_next = torch.ones_like(alpha_t)

        sigma_t = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)

        pred_dir = torch.sqrt(1 - alpha_t_next - sigma_t**2) * eps_pred
        noise = torch.randn_like(x) if eta > 0 and i + 1 < len(timesteps) else 0
        x = torch.sqrt(alpha_t_next) * x_0_pred + pred_dir + sigma_t * noise

    return x, intermediates
