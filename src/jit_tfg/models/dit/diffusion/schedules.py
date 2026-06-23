"""DDPM noise schedule utilities.

This module provides beta schedules and precomputed DDPM coefficients
for the DiT diffusion process.

DDPM Forward Process:
    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    where:
    - beta_t: Noise schedule at timestep t
    - alpha_t = 1 - beta_t
    - alpha_bar_t = prod(alpha_i, i=0..t): Cumulative product

References:
    - "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    - "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Linear beta schedule.

    Args:
        num_timesteps: Total number of timesteps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.

    Returns:
        Beta values of shape (num_timesteps,).
    """
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
) -> torch.Tensor:
    """Cosine beta schedule from "Improved DDPM".

    Args:
        num_timesteps: Total number of timesteps.
        s: Small offset to prevent beta from being too small at t=0.

    Returns:
        Beta values of shape (num_timesteps,).
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999)


@dataclass
class DDPMSchedule:
    """Precomputed DDPM diffusion schedule coefficients.

    Stores all precomputed values needed for DDPM forward and reverse
    diffusion processes.

    DDPM Forward:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    DDPM Reverse (mean):
        mu_t = (1/sqrt(alpha_t)) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_pred)

    Attributes:
        betas: Noise schedule (T,).
        log_betas: Log of betas (T,). Upper bound for LEARNED_RANGE variance.
        alphas: 1 - betas (T,).
        alphas_cumprod: Cumulative product of alphas (T,).
        alphas_cumprod_prev: alphas_cumprod shifted by 1 (T,).
        sqrt_alphas_cumprod: sqrt(alpha_bar_t) (T,).
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar_t) (T,).
        sqrt_recip_alphas_cumprod: 1/sqrt(alpha_bar_t) (T,).
        sqrt_recipm1_alphas_cumprod: sqrt(1/alpha_bar_t - 1) (T,).
        posterior_variance: Variance of q(x_{t-1} | x_t, x_0) (T,).
        posterior_log_variance_clipped: Log of posterior variance, clipped (T,).
        posterior_mean_coef1: Coefficient for x_0 in posterior mean (T,).
        posterior_mean_coef2: Coefficient for x_t in posterior mean (T,).
        num_timesteps: Total number of timesteps.
    """

    betas: torch.Tensor
    log_betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor
    num_timesteps: int

    @classmethod
    def from_beta_schedule(
        cls,
        schedule_type: Literal["linear", "cosine"] = "linear",
        num_timesteps: int = 1000,
        **kwargs,
    ) -> DDPMSchedule:
        """Create schedule from a named beta schedule.

        Args:
            schedule_type: Type of beta schedule ("linear" or "cosine").
            num_timesteps: Total number of timesteps.
            **kwargs: Additional arguments for the schedule function.

        Returns:
            DDPMSchedule instance with precomputed coefficients.
        """
        if schedule_type == "linear":
            betas = linear_beta_schedule(num_timesteps, **kwargs)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(num_timesteps, **kwargs)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        return cls.from_betas(betas)

    @classmethod
    def from_betas(cls, betas: torch.Tensor) -> DDPMSchedule:
        """Create schedule from beta values.

        Args:
            betas: Beta values of shape (T,).

        Returns:
            DDPMSchedule instance with precomputed coefficients.
        """
        betas = betas.to(torch.float64)
        num_timesteps = len(betas)

        log_betas = torch.log(betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]])

        # Sqrt coefficients for forward diffusion
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        # Posterior q(x_{t-1} | x_t, x_0) coefficients
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Clip t=0 using posterior_variance[1] (matches original DiT's
        # np.append(posterior_variance[1], posterior_variance[1:]))
        posterior_log_variance_clipped = torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]]))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        return cls(
            betas=betas,
            log_betas=log_betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
            posterior_variance=posterior_variance,
            posterior_log_variance_clipped=posterior_log_variance_clipped,
            posterior_mean_coef1=posterior_mean_coef1,
            posterior_mean_coef2=posterior_mean_coef2,
            num_timesteps=num_timesteps,
        )

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> DDPMSchedule:
        """Move all tensors to a device and dtype.

        Args:
            device: Target device.
            dtype: Target dtype.

        Returns:
            New DDPMSchedule with tensors on the target device.
        """
        return DDPMSchedule(
            betas=self.betas.to(device, dtype),
            log_betas=self.log_betas.to(device, dtype),
            alphas=self.alphas.to(device, dtype),
            alphas_cumprod=self.alphas_cumprod.to(device, dtype),
            alphas_cumprod_prev=self.alphas_cumprod_prev.to(device, dtype),
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod.to(device, dtype),
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod.to(device, dtype),
            sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod.to(device, dtype),
            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod.to(device, dtype),
            posterior_variance=self.posterior_variance.to(device, dtype),
            posterior_log_variance_clipped=self.posterior_log_variance_clipped.to(device, dtype),
            posterior_mean_coef1=self.posterior_mean_coef1.to(device, dtype),
            posterior_mean_coef2=self.posterior_mean_coef2.to(device, dtype),
            num_timesteps=self.num_timesteps,
        )

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract values from a at timesteps t, with shape broadcasting.

        Args:
            a: Coefficient tensor of shape (T,).
            t: Timestep indices of shape (B,).
            x_shape: Target shape for broadcasting.

        Returns:
            Extracted values of shape (B, 1, 1, 1) for 4D inputs.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from forward diffusion q(x_t | x_0).

        Args:
            x_0: Clean data of shape (B, C, H, W).
            t: Timesteps of shape (B,).
            noise: Optional noise tensor (if None, sampled from N(0, I)).

        Returns:
            Noisy sample x_t of shape (B, C, H, W).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and epsilon prediction.

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)

        Args:
            x_t: Noisy sample of shape (B, C, H, W).
            t: Timesteps of shape (B,).
            eps: Epsilon prediction of shape (B, C, H, W).

        Returns:
            Predicted x_0 of shape (B, C, H, W).
        """
        sqrt_recip_alpha = self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alpha = self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alpha * x_t - sqrt_recipm1_alpha * eps

    def q_posterior_mean_variance(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior q(x_{t-1} | x_t, x_0) mean and variance.

        Args:
            x_0: Clean data of shape (B, C, H, W).
            x_t: Noisy sample of shape (B, C, H, W).
            t: Timesteps of shape (B,).

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance).
        """
        coef1 = self.extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self.extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = coef1 * x_0 + coef2 * x_t

        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance
