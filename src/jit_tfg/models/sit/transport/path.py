"""Flow matching interpolation paths for SiT.

This module provides interpolation paths used in SiT's flow matching framework.
The primary path is the Linear path, which is identical to JiT's flow matching.

Convention:
    - t=0: Pure noise (x_0 = ε)
    - t=1: Clean data (x_1 = x)
    - Forward: x_t = t*x + (1-t)*ε
    - Velocity: v = dx_t/dt = x - ε

This is the SAME convention as JiT, making integration straightforward.
"""

from __future__ import annotations

import torch


def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand time tensor t to broadcastable shape with x.

    Args:
        t: Time tensor of shape (B,).
        x: Data tensor of shape (B, ...).

    Returns:
        Expanded time tensor of shape (B, 1, 1, ...).
    """
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


class LinearPath:
    """Linear coupling plan for flow matching.

    The linear path interpolates between noise and data:
        x_t = α_t * x_1 + σ_t * x_0
    where:
        α_t = t (coefficient for clean data)
        σ_t = 1 - t (coefficient for noise)

    Velocity (the prediction target):
        v = dx_t/dt = d(α_t)/dt * x_1 + d(σ_t)/dt * x_0
        v = x_1 - x_0 = x - ε (data minus noise)

    This gives us the conversion formulas:
        x = x_t + (1-t) * v  (v → x_0 prediction)
        ε = x_t - t * v      (v → noise prediction)
    """

    def __init__(self, sigma: float = 0.0) -> None:
        """Initialize linear path.

        Args:
            sigma: Optional noise scale (unused for linear path).
        """
        self.sigma = sigma

    def compute_alpha_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the data coefficient and its derivative.

        Args:
            t: Time tensor.

        Returns:
            Tuple of (α_t, d_α_t/dt):
                α_t = t
                d_α_t/dt = 1
        """
        return t, torch.ones_like(t)

    def compute_sigma_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the noise coefficient and its derivative.

        Args:
            t: Time tensor.

        Returns:
            Tuple of (σ_t, d_σ_t/dt):
                σ_t = 1 - t
                d_σ_t/dt = -1
        """
        return 1 - t, -torch.ones_like(t)

    def compute_xt(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Sample x_t from interpolation path.

        Args:
            t: Time tensor of shape (B,).
            x0: Noise samples of shape (B, C, H, W).
            x1: Clean data of shape (B, C, H, W).

        Returns:
            Interpolated samples x_t of shape (B, C, H, W).
        """
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_ut(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the target velocity at x_t.

        The velocity is the derivative of the interpolation path:
            v = dx_t/dt = d_α_t * x_1 + d_σ_t * x_0 = x_1 - x_0

        Args:
            t: Time tensor of shape (B,).
            x0: Noise samples of shape (B, C, H, W).
            x1: Clean data of shape (B, C, H, W).
            xt: Interpolated samples (unused for linear path).

        Returns:
            Velocity of shape (B, C, H, W).
        """
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def get_x0_from_velocity(self, velocity: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert velocity prediction to clean data prediction (x_0).

        Given v = x - ε and x_t = t*x + (1-t)*ε:
            x_t = t*x + (1-t)*(x - v) = t*x + (1-t)*x - (1-t)*v
            x_t = x - (1-t)*v
            x = x_t + (1-t)*v

        Args:
            velocity: Velocity prediction of shape (B, C, H, W).
            xt: Current noisy sample of shape (B, C, H, W).
            t: Time tensor of shape (B,) or (B, 1, 1, 1).

        Returns:
            Clean data prediction of shape (B, C, H, W).
        """
        if t.ndim == 1:
            t = expand_t_like_x(t, xt)
        return xt + (1 - t) * velocity

    def get_noise_from_velocity(self, velocity: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert velocity prediction to noise prediction (ε).

        Given v = x - ε:
            ε = x - v
        And x = x_t + (1-t)*v:
            ε = x_t + (1-t)*v - v = x_t - t*v

        Args:
            velocity: Velocity prediction of shape (B, C, H, W).
            xt: Current noisy sample of shape (B, C, H, W).
            t: Time tensor of shape (B,) or (B, 1, 1, 1).

        Returns:
            Noise prediction of shape (B, C, H, W).
        """
        if t.ndim == 1:
            t = expand_t_like_x(t, xt)
        return xt - t * velocity

    def get_velocity_from_x0_and_noise(self, x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Compute velocity from x_0 and noise.

        Args:
            x0: Clean data of shape (B, C, H, W).
            noise: Noise of shape (B, C, H, W).

        Returns:
            Velocity v = x_0 - noise.
        """
        return x0 - noise

    def plan(
        self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute interpolation and velocity for training.

        Args:
            t: Time tensor of shape (B,).
            x0: Noise samples of shape (B, C, H, W).
            x1: Clean data of shape (B, C, H, W).

        Returns:
            Tuple of (t, x_t, u_t):
                t: Time tensor
                x_t: Interpolated sample
                u_t: Target velocity
        """
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut
