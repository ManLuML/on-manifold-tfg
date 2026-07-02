"""Configuration dataclass for TFG hyperparameters.

This module defines the TFGConfig dataclass that encapsulates all 7 TFG
hyperparameters plus additional configuration for integration with JiT.

TFG Design Space (from the paper):
    The 7 hyperparameters control mean guidance, variance guidance,
    implicit dynamics, and recurrence mechanisms.
"""

from dataclasses import dataclass
from typing import Literal

# Schedule types for time-dependent guidance strengths
ScheduleType = Literal["constant", "increase", "decrease"]


@dataclass
class TFGConfig:
    """Configuration for Training-Free Guidance.

    TFG Hyperparameters (7 total):
        N_recur (recur_steps): Number of recurrence steps per timestep.
            Higher values improve guidance at the cost of NFE.

        N_iter (iter_steps): Number of iterations for mean guidance.
            Controls refinement of the x_0 estimate.

        sigma (gamma_bar): Smoothing parameter for implicit dynamics.
            Controls Gaussian kernel width for gradient smoothing.

        rho (rho_bar): Scalar strength for variance guidance.
            Gradient computed w.r.t. x_t (noisy state).

        rho_schedule (s_rho): Time-dependent schedule for rho.
            Options: "constant", "increase", "decrease"

        mu (mu_bar): Scalar strength for mean guidance.
            Gradient computed w.r.t. x_0|t (clean data estimate).

        mu_schedule (s_mu): Time-dependent schedule for mu.
            Options: "constant", "increase", "decrease"

    Additional Parameters:
        eps_bsz: Batch size for Monte Carlo estimation of smoothed gradients.
        clip_scale: Maximum gradient norm for clipping.
        clip_x0: Whether to clip x_0 estimates to [-1, 1].

    Attributes:
        recur_steps: Number of recurrence steps (N_recur).
        iter_steps: Number of mean guidance iterations (N_iter).
        sigma: Smoothing parameter (gamma_bar).
        rho: Variance guidance strength (rho_bar).
        rho_schedule: Schedule for variance guidance.
        mu: Mean guidance strength (mu_bar).
        mu_schedule: Schedule for mean guidance.
        sigma_schedule: Schedule for smoothing parameter.
        eps_bsz: Monte Carlo batch size.
        clip_scale: Gradient clipping scale.
        clip_x0: Whether to clip x_0 predictions.
        clip_sample_range: Range for clipping x_0.
    """

    # TFG Core Hyperparameters (7 total)
    recur_steps: int = 1
    """Number of recurrence steps (N_recur). Default: 1."""

    iter_steps: int = 1
    """Number of mean guidance iterations (N_iter). Default: 1."""

    sigma: float = 0.01
    """Smoothing parameter (gamma_bar). Default: 0.01."""

    rho: float = 1.0
    """Variance guidance strength (rho_bar). Default: 1.0."""

    rho_schedule: ScheduleType = "increase"
    """Time schedule for variance guidance. Default: 'increase'."""

    mu: float = 1.0
    """Mean guidance strength (mu_bar). Default: 1.0."""

    mu_schedule: ScheduleType = "increase"
    """Time schedule for mean guidance. Default: 'increase'."""

    sigma_schedule: ScheduleType = "decrease"
    """Time schedule for smoothing. Default: 'decrease'."""

    normalize_schedules: bool = False
    """Normalize schedules so average equals base_value. Default: False.

    When True, 'increase' and 'decrease' schedules are normalized by
    multiplying by 2.0, so the average value over t∈[0,1] equals base_value.
    This matches the original TFG paper's normalization formula:
        value = base * scheduler[t] * len(s) / sum(s)

    Example with rho=2.0, rho_schedule="increase":
    - normalize_schedules=False: t=0.5 → rho = 2.0 * 0.5 = 1.0
    - normalize_schedules=True:  t=0.5 → rho = 2.0 * 0.5 * 2.0 = 2.0
    """

    # Monte Carlo & Numerical Stability
    eps_bsz: int = 4
    """Batch size for MC gradient estimation. Default: 4."""

    clip_scale: float = 100.0
    """Maximum gradient norm for clipping. Default: 100.0."""

    clip_x0: bool = True
    """Whether to clip x_0 estimates. Default: True."""

    clip_sample_range: float = 1.0
    """Range for x_0 clipping in pixel space: [-range, range]. Default: 1.0."""

    clip_sample_range_latent: float = 5.0
    """Range for x_0 clipping in VAE latent space: [-range, range]. Default: 5.0.

    DiT and other latent diffusion models operate in VAE latent space where
    values typically range from -5 to 5, not -1 to 1 like pixel space.
    """

    # Guidance Space Options
    rescale_mode: Literal["clip", "original"] = "clip"
    """Gradient rescaling mode for image data. Default: 'clip'.

    - 'clip': Clip image gradients when norm exceeds clip_scale (our default).
    - 'original': No clipping for images, matching original TFG implementation
      (edm/tasks/utils.py). The original rescale_grad computes scale but returns
      gradient unchanged when node_mask is None (image data).
    """

    lambda_mode: Literal["flow_guidance", "identity", "auto"] = "auto"
    """Lambda_t mode for v-space/v2-space guidance scaling. Default: 'auto'.

    Only affects v-space and v2-space guidance (x-space ignores this).

    - 'auto' (default): Automatically select based on prediction target:
      x-prediction -> identity, v/e-prediction -> flow_guidance.
      Recommended because 'flow_guidance' with rho_schedule='increase'
      causes schedule inversion for x-prediction models:
      rho(t)*lambda_t = base_rho*t*(1-t)/t = base_rho*(1-t), which is
      a decrease profile even though 'increase' was intended.
    - 'flow_guidance': lambda_t = (1-t)/t. Derived from "On the Guidance of
      Flow Matching" (Feng et al.) g^{cov-G} framework, assuming v-prediction
      model Jacobian structure. Use for v-prediction models (SiT).
    - 'identity': lambda_t = 1.0. No time-dependent scaling. This gives
      v-space with step-count stability but without the (1-t)/t profile.
      More appropriate for x-prediction models where gradients are
      already bounded (Theorem 3.1).
    """

    # Device Configuration
    device: str = "cuda"
    """Device for computation. Default: 'cuda'."""

    seed: int = 42
    """Random seed for reproducibility. Default: 42."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.rho < 0:
            raise ValueError(f"rho must be >= 0, got {self.rho}")
        if self.mu < 0:
            raise ValueError(f"mu must be >= 0, got {self.mu}")
        if self.recur_steps < 1:
            raise ValueError(f"recur_steps must be >= 1, got {self.recur_steps}")
        if self.iter_steps < 1:
            raise ValueError(f"iter_steps must be >= 1, got {self.iter_steps}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")
        if self.eps_bsz < 1:
            raise ValueError(f"eps_bsz must be >= 1, got {self.eps_bsz}")

    @classmethod
    def default(cls) -> "TFGConfig":
        """Create a default TFG configuration."""
        return cls()

    @classmethod
    def strong_guidance(cls) -> "TFGConfig":
        """Create a configuration for strong guidance (higher mu, rho)."""
        return cls(
            recur_steps=3,
            iter_steps=3,
            rho=2.0,
            mu=2.0,
            sigma=0.02,
        )

    @classmethod
    def light_guidance(cls) -> "TFGConfig":
        """Create a configuration for light guidance (lower mu, rho)."""
        return cls(
            recur_steps=1,
            iter_steps=1,
            rho=0.5,
            mu=0.5,
            sigma=0.005,
        )
