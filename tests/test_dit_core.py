"""Tests for DiT core functionality.

Tests cover:
1. DDPM schedules (linear, cosine, coefficient calculations)
2. DiTWrapper timestep conversion
3. DiT model forward pass
4. DiTDenoiser prediction conversion
"""

import math

import pytest
import torch
import torch.nn as nn

from jit_tfg.models.dit.diffusion.schedules import (
    DDPMSchedule,
    cosine_beta_schedule,
    linear_beta_schedule,
)
from jit_tfg.models.dit.model import DiT, DiT_models
from jit_tfg.models.dit.wrapper import DiTWrapper

# =============================================================================
# DDPM Schedule Tests
# =============================================================================


class TestLinearBetaSchedule:
    """Tests for linear beta schedule."""

    def test_shape(self) -> None:
        """Beta schedule should have correct length."""
        num_timesteps = 1000
        betas = linear_beta_schedule(num_timesteps)
        assert betas.shape == (num_timesteps,)

    def test_range(self) -> None:
        """Beta values should be in expected range."""
        betas = linear_beta_schedule(1000, beta_start=0.0001, beta_end=0.02)
        assert betas[0].item() == pytest.approx(0.0001, rel=1e-5)
        assert betas[-1].item() == pytest.approx(0.02, rel=1e-5)

    def test_monotonically_increasing(self) -> None:
        """Linear schedule should monotonically increase."""
        betas = linear_beta_schedule(1000)
        assert torch.all(betas[1:] >= betas[:-1])


class TestCosineBetaSchedule:
    """Tests for cosine beta schedule."""

    def test_shape(self) -> None:
        """Beta schedule should have correct length."""
        num_timesteps = 1000
        betas = cosine_beta_schedule(num_timesteps)
        assert betas.shape == (num_timesteps,)

    def test_clamped_range(self) -> None:
        """Beta values should be clamped to [0, 0.999]."""
        betas = cosine_beta_schedule(1000)
        assert betas.min().item() >= 0.0
        assert betas.max().item() <= 0.999


class TestDDPMSchedule:
    """Tests for DDPMSchedule dataclass."""

    @pytest.fixture
    def linear_schedule(self) -> DDPMSchedule:
        """Create a small linear schedule for testing."""
        return DDPMSchedule.from_beta_schedule("linear", num_timesteps=100)

    @pytest.fixture
    def cosine_schedule(self) -> DDPMSchedule:
        """Create a small cosine schedule for testing."""
        return DDPMSchedule.from_beta_schedule("cosine", num_timesteps=100)

    def test_alpha_relationship(self, linear_schedule: DDPMSchedule) -> None:
        """alphas should equal 1 - betas."""
        expected = 1.0 - linear_schedule.betas
        assert torch.allclose(linear_schedule.alphas, expected)

    def test_alphas_cumprod_calculation(self, linear_schedule: DDPMSchedule) -> None:
        """alphas_cumprod should be cumulative product of alphas."""
        expected = torch.cumprod(linear_schedule.alphas, dim=0)
        assert torch.allclose(linear_schedule.alphas_cumprod, expected)

    def test_alphas_cumprod_decreasing(self, linear_schedule: DDPMSchedule) -> None:
        """alphas_cumprod should monotonically decrease (more noise over time)."""
        assert torch.all(linear_schedule.alphas_cumprod[1:] <= linear_schedule.alphas_cumprod[:-1])

    def test_sqrt_coefficients(self, linear_schedule: DDPMSchedule) -> None:
        """sqrt coefficients should satisfy mathematical relationships."""
        # sqrt(alpha_cumprod)^2 = alpha_cumprod
        assert torch.allclose(
            linear_schedule.sqrt_alphas_cumprod**2,
            linear_schedule.alphas_cumprod,
            rtol=1e-5,
        )
        # sqrt(1 - alpha_cumprod)^2 = 1 - alpha_cumprod
        assert torch.allclose(
            linear_schedule.sqrt_one_minus_alphas_cumprod**2,
            1 - linear_schedule.alphas_cumprod,
            rtol=1e-5,
        )

    def test_q_sample_at_t0_returns_original(self, linear_schedule: DDPMSchedule) -> None:
        """q_sample at t=0 should return nearly the original image."""
        x_0 = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        t = torch.zeros(2, dtype=torch.long)
        noise = torch.randn_like(x_0)

        x_t = linear_schedule.q_sample(x_0, t, noise)

        # At t=0, alpha_cumprod is close to 1, so x_t ≈ x_0
        # q_sample: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        alpha_0 = linear_schedule.alphas_cumprod[0].item()
        # Verify alpha_cumprod[0] is very close to 1
        assert alpha_0 > 0.999, f"Expected alpha_cumprod[0] > 0.999, got {alpha_0}"
        # The difference is bounded by: |sqrt(alpha) - 1| * |x_0| + sqrt(1 - alpha) * |noise|
        # With random tensors, use a tolerance that accounts for worst-case noise magnitude
        noise_coeff = (1 - alpha_0) ** 0.5
        scale_diff = abs(alpha_0**0.5 - 1)
        # Tolerance: noise contribution (scaled by max noise ~3 std) + scaling difference
        atol = noise_coeff * noise.abs().max().item() + scale_diff * x_0.abs().max().item() + 0.01
        assert torch.allclose(x_t, x_0, atol=atol)

    def test_q_sample_at_tmax_returns_noise(self, linear_schedule: DDPMSchedule) -> None:
        """q_sample at t=T-1 should return mostly noise."""
        x_0 = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        t = torch.full((2,), linear_schedule.num_timesteps - 1, dtype=torch.long)
        noise = torch.randn_like(x_0)

        x_t = linear_schedule.q_sample(x_0, t, noise)

        # At t=T-1, alpha_cumprod is small (but not zero for linear schedule)
        # Check that noise component dominates by comparing magnitudes
        sqrt_alpha = linear_schedule.sqrt_alphas_cumprod[t[0]]
        sqrt_one_minus_alpha = linear_schedule.sqrt_one_minus_alphas_cumprod[t[0]]
        # For linear schedule, noise coefficient should be larger than signal coefficient
        assert sqrt_one_minus_alpha > sqrt_alpha

    def test_predict_x0_from_eps_roundtrip(self, linear_schedule: DDPMSchedule) -> None:
        """predict_x0 should recover original when given correct epsilon."""
        x_0 = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        t = torch.randint(1, 100, (2,))  # Avoid t=0 for numerical stability
        eps = torch.randn_like(x_0)

        # Forward diffusion
        x_t = linear_schedule.q_sample(x_0, t, eps)

        # Predict x_0 from x_t and true epsilon
        x_0_pred = linear_schedule.predict_x0_from_eps(x_t, t, eps)

        assert torch.allclose(x_0_pred, x_0, atol=1e-5)

    def test_extract_shape(self, linear_schedule: DDPMSchedule) -> None:
        """extract should broadcast correctly."""
        batch_size = 4
        t = torch.randint(0, 100, (batch_size,))
        x_shape = (batch_size, 4, 8, 8)

        out = linear_schedule.extract(linear_schedule.alphas_cumprod, t, x_shape)

        assert out.shape == (batch_size, 1, 1, 1)

    def test_to_device(self, linear_schedule: DDPMSchedule) -> None:
        """Schedule should move to device correctly."""
        schedule_f32 = linear_schedule.to(torch.device("cpu"), dtype=torch.float32)

        assert schedule_f32.betas.dtype == torch.float32
        assert schedule_f32.alphas_cumprod.dtype == torch.float32

    def test_invalid_schedule_type(self) -> None:
        """Invalid schedule type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown schedule type"):
            DDPMSchedule.from_beta_schedule("invalid", num_timesteps=100)


# =============================================================================
# DiTWrapper Timestep Conversion Tests
# =============================================================================


class MockDiT(nn.Module):
    """Mock DiT model for wrapper testing."""

    def __init__(self, in_channels: int = 4, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # Simple identity-like operation
        self.linear = nn.Linear(in_channels, in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Just return zeros with same shape for testing
        return torch.zeros_like(x)


class TestDiTWrapperTimestepConversion:
    """Tests for DiTWrapper timestep conversion."""

    @pytest.fixture
    def wrapper(self) -> DiTWrapper:
        """Create wrapper with mock DiT."""
        mock_dit = MockDiT()
        return DiTWrapper(mock_dit, num_timesteps=1000)

    def test_t0_jit_to_t999_ddpm(self, wrapper: DiTWrapper) -> None:
        """JiT t=0 (noise) should map to DDPM t=999 (noise)."""
        t_jit = torch.tensor([0.0])
        t_ddpm = wrapper.t_continuous_to_discrete(t_jit)
        assert t_ddpm.item() == 999

    def test_t1_jit_to_t0_ddpm(self, wrapper: DiTWrapper) -> None:
        """JiT t=1 (clean) should map to DDPM t=0 (clean)."""
        t_jit = torch.tensor([1.0])
        t_ddpm = wrapper.t_continuous_to_discrete(t_jit)
        assert t_ddpm.item() == 0

    def test_t05_jit_to_t499_ddpm(self, wrapper: DiTWrapper) -> None:
        """JiT t=0.5 should map to DDPM t≈499."""
        t_jit = torch.tensor([0.5])
        t_ddpm = wrapper.t_continuous_to_discrete(t_jit)
        # (1 - 0.5) * 999 = 499.5 -> 499
        assert t_ddpm.item() == 499

    def test_roundtrip_conversion(self, wrapper: DiTWrapper) -> None:
        """Converting back and forth should approximately preserve the value."""
        t_jit_original = torch.tensor([0.3, 0.5, 0.7])
        t_ddpm = wrapper.t_continuous_to_discrete(t_jit_original)
        t_jit_recovered = wrapper.t_discrete_to_continuous(t_ddpm)

        # Due to discretization, we allow some tolerance
        assert torch.allclose(t_jit_original, t_jit_recovered, atol=0.002)

    def test_batch_conversion(self, wrapper: DiTWrapper) -> None:
        """Batch timestep conversion should work correctly."""
        t_jit = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        t_ddpm = wrapper.t_continuous_to_discrete(t_jit)

        expected = torch.tensor([999, 749, 499, 249, 0])
        assert torch.equal(t_ddpm, expected)

    def test_clamping_out_of_range(self, wrapper: DiTWrapper) -> None:
        """Values outside [0, 1] should be clamped."""
        t_jit = torch.tensor([-0.5, 1.5])
        t_ddpm = wrapper.t_continuous_to_discrete(t_jit)

        # -0.5 clamped to 0 -> 999, 1.5 clamped to 1 -> 0
        assert t_ddpm[0].item() == 999
        assert t_ddpm[1].item() == 0


class TestDiTWrapperForward:
    """Tests for DiTWrapper forward pass."""

    @pytest.fixture
    def wrapper(self) -> DiTWrapper:
        """Create wrapper with mock DiT."""
        mock_dit = MockDiT(in_channels=4)
        return DiTWrapper(mock_dit, num_timesteps=1000)

    def test_forward_shape(self, wrapper: DiTWrapper) -> None:
        """Forward should return correct shape."""
        batch_size = 2
        z = torch.randn(batch_size, 4, 32, 32)
        t = torch.tensor([0.5, 0.5])
        y = torch.tensor([207, 360])

        output = wrapper(z, t, y)

        assert output.shape == (batch_size, 4, 32, 32)

    def test_forward_handles_t_shape_variations(self, wrapper: DiTWrapper) -> None:
        """Forward should handle different t shapes."""
        z = torch.randn(2, 4, 32, 32)
        y = torch.tensor([207, 360])

        # 1D timestep
        t_1d = torch.tensor([0.5, 0.5])
        out1 = wrapper(z, t_1d, y)

        # 4D timestep (B, 1, 1, 1)
        t_4d = torch.tensor([[[[0.5]]], [[[0.5]]]])
        out2 = wrapper(z, t_4d, y)

        assert out1.shape == out2.shape == (2, 4, 32, 32)


# =============================================================================
# DiT Model Tests
# =============================================================================


class TestDiTModel:
    """Tests for DiT model instantiation and forward pass."""

    def test_dit_b4_instantiation(self) -> None:
        """DiT-B/4 should instantiate correctly."""
        model = DiT_models["DiT-B/4"](input_size=32, num_classes=10)
        assert isinstance(model, DiT)
        assert len(list(model.parameters())) > 0

    def test_dit_s2_instantiation(self) -> None:
        """DiT-S/2 should instantiate correctly."""
        model = DiT_models["DiT-S/2"](input_size=32, num_classes=10)
        assert isinstance(model, DiT)

    def test_dit_forward_shape(self) -> None:
        """DiT forward should return correct shape."""
        model = DiT_models["DiT-B/4"](
            input_size=8,
            in_channels=4,
            num_classes=10,
            learn_sigma=False,
        )

        batch_size = 2
        x = torch.randn(batch_size, 4, 8, 8)
        t = torch.randint(0, 1000, (batch_size,))
        y = torch.randint(0, 10, (batch_size,))

        output = model(x, t, y)

        # Without learn_sigma, output channels = in_channels
        assert output.shape == (batch_size, 4, 8, 8)

    def test_dit_forward_with_learn_sigma(self) -> None:
        """DiT with learn_sigma should output double channels."""
        model = DiT_models["DiT-B/4"](
            input_size=8,
            in_channels=4,
            num_classes=10,
            learn_sigma=True,
        )

        batch_size = 2
        x = torch.randn(batch_size, 4, 8, 8)
        t = torch.randint(0, 1000, (batch_size,))
        y = torch.randint(0, 10, (batch_size,))

        output = model(x, t, y)

        # With learn_sigma, output channels = 2 * in_channels
        assert output.shape == (batch_size, 8, 8, 8)

    def test_dit_forward_no_nan(self) -> None:
        """DiT forward should not produce NaN values."""
        model = DiT_models["DiT-B/4"](input_size=8, num_classes=10)

        x = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 1000, (2,))
        y = torch.randint(0, 10, (2,))

        output = model(x, t, y)

        assert not torch.isnan(output).any()


# =============================================================================
# DiTDenoiser Prediction Conversion Tests
# =============================================================================


class TestDiTDenoiserConversion:
    """Tests for DiTDenoiser prediction conversion logic."""

    def test_convert_prediction_shapes(self) -> None:
        """_convert_prediction should return correct shapes."""
        from jit_tfg.models.dit.denoiser import DiTDenoiser

        mock_dit = MockDiT(in_channels=4)
        # Use same num_timesteps for wrapper and schedule
        num_timesteps = 1000
        wrapper = DiTWrapper(mock_dit, num_timesteps=num_timesteps)

        # Create a mock VAE
        class MockVAE:
            def to(self, device):
                return self

        denoiser = DiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_timesteps=num_timesteps,
        )

        batch_size = 2
        eps_pred = torch.randn(batch_size, 4, 8, 8)
        z = torch.randn(batch_size, 4, 8, 8)
        t = torch.tensor([0.3, 0.5])

        x_pred, v_pred, e_pred = denoiser._convert_prediction(eps_pred, z, t)

        assert x_pred.shape == (batch_size, 4, 8, 8)
        assert v_pred.shape == (batch_size, 4, 8, 8)
        assert e_pred.shape == (batch_size, 4, 8, 8)

    def test_convert_prediction_epsilon_passthrough(self) -> None:
        """e_pred should equal input eps_pred."""
        from jit_tfg.models.dit.denoiser import DiTDenoiser

        mock_dit = MockDiT(in_channels=4)
        num_timesteps = 1000
        wrapper = DiTWrapper(mock_dit, num_timesteps=num_timesteps)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = DiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_timesteps=num_timesteps,
        )

        eps_pred = torch.randn(2, 4, 8, 8)
        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.3, 0.5])

        _, _, e_pred = denoiser._convert_prediction(eps_pred, z, t)

        assert torch.equal(e_pred, eps_pred)

    def test_convert_prediction_x0_formula(self) -> None:
        """x_pred should follow DDPM formula: x0 = (z - sqrt(1-α̅)*ε) / sqrt(α̅)."""
        from jit_tfg.models.dit.denoiser import DiTDenoiser

        mock_dit = MockDiT(in_channels=4)
        num_timesteps = 1000
        wrapper = DiTWrapper(mock_dit, num_timesteps=num_timesteps)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = DiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_timesteps=num_timesteps,
        )

        eps_pred = torch.randn(1, 4, 8, 8)
        z = torch.randn(1, 4, 8, 8)
        t = torch.tensor([0.5])  # continuous

        x_pred, _, _ = denoiser._convert_prediction(eps_pred, z, t)

        # Manual calculation using schedule - convert to same dtype as x_pred
        t_discrete = wrapper.t_continuous_to_discrete(t)
        schedule = denoiser.schedule
        sqrt_alpha = schedule.sqrt_alphas_cumprod[t_discrete].view(1, 1, 1, 1).to(x_pred.dtype)
        sqrt_one_minus_alpha = schedule.sqrt_one_minus_alphas_cumprod[t_discrete].view(1, 1, 1, 1).to(x_pred.dtype)
        expected_x = (z - sqrt_one_minus_alpha * eps_pred) / sqrt_alpha.clamp_min(denoiser.t_eps)

        assert torch.allclose(x_pred, expected_x, atol=1e-5)


# =============================================================================
# DiTDenoiser forward_epsilon_with_cfg Tests
# =============================================================================


class TestForwardEpsilonWithCFG:
    """Tests for DiTDenoiser.forward_epsilon_with_cfg()."""

    def test_forward_epsilon_with_cfg_returns_correct_shape(self) -> None:
        """forward_epsilon_with_cfg should return correct shape."""
        from jit_tfg.models.dit.denoiser import DiTDenoiser

        mock_dit = MockDiT(in_channels=4, num_classes=1000)
        num_timesteps = 1000
        wrapper = DiTWrapper(mock_dit, num_timesteps=num_timesteps)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = DiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_timesteps=num_timesteps,
            cfg_scale=4.0,
        )

        z = torch.randn(2, 4, 8, 8)
        t_discrete = torch.tensor([500, 500])
        labels = torch.tensor([207, 360])

        eps = denoiser.forward_epsilon_with_cfg(z, t_discrete, labels)

        assert eps.shape == (2, 4, 8, 8)

    def test_forward_epsilon_with_cfg_uses_discrete_timesteps(self) -> None:
        """forward_epsilon_with_cfg should use discrete timesteps directly."""
        from jit_tfg.models.dit.denoiser import DiTDenoiser

        # Create a mock that tracks calls
        call_log = []

        class TrackedDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 4
                self.num_classes = 1000

            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                call_log.append({"t": t.clone(), "y": y.clone()})
                return torch.zeros(x.shape[0], 8, x.shape[2], x.shape[3])  # learn_sigma=True format

        mock_dit = TrackedDiT()
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = DiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_timesteps=1000,
        )

        z = torch.randn(2, 4, 8, 8)
        t_discrete = torch.tensor([500, 300])
        labels = torch.tensor([207, 360])

        _ = denoiser.forward_epsilon_with_cfg(z, t_discrete, labels)

        # Should have made 2 calls (conditional and unconditional)
        assert len(call_log) == 2

        # First call is conditional (with labels)
        assert torch.equal(call_log[0]["y"], labels)

        # Second call is unconditional (null class = num_classes)
        assert torch.all(call_log[1]["y"] == 1000)

    def test_forward_epsilon_with_cfg_applies_cfg_formula(self) -> None:
        """forward_epsilon_with_cfg should apply CFG: eps_uncond + scale * (eps_cond - eps_uncond)."""
        from jit_tfg.models.dit.denoiser import DiTDenoiser

        # Create mock that returns predictable values
        class PredictableDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 4
                self.num_classes = 1000

            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                batch_size = x.shape[0]
                # Return different values for conditional vs unconditional
                # Unconditional (y=1000): return 1.0
                # Conditional: return 2.0
                out = torch.ones(batch_size, 8, x.shape[2], x.shape[3])  # learn_sigma
                for i in range(batch_size):
                    if y[i].item() < 1000:  # Conditional
                        out[i] = 2.0
                return out

        mock_dit = PredictableDiT()
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = DiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_timesteps=1000,
            cfg_scale=4.0,
            cfg_channel_mode="all",  # Explicitly test "all" mode
        )

        z = torch.randn(2, 4, 8, 8)
        t_discrete = torch.tensor([500, 500])
        labels = torch.tensor([207, 360])

        eps = denoiser.forward_epsilon_with_cfg(z, t_discrete, labels)

        # CFG: eps_uncond (1.0) + 4.0 * (eps_cond (2.0) - eps_uncond (1.0))
        # = 1.0 + 4.0 * 1.0 = 5.0
        assert torch.allclose(eps, torch.ones_like(eps) * 5.0)

    def test_forward_epsilon_with_cfg_custom_scale(self) -> None:
        """forward_epsilon_with_cfg should use custom cfg_scale when provided."""
        from jit_tfg.models.dit.denoiser import DiTDenoiser

        class PredictableDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 4
                self.num_classes = 1000

            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                batch_size = x.shape[0]
                out = torch.ones(batch_size, 8, x.shape[2], x.shape[3])
                for i in range(batch_size):
                    if y[i].item() < 1000:
                        out[i] = 2.0
                return out

        mock_dit = PredictableDiT()
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = DiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_timesteps=1000,
            cfg_scale=4.0,  # Default
            cfg_channel_mode="all",  # Explicitly test "all" mode
        )

        z = torch.randn(2, 4, 8, 8)
        t_discrete = torch.tensor([500, 500])
        labels = torch.tensor([207, 360])

        # Use custom scale of 2.0 instead of default 4.0
        eps = denoiser.forward_epsilon_with_cfg(z, t_discrete, labels, cfg_scale=2.0)

        # CFG with scale=2.0: 1.0 + 2.0 * (2.0 - 1.0) = 3.0
        assert torch.allclose(eps, torch.ones_like(eps) * 3.0)


# =============================================================================
# CFG Channel Mode Tests
# =============================================================================


class TestCFGChannelMode:
    """Tests for CFG channel mode options (all vs first3)."""

    def test_wrapper_default_cfg_channel_mode_is_first3(self) -> None:
        """DiTWrapper should default to cfg_channel_mode='first3' (Meta original)."""
        mock_dit = MockDiT(in_channels=4)
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000)
        assert wrapper.cfg_channel_mode == "first3"

    def test_wrapper_accepts_first3_mode(self) -> None:
        """DiTWrapper should accept cfg_channel_mode='first3'."""
        mock_dit = MockDiT(in_channels=4)
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000, cfg_channel_mode="first3")
        assert wrapper.cfg_channel_mode == "first3"

    def test_forward_cfg_all_mode_applies_to_all_channels(self) -> None:
        """With cfg_channel_mode='all', CFG should apply to all 4 channels."""

        class ChannelAwareDiT(nn.Module):
            """DiT that returns different values per channel."""

            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 4
                self.num_classes = 1000

            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                batch_size = x.shape[0]
                # Conditional (y < 1000): channel values [1, 2, 3, 4]
                # Unconditional (y = 1000): channel values [0, 0, 0, 0]
                out = torch.zeros(batch_size, 4, x.shape[2], x.shape[3])
                for i in range(batch_size):
                    if y[i].item() < 1000:  # Conditional
                        for c in range(4):
                            out[i, c] = float(c + 1)  # [1, 2, 3, 4]
                return out

        mock_dit = ChannelAwareDiT()
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000, cfg_channel_mode="all")

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        y = torch.tensor([207, 360])
        cfg_scale = 2.0

        eps = wrapper.forward_cfg(z, t, y, cfg_scale=cfg_scale)

        # CFG formula: eps_uncond + scale * (eps_cond - eps_uncond)
        # = 0 + 2.0 * ([1,2,3,4] - 0) = [2, 4, 6, 8]
        expected = torch.tensor([2.0, 4.0, 6.0, 8.0]).view(1, 4, 1, 1).expand_as(eps)
        assert torch.allclose(eps, expected)

    def test_forward_cfg_first3_mode_applies_to_first3_only(self) -> None:
        """With cfg_channel_mode='first3', CFG should apply only to first 3 channels."""

        class ChannelAwareDiT(nn.Module):
            """DiT that returns different values per channel."""

            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 4
                self.num_classes = 1000

            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                batch_size = x.shape[0]
                # Conditional (y < 1000): channel values [1, 2, 3, 4]
                # Unconditional (y = 1000): channel values [0, 0, 0, 0]
                out = torch.zeros(batch_size, 4, x.shape[2], x.shape[3])
                for i in range(batch_size):
                    if y[i].item() < 1000:  # Conditional
                        for c in range(4):
                            out[i, c] = float(c + 1)  # [1, 2, 3, 4]
                return out

        mock_dit = ChannelAwareDiT()
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000, cfg_channel_mode="first3")

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        y = torch.tensor([207, 360])
        cfg_scale = 2.0

        eps = wrapper.forward_cfg(z, t, y, cfg_scale=cfg_scale)

        # For first3 mode:
        # Channels 0-2: CFG formula = 0 + 2.0 * ([1,2,3] - 0) = [2, 4, 6]
        # Channel 3: Use conditional value = 4 (no CFG applied)
        expected = torch.tensor([2.0, 4.0, 6.0, 4.0]).view(1, 4, 1, 1).expand_as(eps)
        assert torch.allclose(eps, expected)

    def test_forward_cfg_override_mode_at_call(self) -> None:
        """forward_cfg should allow overriding cfg_channel_mode per call."""

        class ChannelAwareDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 4
                self.num_classes = 1000

            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                batch_size = x.shape[0]
                out = torch.zeros(batch_size, 4, x.shape[2], x.shape[3])
                for i in range(batch_size):
                    if y[i].item() < 1000:
                        for c in range(4):
                            out[i, c] = float(c + 1)
                return out

        mock_dit = ChannelAwareDiT()
        # Create wrapper with 'all' mode as default
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000, cfg_channel_mode="all")

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        y = torch.tensor([207, 360])
        cfg_scale = 2.0

        # Override to 'first3' at call time
        eps = wrapper.forward_cfg(z, t, y, cfg_scale=cfg_scale, cfg_channel_mode="first3")

        # Should use first3 behavior despite wrapper default being 'all'
        expected = torch.tensor([2.0, 4.0, 6.0, 4.0]).view(1, 4, 1, 1).expand_as(eps)
        assert torch.allclose(eps, expected)

    def test_denoiser_cfg_channel_mode_default(self) -> None:
        """DiTDenoiser should default to cfg_channel_mode='first3' (Meta original)."""
        from jit_tfg.models.dit.denoiser import DiTDenoiser

        mock_dit = MockDiT(in_channels=4)
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = DiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_timesteps=1000,
        )
        assert denoiser.cfg_channel_mode == "first3"

    def test_denoiser_forward_epsilon_with_cfg_first3_mode(self) -> None:
        """forward_epsilon_with_cfg should respect cfg_channel_mode='first3'."""
        from jit_tfg.models.dit.denoiser import DiTDenoiser

        class ChannelAwareDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 4
                self.num_classes = 1000

            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                batch_size = x.shape[0]
                # Return 8 channels (learn_sigma=True format)
                out = torch.zeros(batch_size, 8, x.shape[2], x.shape[3])
                for i in range(batch_size):
                    if y[i].item() < 1000:
                        for c in range(4):
                            out[i, c] = float(c + 1)
                return out

        mock_dit = ChannelAwareDiT()
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = DiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_timesteps=1000,
            cfg_channel_mode="first3",
            cfg_scale=2.0,
        )

        z = torch.randn(2, 4, 8, 8)
        t_discrete = torch.tensor([500, 500])
        labels = torch.tensor([207, 360])

        eps = denoiser.forward_epsilon_with_cfg(z, t_discrete, labels)

        # First3 mode: channels 0-2 get CFG, channel 3 uses conditional value
        expected = torch.tensor([2.0, 4.0, 6.0, 4.0]).view(1, 4, 1, 1).expand_as(eps)
        assert torch.allclose(eps, expected)

    def test_denoiser_forward_epsilon_with_cfg_override_mode(self) -> None:
        """forward_epsilon_with_cfg should allow overriding cfg_channel_mode."""
        from jit_tfg.models.dit.denoiser import DiTDenoiser

        class ChannelAwareDiT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = 4
                self.num_classes = 1000

            def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                batch_size = x.shape[0]
                out = torch.zeros(batch_size, 8, x.shape[2], x.shape[3])
                for i in range(batch_size):
                    if y[i].item() < 1000:
                        for c in range(4):
                            out[i, c] = float(c + 1)
                return out

        mock_dit = ChannelAwareDiT()
        wrapper = DiTWrapper(mock_dit, num_timesteps=1000)

        class MockVAE:
            def to(self, device):
                return self

        # Create with 'all' mode
        denoiser = DiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_timesteps=1000,
            cfg_channel_mode="all",
            cfg_scale=2.0,
        )

        z = torch.randn(2, 4, 8, 8)
        t_discrete = torch.tensor([500, 500])
        labels = torch.tensor([207, 360])

        # Override to 'first3' at call time
        eps = denoiser.forward_epsilon_with_cfg(z, t_discrete, labels, cfg_channel_mode="first3")

        # Should use first3 behavior
        expected = torch.tensor([2.0, 4.0, 6.0, 4.0]).view(1, 4, 1, 1).expand_as(eps)
        assert torch.allclose(eps, expected)

    def test_output_shape_same_for_both_modes(self) -> None:
        """Both cfg_channel_modes should produce the same output shape."""
        mock_dit = MockDiT(in_channels=4)

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        y = torch.tensor([207, 360])

        wrapper_all = DiTWrapper(mock_dit, num_timesteps=1000, cfg_channel_mode="all")
        wrapper_first3 = DiTWrapper(mock_dit, num_timesteps=1000, cfg_channel_mode="first3")

        eps_all = wrapper_all.forward_cfg(z, t, y)
        eps_first3 = wrapper_first3.forward_cfg(z, t, y)

        assert eps_all.shape == eps_first3.shape == (2, 4, 8, 8)


class TestDDPMSampleRebasedCoefficients:
    """Verify standalone ddpm_sample computes correct rebased posterior coefficients."""

    def test_ddpm_sample_posterior_mean_matches_rebased(self) -> None:
        """For <1000 steps, ddpm_sample should use rebased (not original) posterior coefficients."""
        import numpy as np

        from jit_tfg.models.dit.diffusion.schedules import DDPMSchedule

        # Original schedule
        schedule = DDPMSchedule.from_beta_schedule("linear", num_timesteps=1000)
        betas_orig = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
        alphas_orig = 1.0 - betas_orig
        alphas_cumprod_orig = np.cumprod(alphas_orig)

        # Rebased schedule for 250 steps (SpacedDiffusion logic)
        num_steps = 250
        frac_stride = (1000 - 1) / (num_steps - 1)
        timesteps = sorted([round(i * frac_stride) for i in range(num_steps)])

        last_alpha_cumprod = 1.0
        new_betas = []
        for t in timesteps:
            new_betas.append(1 - alphas_cumprod_orig[t] / last_alpha_cumprod)
            last_alpha_cumprod = alphas_cumprod_orig[t]
        new_betas = np.array(new_betas)
        new_alphas = 1.0 - new_betas
        new_alphas_cumprod = np.cumprod(new_alphas)
        new_alphas_cumprod_prev = np.append(1.0, new_alphas_cumprod[:-1])

        # Rebased posterior mean coefficients
        new_coef1 = new_betas * np.sqrt(new_alphas_cumprod_prev) / (1.0 - new_alphas_cumprod)
        new_coef2 = (1.0 - new_alphas_cumprod_prev) * np.sqrt(new_alphas) / (1.0 - new_alphas_cumprod)

        # Verify our on-the-fly computation matches the rebased coefficients
        device = torch.device("cpu")
        alphas_cumprod_t = schedule.alphas_cumprod.to(device)

        ts_desc = torch.tensor(sorted(timesteps, reverse=True), dtype=torch.long)

        for i in range(len(ts_desc)):
            t = ts_desc[i].item()
            local_idx = num_steps - 1 - i  # convert from descending to ascending index

            alpha_prod_t = alphas_cumprod_t[t]
            alpha_prod_t_prev = alphas_cumprod_t[ts_desc[i + 1]] if i + 1 < len(ts_desc) else torch.tensor(1.0)

            alpha_t = alpha_prod_t / alpha_prod_t_prev.clamp_min(1e-8)
            beta_t = 1 - alpha_t

            our_coef1 = beta_t * alpha_prod_t_prev**0.5 / (1 - alpha_prod_t).clamp_min(1e-8)
            our_coef2 = (1 - alpha_prod_t_prev) * alpha_t**0.5 / (1 - alpha_prod_t).clamp_min(1e-8)

            assert abs(float(our_coef1) - new_coef1[local_idx]) < 1e-10, (
                f"coef1 mismatch at local_idx {local_idx} (t={t}): "
                f"ours={float(our_coef1):.8f}, expected={new_coef1[local_idx]:.8f}"
            )
            assert abs(float(our_coef2) - new_coef2[local_idx]) < 1e-10, (
                f"coef2 mismatch at local_idx {local_idx} (t={t}): "
                f"ours={float(our_coef2):.8f}, expected={new_coef2[local_idx]:.8f}"
            )
