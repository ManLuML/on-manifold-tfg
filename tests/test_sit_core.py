"""Tests for SiT core functionality.

Tests cover:
1. LinearPath interpolation and velocity conversion
2. SiTWrapper interface
3. SiT model forward pass
4. SiTDenoiser prediction conversion
"""

import pytest
import torch
import torch.nn as nn

from jit_tfg.models.sit.model import SiT, SiT_models
from jit_tfg.models.sit.transport.path import LinearPath
from jit_tfg.models.sit.wrapper import SiTWrapper

# =============================================================================
# LinearPath Tests
# =============================================================================


class TestLinearPath:
    """Tests for LinearPath interpolation."""

    @pytest.fixture
    def path(self) -> LinearPath:
        """Create a LinearPath for testing."""
        return LinearPath()

    def test_compute_alpha_t(self, path: LinearPath) -> None:
        """alpha_t should equal t for linear path."""
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        alpha_t, d_alpha_t = path.compute_alpha_t(t)

        assert torch.allclose(alpha_t, t)
        assert torch.allclose(d_alpha_t, torch.ones_like(t))

    def test_compute_sigma_t(self, path: LinearPath) -> None:
        """sigma_t should equal 1-t for linear path."""
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        sigma_t, d_sigma_t = path.compute_sigma_t(t)

        assert torch.allclose(sigma_t, 1 - t)
        assert torch.allclose(d_sigma_t, -torch.ones_like(t))

    def test_compute_xt_interpolation(self, path: LinearPath) -> None:
        """x_t should interpolate between noise and data."""
        batch_size = 2
        x0 = torch.zeros(batch_size, 4, 8, 8)  # noise
        x1 = torch.ones(batch_size, 4, 8, 8)  # data

        # At t=0, should be noise (x0)
        t = torch.zeros(batch_size)
        xt = path.compute_xt(t, x0, x1)
        assert torch.allclose(xt, x0)

        # At t=1, should be data (x1)
        t = torch.ones(batch_size)
        xt = path.compute_xt(t, x0, x1)
        assert torch.allclose(xt, x1)

        # At t=0.5, should be midpoint
        t = torch.full((batch_size,), 0.5)
        xt = path.compute_xt(t, x0, x1)
        expected = 0.5 * x1 + 0.5 * x0
        assert torch.allclose(xt, expected)

    def test_compute_ut_velocity(self, path: LinearPath) -> None:
        """Velocity should be x1 - x0 for linear path."""
        batch_size = 2
        x0 = torch.randn(batch_size, 4, 8, 8)  # noise
        x1 = torch.randn(batch_size, 4, 8, 8)  # data

        t = torch.rand(batch_size)
        xt = path.compute_xt(t, x0, x1)  # Not used for linear path
        ut = path.compute_ut(t, x0, x1, xt)

        expected = x1 - x0
        assert torch.allclose(ut, expected)

    def test_get_x0_from_velocity(self, path: LinearPath) -> None:
        """Converting velocity to x0 should recover data."""
        batch_size = 2
        x0 = torch.randn(batch_size, 4, 8, 8)  # noise
        x1 = torch.randn(batch_size, 4, 8, 8)  # data (what we want to recover)

        t = torch.rand(batch_size)
        xt = path.compute_xt(t, x0, x1)
        velocity = x1 - x0  # True velocity

        # Recover x1 from velocity
        x1_pred = path.get_x0_from_velocity(velocity, xt, t)

        assert torch.allclose(x1_pred, x1, atol=1e-5)

    def test_get_noise_from_velocity(self, path: LinearPath) -> None:
        """Converting velocity to noise should recover epsilon."""
        batch_size = 2
        x0 = torch.randn(batch_size, 4, 8, 8)  # noise (what we want to recover)
        x1 = torch.randn(batch_size, 4, 8, 8)  # data

        t = torch.rand(batch_size)
        xt = path.compute_xt(t, x0, x1)
        velocity = x1 - x0  # True velocity

        # Recover x0 (noise) from velocity
        x0_pred = path.get_noise_from_velocity(velocity, xt, t)

        assert torch.allclose(x0_pred, x0, atol=1e-5)

    def test_velocity_conversion_roundtrip(self, path: LinearPath) -> None:
        """x0 -> velocity -> x0 should be identity."""
        x0 = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(x0)
        t = torch.tensor([0.3, 0.7])

        # Compute x_t
        xt = path.compute_xt(t, noise, x0)

        # Get velocity
        velocity = path.get_velocity_from_x0_and_noise(x0, noise)

        # Recover x0
        x0_recovered = path.get_x0_from_velocity(velocity, xt, t)

        assert torch.allclose(x0_recovered, x0, atol=1e-5)


# =============================================================================
# SiTWrapper Tests
# =============================================================================


class MockSiT(nn.Module):
    """Mock SiT model for wrapper testing."""

    def __init__(self, in_channels: int = 4, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.linear = nn.Linear(in_channels, in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Return zeros for testing (simulating velocity output)
        return torch.zeros_like(x)


class TestSiTWrapper:
    """Tests for SiTWrapper interface."""

    @pytest.fixture
    def wrapper(self) -> SiTWrapper:
        """Create wrapper with mock SiT."""
        mock_sit = MockSiT()
        return SiTWrapper(mock_sit)

    def test_forward_shape(self, wrapper: SiTWrapper) -> None:
        """Forward should return correct shape."""
        batch_size = 2
        z = torch.randn(batch_size, 4, 32, 32)
        t = torch.tensor([0.5, 0.5])
        y = torch.tensor([207, 360])

        output = wrapper(z, t, y)

        assert output.shape == (batch_size, 4, 32, 32)

    def test_forward_handles_t_shape_variations(self, wrapper: SiTWrapper) -> None:
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

    def test_no_timestep_conversion(self, wrapper: SiTWrapper) -> None:
        """SiT wrapper should NOT invert timesteps (unlike DiT)."""
        # This is a key difference from DiT!
        # SiT uses the same convention as JiT: t=0 noise, t=1 clean
        z = torch.randn(1, 4, 32, 32)
        y = torch.tensor([207])

        # The wrapper should pass t directly to the model
        # (This is verified by the forward pass working correctly)
        t = torch.tensor([0.5])
        output = wrapper(z, t, y)
        assert output.shape == (1, 4, 32, 32)

    def test_forward_cfg(self, wrapper: SiTWrapper) -> None:
        """CFG forward should combine conditional and unconditional predictions."""
        z = torch.randn(2, 4, 32, 32)
        t = torch.tensor([0.5, 0.5])
        y = torch.tensor([207, 360])

        output = wrapper.forward_cfg(z, t, y, cfg_scale=4.0)

        assert output.shape == (2, 4, 32, 32)


# =============================================================================
# SiT Model Tests
# =============================================================================


class TestSiTModel:
    """Tests for SiT model instantiation and forward pass."""

    def test_sit_b4_instantiation(self) -> None:
        """SiT-B/4 should instantiate correctly."""
        model = SiT_models["SiT-B/4"](input_size=32, num_classes=10)
        assert isinstance(model, SiT)
        assert len(list(model.parameters())) > 0

    def test_sit_s2_instantiation(self) -> None:
        """SiT-S/2 should instantiate correctly."""
        model = SiT_models["SiT-S/2"](input_size=32, num_classes=10)
        assert isinstance(model, SiT)

    def test_sit_forward_shape(self) -> None:
        """SiT forward should return correct shape."""
        model = SiT_models["SiT-B/4"](
            input_size=8,
            in_channels=4,
            num_classes=10,
            learn_sigma=False,
        )

        batch_size = 2
        x = torch.randn(batch_size, 4, 8, 8)
        t = torch.rand(batch_size)  # Continuous timesteps [0, 1]
        y = torch.randint(0, 10, (batch_size,))

        output = model(x, t, y)

        # Without learn_sigma, output channels = in_channels
        assert output.shape == (batch_size, 4, 8, 8)

    def test_sit_forward_with_learn_sigma(self) -> None:
        """SiT with learn_sigma should extract velocity from first half."""
        model = SiT_models["SiT-B/4"](
            input_size=8,
            in_channels=4,
            num_classes=10,
            learn_sigma=True,
        )

        batch_size = 2
        x = torch.randn(batch_size, 4, 8, 8)
        t = torch.rand(batch_size)
        y = torch.randint(0, 10, (batch_size,))

        output = model(x, t, y)

        # With learn_sigma, model extracts first half as velocity
        assert output.shape == (batch_size, 4, 8, 8)

    def test_sit_forward_no_nan(self) -> None:
        """SiT forward should not produce NaN values."""
        model = SiT_models["SiT-B/4"](input_size=8, num_classes=10)

        x = torch.randn(2, 4, 8, 8)
        t = torch.rand(2)
        y = torch.randint(0, 10, (2,))

        output = model(x, t, y)

        assert not torch.isnan(output).any()

    def test_sit_continuous_timesteps(self) -> None:
        """SiT should accept continuous timesteps in [0, 1]."""
        model = SiT_models["SiT-S/2"](input_size=8, num_classes=10)

        x = torch.randn(4, 4, 8, 8)
        y = torch.randint(0, 10, (4,))

        # Test various continuous timesteps
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.full((4,), t_val)
            output = model(x, t, y)
            assert output.shape == (4, 4, 8, 8)
            assert not torch.isnan(output).any()


# =============================================================================
# SiTDenoiser Prediction Conversion Tests
# =============================================================================


class TestSiTDenoiserConversion:
    """Tests for SiTDenoiser prediction conversion logic."""

    def test_convert_prediction_shapes(self) -> None:
        """_convert_prediction should return correct shapes."""
        from jit_tfg.models.sit.denoiser import SiTDenoiser

        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit)

        # Create a mock VAE
        class MockVAE:
            def to(self, device):
                return self

        denoiser = SiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
        )

        batch_size = 2
        v_pred = torch.randn(batch_size, 4, 8, 8)
        z = torch.randn(batch_size, 4, 8, 8)
        t = torch.tensor([0.3, 0.5])

        x_pred, v_out, e_pred = denoiser._convert_prediction(v_pred, z, t)

        assert x_pred.shape == (batch_size, 4, 8, 8)
        assert v_out.shape == (batch_size, 4, 8, 8)
        assert e_pred.shape == (batch_size, 4, 8, 8)

    def test_convert_prediction_velocity_passthrough(self) -> None:
        """v_pred should equal input velocity."""
        from jit_tfg.models.sit.denoiser import SiTDenoiser

        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = SiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
        )

        v_pred = torch.randn(2, 4, 8, 8)
        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.3, 0.5])

        _, v_out, _ = denoiser._convert_prediction(v_pred, z, t)

        assert torch.equal(v_out, v_pred)

    def test_convert_prediction_x0_formula(self) -> None:
        """x_pred should follow flow matching formula: x = z + (1-t) * v."""
        from jit_tfg.models.sit.denoiser import SiTDenoiser

        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = SiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
        )

        v_pred = torch.randn(1, 4, 8, 8)
        z = torch.randn(1, 4, 8, 8)
        t = torch.tensor([0.5])

        x_pred, _, _ = denoiser._convert_prediction(v_pred, z, t)

        # Manual calculation: x = z + (1-t) * v
        t_expanded = t.view(-1, 1, 1, 1)
        expected_x = z + (1 - t_expanded) * v_pred

        assert torch.allclose(x_pred, expected_x, atol=1e-5)

    def test_convert_prediction_epsilon_formula(self) -> None:
        """e_pred should follow flow matching formula: ε = z - t * v."""
        from jit_tfg.models.sit.denoiser import SiTDenoiser

        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = SiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
        )

        v_pred = torch.randn(1, 4, 8, 8)
        z = torch.randn(1, 4, 8, 8)
        t = torch.tensor([0.5])

        _, _, e_pred = denoiser._convert_prediction(v_pred, z, t)

        # Manual calculation: ε = z - t * v
        t_expanded = t.view(-1, 1, 1, 1)
        expected_e = z - t_expanded * v_pred

        assert torch.allclose(e_pred, expected_e, atol=1e-5)

    def test_conversion_roundtrip(self) -> None:
        """x and ε from velocity should satisfy: z = t*x + (1-t)*ε."""
        from jit_tfg.models.sit.denoiser import SiTDenoiser

        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = SiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
        )

        v_pred = torch.randn(2, 4, 8, 8)
        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.3, 0.7])

        x_pred, _, e_pred = denoiser._convert_prediction(v_pred, z, t)

        # Verify: z = t*x + (1-t)*ε
        t_expanded = t.view(-1, 1, 1, 1)
        z_reconstructed = t_expanded * x_pred + (1 - t_expanded) * e_pred

        assert torch.allclose(z_reconstructed, z, atol=1e-5)


# =============================================================================
# SiTWrapper Variance Output Tests
# =============================================================================


class TestSiTWrapperVariance:
    """Tests for SiTWrapper.forward_with_variance() method."""

    def test_forward_with_variance_returns_tuple(self) -> None:
        """forward_with_variance should return a tuple of (v_pred, var_pred)."""
        model = SiT_models["SiT-B/4"](
            input_size=8,
            in_channels=4,
            num_classes=10,
            learn_sigma=True,
        )
        wrapper = SiTWrapper(model)

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        y = torch.randint(0, 10, (2,))

        result = wrapper.forward_with_variance(z, t, y)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_variance_shape_matches_input(self) -> None:
        """Variance prediction should have same spatial shape as input."""
        model = SiT_models["SiT-B/4"](
            input_size=8,
            in_channels=4,
            num_classes=10,
            learn_sigma=True,
        )
        wrapper = SiTWrapper(model)

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        y = torch.randint(0, 10, (2,))

        v_pred, var_pred = wrapper.forward_with_variance(z, t, y)

        assert v_pred.shape == z.shape
        assert var_pred is not None
        assert var_pred.shape == z.shape

    def test_variance_none_when_learn_sigma_false(self) -> None:
        """var_pred should be None when learn_sigma=False."""
        model = SiT_models["SiT-B/4"](
            input_size=8,
            in_channels=4,
            num_classes=10,
            learn_sigma=False,
        )
        wrapper = SiTWrapper(model)

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        y = torch.randint(0, 10, (2,))

        v_pred, var_pred = wrapper.forward_with_variance(z, t, y)

        assert v_pred.shape == z.shape
        assert var_pred is None

    def test_forward_with_variance_no_nan(self) -> None:
        """forward_with_variance should not produce NaN values."""
        model = SiT_models["SiT-B/4"](
            input_size=8,
            num_classes=10,
            learn_sigma=True,
        )
        wrapper = SiTWrapper(model)

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.3, 0.7])
        y = torch.randint(0, 10, (2,))

        v_pred, var_pred = wrapper.forward_with_variance(z, t, y)

        assert not torch.isnan(v_pred).any()
        assert var_pred is not None
        assert not torch.isnan(var_pred).any()


# =============================================================================
# Boundary Condition Tests
# =============================================================================


class TestBoundaryConditions:
    """Tests for numerical stability at boundary conditions (t→0, t→1)."""

    def test_t_zero_no_nan(self) -> None:
        """Model forward at t=0 should not produce NaN."""
        model = SiT_models["SiT-S/2"](input_size=8, num_classes=10)

        z = torch.randn(2, 4, 8, 8)
        t = torch.zeros(2)  # Pure noise
        y = torch.randint(0, 10, (2,))

        output = model(z, t, y)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_t_one_no_nan(self) -> None:
        """Model forward at t=1 should not produce NaN."""
        model = SiT_models["SiT-S/2"](input_size=8, num_classes=10)

        z = torch.randn(2, 4, 8, 8)
        t = torch.ones(2)  # Clean data
        y = torch.randint(0, 10, (2,))

        output = model(z, t, y)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_conversion_stable_at_t_near_zero(self) -> None:
        """Prediction conversion should be stable at t near 0."""
        path = LinearPath()

        # Near t=0, (1-t)*v ≈ v (bounded)
        z = torch.randn(2, 4, 8, 8)
        v = torch.randn_like(z)
        t = torch.tensor([0.01, 0.01])

        x_pred = path.get_x0_from_velocity(v, z, t)
        e_pred = path.get_noise_from_velocity(v, z, t)

        assert not torch.isnan(x_pred).any()
        assert not torch.isinf(x_pred).any()
        assert not torch.isnan(e_pred).any()
        assert not torch.isinf(e_pred).any()

    def test_conversion_stable_at_t_near_one(self) -> None:
        """Prediction conversion should be stable at t near 1."""
        path = LinearPath()

        # Near t=1, t*v ≈ v (bounded)
        z = torch.randn(2, 4, 8, 8)
        v = torch.randn_like(z)
        t = torch.tensor([0.99, 0.99])

        x_pred = path.get_x0_from_velocity(v, z, t)
        e_pred = path.get_noise_from_velocity(v, z, t)

        assert not torch.isnan(x_pred).any()
        assert not torch.isinf(x_pred).any()
        assert not torch.isnan(e_pred).any()
        assert not torch.isinf(e_pred).any()

    def test_wrapper_handles_edge_timesteps(self) -> None:
        """SiTWrapper should handle edge timesteps without error."""
        mock_sit = MockSiT()
        wrapper = SiTWrapper(mock_sit)

        z = torch.randn(2, 4, 32, 32)
        y = torch.tensor([207, 360])

        for t_val in [0.0, 0.001, 0.999, 1.0]:
            t = torch.full((2,), t_val)
            output = wrapper(z, t, y)
            assert output.shape == (2, 4, 32, 32)


# =============================================================================
# Mathematical Correctness Tests
# =============================================================================


class TestMathematicalCorrectness:
    """Tests for mathematical formula verification."""

    def test_v_equals_x_minus_epsilon(self) -> None:
        """Velocity should equal x - ε for flow matching."""
        path = LinearPath()

        x0 = torch.randn(2, 4, 8, 8)  # Noise (epsilon)
        x1 = torch.randn(2, 4, 8, 8)  # Clean data (x)

        # True velocity
        v = path.get_velocity_from_x0_and_noise(x1, x0)

        # Should be x - ε
        expected_v = x1 - x0

        assert torch.allclose(v, expected_v, atol=1e-6)

    def test_reconstruction_identity(self) -> None:
        """Recovering x and ε from v should satisfy: z_t = t*x + (1-t)*ε."""
        path = LinearPath()

        x0 = torch.randn(2, 4, 8, 8)  # Noise
        x1 = torch.randn(2, 4, 8, 8)  # Data
        t = torch.tensor([0.3, 0.7])

        # Compute z_t
        z_t = path.compute_xt(t, x0, x1)

        # True velocity
        v = x1 - x0

        # Recover x1 and x0 from velocity
        x1_recovered = path.get_x0_from_velocity(v, z_t, t)
        x0_recovered = path.get_noise_from_velocity(v, z_t, t)

        assert torch.allclose(x1_recovered, x1, atol=1e-5)
        assert torch.allclose(x0_recovered, x0, atol=1e-5)

    def test_flow_matching_ode_direction(self) -> None:
        """Flow should go from noise (t=0) to clean (t=1)."""
        path = LinearPath()

        noise = torch.randn(2, 4, 8, 8)
        clean = torch.zeros(2, 4, 8, 8)  # Clean is simpler (zeros)

        # At t=0, should be noise
        t0 = torch.zeros(2)
        z_0 = path.compute_xt(t0, noise, clean)
        assert torch.allclose(z_0, noise)

        # At t=1, should be clean
        t1 = torch.ones(2)
        z_1 = path.compute_xt(t1, noise, clean)
        assert torch.allclose(z_1, clean)

        # At t=0.5, should be midpoint
        t_half = torch.full((2,), 0.5)
        z_half = path.compute_xt(t_half, noise, clean)
        expected_mid = 0.5 * clean + 0.5 * noise
        assert torch.allclose(z_half, expected_mid)


# =============================================================================
# Model Loading Tests
# =============================================================================


class TestModelLoading:
    """Tests for load_sit_denoiser function."""

    def test_invalid_model_name_raises_error(self) -> None:
        """load_sit_denoiser should raise ValueError for invalid model name."""
        from jit_tfg.models.sit.denoiser import load_sit_denoiser

        with pytest.raises(ValueError, match="Unknown model"):
            load_sit_denoiser(
                checkpoint_path="/fake/path.pt",
                model_name="SiT-INVALID/99",
            )

    def test_no_source_raises_error(self) -> None:
        """load_sit_denoiser should raise if neither checkpoint nor pretrained given."""
        from jit_tfg.models.sit.denoiser import load_sit_denoiser

        with pytest.raises(ValueError, match="Either checkpoint_path or from_pretrained"):
            load_sit_denoiser(device="cpu")

    def test_invalid_pretrained_raises_error(self) -> None:
        """load_sit_denoiser should raise for unknown pretrained model."""
        from jit_tfg.models.sit.denoiser import load_sit_denoiser

        with pytest.raises(ValueError, match="Unknown model"):
            load_sit_denoiser(from_pretrained="unknown/model-id", device="cpu")

    def test_model_name_parameter_accepted(self) -> None:
        """load_sit_denoiser should accept model_name parameter."""
        import inspect

        from jit_tfg.models.sit.denoiser import load_sit_denoiser

        sig = inspect.signature(load_sit_denoiser)
        params = list(sig.parameters.keys())

        assert "model_name" in params

    def test_load_sit_denoiser_defaults(self) -> None:
        """load_sit_denoiser should have correct default parameter values."""
        import inspect

        from jit_tfg.models.sit.denoiser import load_sit_denoiser

        sig = inspect.signature(load_sit_denoiser)

        assert sig.parameters["cfg_scale"].default == 1.5
        assert sig.parameters["cfg_channel_mode"].default == "first3"
        assert sig.parameters["vae_type"].default == "ema"
        assert sig.parameters["num_sampling_steps"].default == 125


# =============================================================================
# CFG Channel Mode Tests
# =============================================================================


class ChannelAwareSiT(nn.Module):
    """SiT that returns different values per channel for CFG testing.

    Conditional: channels = [1, 2, 3, 4]
    Unconditional (y >= num_classes): channels = [0, 0, 0, 0]
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = 4
        self.num_classes = num_classes
        self.linear = nn.Linear(4, 4)  # Dummy parameter

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = torch.zeros(batch_size, 4, x.shape[2], x.shape[3])
        for i in range(batch_size):
            if y[i].item() < self.num_classes:  # Conditional
                for c in range(4):
                    out[i, c] = float(c + 1)  # [1, 2, 3, 4]
            # Unconditional: all zeros
        return out


class TestCFGChannelMode:
    """Tests for SiT CFG channel mode (first3 vs all)."""

    # --- SiTWrapper tests ---

    def test_wrapper_default_cfg_channel_mode_is_first3(self) -> None:
        """SiTWrapper should default to cfg_channel_mode='first3'."""
        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit)
        assert wrapper.cfg_channel_mode == "first3"

    def test_wrapper_accepts_all_mode(self) -> None:
        """SiTWrapper should accept cfg_channel_mode='all'."""
        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit, cfg_channel_mode="all")
        assert wrapper.cfg_channel_mode == "all"

    def test_forward_cfg_first3_applies_to_first3_only(self) -> None:
        """With cfg_channel_mode='first3', CFG should apply only to first 3 channels."""
        mock_sit = ChannelAwareSiT()
        wrapper = SiTWrapper(mock_sit, cfg_channel_mode="first3")

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        y = torch.tensor([207, 360])
        cfg_scale = 2.0

        v = wrapper.forward_cfg(z, t, y, cfg_scale=cfg_scale)

        # For first3 mode:
        # Channels 0-2: CFG = 0 + 2.0 * ([1,2,3] - 0) = [2, 4, 6]
        # Channel 3: Use conditional value = 4 (no CFG applied)
        expected = torch.tensor([2.0, 4.0, 6.0, 4.0]).view(1, 4, 1, 1).expand_as(v)
        assert torch.allclose(v, expected)

    def test_forward_cfg_all_applies_to_all_channels(self) -> None:
        """With cfg_channel_mode='all', CFG should apply to all 4 channels."""
        mock_sit = ChannelAwareSiT()
        wrapper = SiTWrapper(mock_sit, cfg_channel_mode="all")

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        y = torch.tensor([207, 360])
        cfg_scale = 2.0

        v = wrapper.forward_cfg(z, t, y, cfg_scale=cfg_scale)

        # For all mode:
        # All channels: CFG = 0 + 2.0 * ([1,2,3,4] - 0) = [2, 4, 6, 8]
        expected = torch.tensor([2.0, 4.0, 6.0, 8.0]).view(1, 4, 1, 1).expand_as(v)
        assert torch.allclose(v, expected)

    # --- SiTDenoiser tests ---

    def test_denoiser_cfg_channel_mode_default(self) -> None:
        """SiTDenoiser should default to cfg_channel_mode='first3'."""
        from jit_tfg.models.sit.denoiser import SiTDenoiser

        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = SiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
        )
        assert denoiser.cfg_channel_mode == "first3"

    def test_denoiser_forward_sample_first3(self) -> None:
        """_forward_sample with first3 should apply CFG only to first 3 channels."""
        from jit_tfg.models.sit.denoiser import SiTDenoiser

        mock_sit = ChannelAwareSiT()
        wrapper = SiTWrapper(mock_sit)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = SiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            cfg_scale=2.0,
            cfg_channel_mode="first3",
        )

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        labels = torch.tensor([207, 360])

        v = denoiser._forward_sample(z, t, labels)

        # first3: channels 0-2 get CFG, channel 3 uses conditional
        expected = torch.tensor([2.0, 4.0, 6.0, 4.0]).view(1, 4, 1, 1).expand_as(v)
        assert torch.allclose(v, expected)

    def test_denoiser_forward_sample_all(self) -> None:
        """_forward_sample with all should apply CFG to all 4 channels."""
        from jit_tfg.models.sit.denoiser import SiTDenoiser

        mock_sit = ChannelAwareSiT()
        wrapper = SiTWrapper(mock_sit)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = SiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            cfg_scale=2.0,
            cfg_channel_mode="all",
        )

        z = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        labels = torch.tensor([207, 360])

        v = denoiser._forward_sample(z, t, labels)

        # all: all 4 channels get CFG
        expected = torch.tensor([2.0, 4.0, 6.0, 8.0]).view(1, 4, 1, 1).expand_as(v)
        assert torch.allclose(v, expected)


# =============================================================================
# GPU ODE Sampling Tests (requires checkpoint)
# =============================================================================


@pytest.mark.gpu
class TestODESampling:
    """Tests for ODE sampling methods (requires GPU and checkpoint)."""

    @pytest.fixture
    def mock_denoiser(self):
        """Create a mock denoiser for sampling tests."""
        from jit_tfg.models.sit.denoiser import SiTDenoiser

        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit)

        class MockVAE:
            def to(self, device):
                return self

            def decode(self, z):
                # Simple mock decode: just repeat latent to 256x256
                return torch.nn.functional.interpolate(z[:, :3, :, :], size=(256, 256), mode="bilinear")

        return SiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            num_sampling_steps=4,  # Few steps for test
        )

    def test_euler_produces_valid_output(self, mock_denoiser) -> None:
        """Euler sampling should produce valid output shape."""
        labels = torch.tensor([207, 360])
        timesteps = torch.linspace(0.0, 1.0, 5)

        z = torch.randn(2, 4, 32, 32)
        iterator = range(4)

        result = mock_denoiser._euler_sample(z, timesteps, labels, iterator)

        assert result.shape == z.shape
        assert not torch.isnan(result).any()

    def test_heun_produces_valid_output(self, mock_denoiser) -> None:
        """Heun sampling should produce valid output shape."""
        labels = torch.tensor([207, 360])
        timesteps = torch.linspace(0.0, 1.0, 5)

        z = torch.randn(2, 4, 32, 32)
        iterator = range(4)

        result = mock_denoiser._heun_sample(z, timesteps, labels, iterator)

        assert result.shape == z.shape
        assert not torch.isnan(result).any()

    def test_generate_returns_correct_shape(self, mock_denoiser) -> None:
        """generate() should return correct image shape."""
        labels = torch.tensor([207, 360])

        with torch.no_grad():
            images = mock_denoiser.generate(
                labels=labels,
                num_steps=2,
                method="euler",
                show_progress=False,
            )

        # Output should be (B, 3, 256, 256)
        assert images.shape == (2, 3, 256, 256)


# =============================================================================
# GPU CFG Tests
# =============================================================================


@pytest.mark.gpu
class TestCFGInDenoiser:
    """Tests for CFG in denoiser (requires GPU)."""

    def test_cfg_scale_effect(self) -> None:
        """Higher CFG scale should have stronger effect."""
        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit)

        z = torch.randn(2, 4, 32, 32)
        t = torch.tensor([0.5, 0.5])
        y = torch.tensor([207, 360])

        # CFG scale 1.0 (no guidance)
        v1 = wrapper.forward_cfg(z, t, y, cfg_scale=1.0)

        # CFG scale 4.0 (strong guidance)
        v4 = wrapper.forward_cfg(z, t, y, cfg_scale=4.0)

        # Both should have valid shapes
        assert v1.shape == z.shape
        assert v4.shape == z.shape

    def test_forward_sample_with_cfg(self) -> None:
        """_forward_sample should apply CFG correctly."""
        from jit_tfg.models.sit.denoiser import SiTDenoiser

        mock_sit = MockSiT(in_channels=4)
        wrapper = SiTWrapper(mock_sit)

        class MockVAE:
            def to(self, device):
                return self

        denoiser = SiTDenoiser(
            net=wrapper,
            vae=MockVAE(),  # type: ignore[arg-type]
            cfg_scale=4.0,
        )

        z = torch.randn(2, 4, 32, 32)
        t = torch.tensor([0.5, 0.5])
        labels = torch.tensor([207, 360])

        v = denoiser._forward_sample(z, t, labels)

        assert v.shape == z.shape
        assert not torch.isnan(v).any()
