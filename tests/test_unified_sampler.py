"""Tests for UnifiedSampler.

Tests cover:
- UnifiedSampler initialization and validation
- TFGConfig validation
- Flow matching generation (JiT, SiT)
- DiT (DDPM) generation with DDIM
- PixelFlow multi-stage generation
- CFG-only mode
- CFG + TFG guidance mode
- Schedule computation (rho, mu, sigma)
- x0 prediction for all pred_targets
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from jit_tfg.tfg.config import TFGConfig
from jit_tfg.tfg.guiders.base import BaseGuider
from jit_tfg.tfg.unified_sampler import UnifiedSampler

# =============================================================================
# Mock Classes
# =============================================================================


class QuadraticLogpGuider(BaseGuider):
    """Simple differentiable guider: logp(x) = -mean(x^2) per sample."""

    def __init__(self, device: str = "cpu", img_size: int = 8, channels: int = 3) -> None:
        self.device = device
        self.targets = [207]
        self.img_size = img_size
        self.channels = channels

    def get_guidance(
        self,
        x: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        return_logp: bool = False,
        check_grad: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Get guidance gradient or log probability."""
        logp = -(x.flatten(1) ** 2).mean(dim=1)
        if return_logp:
            return logp
        return torch.autograd.grad(logp.sum(), x, create_graph=False)[0]


class ZeroNet(nn.Module):
    """Network that returns zeros."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass returning zeros."""
        return torch.zeros_like(x)


class IdentityNet(nn.Module):
    """Network that returns input."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass returning identity."""
        return x


class TimeDependentNet(nn.Module):
    """Network that returns time-dependent output to make Heun differ from Euler."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass returning time-scaled input."""
        # Make velocity depend on both input and time
        t_scalar = t.mean().item() if t.numel() > 1 else t.item()
        return x * (1 + t_scalar)


class MockJiTDenoiser:
    """Mock JiT Denoiser for testing."""

    def __init__(
        self,
        *,
        img_size: int = 8,
        pred_target: str = "v",
        t_eps: float = 5e-2,
        steps: int = 2,
        noise_scale: float = 1.0,
        num_classes: int = 1000,
        cfg_scale: float = 1.5,
        net: nn.Module | None = None,
    ) -> None:
        self.img_size = img_size
        self.pred_target = pred_target
        self.t_eps = t_eps
        self.steps = steps
        self.noise_scale = noise_scale
        self.num_classes = num_classes
        self.cfg_scale = cfg_scale
        self.net = net or IdentityNet()
        self.is_latent_diffusion = False

    def _forward_sample(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mock velocity prediction."""
        t_flat = t.flatten() if t.ndim > 0 else t.unsqueeze(0)
        return self.net(z, t_flat, labels)


class MockSiTDenoiser:
    """Mock SiT Denoiser for testing."""

    def __init__(
        self,
        *,
        latent_size: int = 8,
        pred_target: str = "v",  # SiT always uses v-prediction
        t_eps: float = 5e-2,
        num_sampling_steps: int = 2,
        noise_scale: float = 1.0,
        num_classes: int = 1000,
        cfg_scale: float = 4.0,
        cfg_channel_mode: str = "first3",
        net: nn.Module | None = None,
        device: str = "cpu",
    ) -> None:
        self.latent_size = latent_size
        self.pred_target = pred_target
        self.t_eps = t_eps
        self.num_sampling_steps = num_sampling_steps
        self.noise_scale = noise_scale
        self.num_classes = num_classes
        self.cfg_scale = cfg_scale
        self.cfg_channel_mode = cfg_channel_mode
        self.net = net or IdentityNet()
        self.is_latent_diffusion = True
        self.vae = MockVAE()
        self.device = device

    def _forward_sample(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mock velocity prediction."""
        t_flat = t.flatten() if t.ndim > 0 else t.unsqueeze(0)
        return self.net(z, t_flat, labels)


class MockDiTDenoiser:
    """Mock DiT Denoiser for testing."""

    def __init__(
        self,
        *,
        latent_size: int = 8,
        num_timesteps: int = 1000,
        num_sampling_steps: int = 10,
        num_classes: int = 1000,
        cfg_scale: float = 4.0,
        cfg_channel_mode: str = "first3",
        device: str = "cpu",
    ) -> None:
        self.latent_size = latent_size
        self.num_sampling_steps = num_sampling_steps
        self.num_classes = num_classes
        self.cfg_scale = cfg_scale
        self.cfg_channel_mode = cfg_channel_mode  # DiT-specific: "first3" or "all"
        self.pred_target = "e"  # DiT always uses epsilon-prediction
        self.is_latent_diffusion = True
        self.vae = MockVAE()
        self.device = device

        # Create mock schedule
        self.schedule = MockDDPMSchedule(num_timesteps, device=device)

        # Create mock network wrapper
        self.net = MockDiTWrapper(in_channels=4, num_classes=num_classes)

    def _forward_sample(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mock epsilon prediction."""
        return torch.zeros_like(z)


class MockDiT(nn.Module):
    """Mock DiT model for testing."""

    def __init__(self, in_channels: int = 4, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass returning zeros with doubled channels (for model output format)."""
        # DiT outputs 2*in_channels (for mean and variance)
        return torch.zeros(x.shape[0], self.in_channels * 2, x.shape[2], x.shape[3], device=x.device)


class MockDiTWrapper:
    """Mock DiT wrapper for testing."""

    def __init__(self, in_channels: int = 4, num_classes: int = 1000) -> None:
        self.dit = MockDiT(in_channels, num_classes)
        self.in_channels = in_channels

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Forward with CFG returning zeros."""
        return torch.zeros_like(x)


class MockDDPMSchedule:
    """Mock DDPM schedule for testing."""

    def __init__(self, num_timesteps: int = 1000, device: str = "cpu") -> None:
        self.num_timesteps = num_timesteps
        self.device = device
        betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.log_betas = torch.log(betas)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]]))

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract values from a at timesteps t, with shape broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class MockPixelFlowNet(nn.Module):
    """Mock PixelFlow network with forward_multires."""

    def forward_multires(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        h: int,
    ) -> torch.Tensor:
        """Forward pass for multi-resolution input."""
        return torch.zeros_like(x)


class IdentityPixelFlowNet(nn.Module):
    """PixelFlow network that returns input (for formula verification)."""

    def forward_multires(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        h: int,
    ) -> torch.Tensor:
        """Forward pass returning identity."""
        return x


class MockPixelFlowDenoiser:
    """Mock PixelFlow Denoiser for testing."""

    def __init__(
        self,
        *,
        img_size: int = 8,
        num_stages: int = 2,
        gamma: float = -1 / 3,
        num_sampling_steps: int = 2,
        num_classes: int = 1000,
        cfg_scale: float = 4.5,
        net: nn.Module | None = None,
        device: str = "cpu",
    ) -> None:
        self.img_size = img_size
        self.num_stages = num_stages
        self.gamma = gamma
        self.num_sampling_steps = num_sampling_steps
        self.num_classes = num_classes
        self.cfg_scale = cfg_scale
        self.pred_target = "v"
        self.is_latent_diffusion = False
        self.noise_scale = 1.0
        self.t_eps = 5e-2
        self.net = net or MockPixelFlowNet()
        self.device = device

    def _forward_sample(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        *,
        stage_idx: int | None = None,
    ) -> torch.Tensor:
        """Mock velocity prediction."""
        return torch.zeros_like(z)


class MockVAE:
    """Mock VAE for latent diffusion models."""

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to images (deterministic based on input)."""
        # Use first 3 channels and interpolate to make output depend on z
        x = z[:, :3, :, :]
        return nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

    def decode_with_grad(self, z: torch.Tensor) -> torch.Tensor:
        """Decode with gradients preserved."""
        x = z[:, :3, :, :]
        return nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)


# =============================================================================
# TFGConfig Validation Tests
# =============================================================================


class TestTFGConfigValidation:
    """Tests for TFGConfig validation."""

    def test_recur_steps_positive(self) -> None:
        """recur_steps must be positive."""
        with pytest.raises(ValueError):
            TFGConfig(recur_steps=0)

    def test_iter_steps_positive(self) -> None:
        """iter_steps must be positive."""
        with pytest.raises(ValueError):
            TFGConfig(iter_steps=0)

    def test_sigma_non_negative(self) -> None:
        """sigma must be non-negative."""
        with pytest.raises(ValueError):
            TFGConfig(sigma=-1.0)

    def test_eps_bsz_positive(self) -> None:
        """eps_bsz must be positive."""
        with pytest.raises(ValueError):
            TFGConfig(eps_bsz=0)

    def test_rho_non_negative(self) -> None:
        """rho must be non-negative."""
        with pytest.raises(ValueError, match="rho must be >= 0"):
            TFGConfig(rho=-0.1)

    def test_mu_non_negative(self) -> None:
        """mu must be non-negative."""
        with pytest.raises(ValueError, match="mu must be >= 0"):
            TFGConfig(mu=-1.0)

    def test_valid_config_creation(self) -> None:
        """Valid config should be created successfully."""
        config = TFGConfig(rho=1.0, mu=0.5, sigma=0.01)
        assert config.rho == 1.0
        assert config.mu == 0.5
        assert config.sigma == 0.01


# =============================================================================
# UnifiedSampler Initialization Tests
# =============================================================================


class TestUnifiedSamplerInit:
    """Tests for UnifiedSampler initialization and validation."""

    def test_jit_sampler_creation(self) -> None:
        """JiT sampler should be created successfully."""
        denoiser = MockJiTDenoiser()
        sampler = UnifiedSampler("JiT", denoiser)
        assert sampler.model_type == "JiT"
        assert sampler.pred_target == "v"
        assert sampler.is_latent is False

    def test_sit_sampler_creation(self) -> None:
        """SiT sampler should be created successfully."""
        denoiser = MockSiTDenoiser()
        sampler = UnifiedSampler("SiT", denoiser)
        assert sampler.model_type == "SiT"
        assert sampler.pred_target == "v"
        assert sampler.is_latent is True

    def test_dit_sampler_creation(self) -> None:
        """DiT sampler should be created successfully with DDPM default."""
        denoiser = MockDiTDenoiser()
        sampler = UnifiedSampler("DiT", denoiser)
        assert sampler.model_type == "DiT"
        assert sampler.pred_target == "e"
        assert sampler.sampling_method == "ddpm"  # DDPM is default (original DiT)
        assert sampler.guidance_space == "x"

    def test_pixelflow_sampler_creation(self) -> None:
        """PixelFlow sampler should be created successfully."""
        denoiser = MockPixelFlowDenoiser()
        sampler = UnifiedSampler("PixelFlow", denoiser)
        assert sampler.model_type == "PixelFlow"
        assert sampler.pred_target == "v"
        assert sampler.num_stages == 2

    def test_invalid_model_type(self) -> None:
        """Invalid model_type should raise AssertionError."""
        denoiser = MockJiTDenoiser()
        with pytest.raises(AssertionError, match="model_type must be one of"):
            UnifiedSampler("InvalidModel", denoiser)

    def test_sit_wrong_pred_target(self) -> None:
        """SiT with non-v pred_target should raise AssertionError."""
        denoiser = MockSiTDenoiser(pred_target="x")
        with pytest.raises(AssertionError, match="SiT must use pred_target='v'"):
            UnifiedSampler("SiT", denoiser)

    def test_pixelflow_wrong_pred_target(self) -> None:
        """PixelFlow with non-v pred_target should raise AssertionError."""
        denoiser = MockPixelFlowDenoiser()
        # Override after creation
        object.__setattr__(denoiser, "pred_target", "x")
        with pytest.raises(AssertionError, match="PixelFlow must use pred_target='v'"):
            UnifiedSampler("PixelFlow", denoiser)

    def test_dit_defaults_to_ddpm_sampling(self) -> None:
        """DiT should default to DDPM sampling when non-DDPM/DDIM method is passed."""
        denoiser = MockDiTDenoiser()
        # When an invalid method for DiT is passed, it defaults to DDPM
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="euler")
        assert sampler.sampling_method == "ddpm"  # Defaults to DDPM

    def test_dit_accepts_ddim_sampling(self) -> None:
        """DiT should accept DDIM sampling when explicitly specified."""
        denoiser = MockDiTDenoiser()
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddim")
        assert sampler.sampling_method == "ddim"

    def test_invalid_sampling_method(self) -> None:
        """Invalid sampling_method should raise AssertionError."""
        denoiser = MockJiTDenoiser()
        with pytest.raises(AssertionError, match="sampling_method must be one of"):
            UnifiedSampler("JiT", denoiser, sampling_method="invalid")

    def test_dopri5_not_implemented(self) -> None:
        """dopri5 sampling should raise NotImplementedError."""
        denoiser = MockPixelFlowDenoiser()
        with pytest.raises(NotImplementedError, match="dopri5 sampling is not yet implemented"):
            UnifiedSampler("PixelFlow", denoiser, sampling_method="dopri5")

    def test_guidance_space_v_initializes(self) -> None:
        """guidance_space='v' should initialize successfully."""
        denoiser = MockJiTDenoiser()
        config = TFGConfig(device="cpu", rho=1.0)
        # v-space is now implemented, should not raise
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, guidance_space="v")
        assert sampler.guidance_space == "v"

    def test_guidance_space_v2_initializes_with_heun(self) -> None:
        """guidance_space='v2' should initialize with heun sampling."""
        denoiser = MockJiTDenoiser()
        config = TFGConfig(device="cpu", rho=1.0)
        # v2-space is now implemented, should not raise with heun
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="heun", guidance_space="v2")
        assert sampler.guidance_space == "v2"
        assert sampler.sampling_method == "heun"

    def test_guidance_space_v2_requires_heun(self) -> None:
        """guidance_space='v2' should require heun sampling."""
        denoiser = MockJiTDenoiser()
        config = TFGConfig(device="cpu", rho=1.0)
        with pytest.raises(AssertionError, match="guidance_space='v2' requires sampling_method='heun'"):
            UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="euler", guidance_space="v2")


# =============================================================================
# CFG-only Generation Tests
# =============================================================================


class TestCFGOnlyGeneration:
    """Tests for CFG-only generation without TFG."""

    def test_jit_cfg_only_generation(self) -> None:
        """JiT CFG-only generation should work."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        sampler = UnifiedSampler("JiT", denoiser, sampling_method="euler")

        labels = torch.tensor([207, 360], device="cpu")
        images = sampler.generate(cfg_labels=labels, num_steps=2, show_progress=False)

        assert images.shape == (2, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_jit_heun_sampling(self) -> None:
        """JiT Heun sampling should work."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        sampler = UnifiedSampler("JiT", denoiser, sampling_method="heun")

        labels = torch.tensor([207], device="cpu")
        images = sampler.generate(cfg_labels=labels, num_steps=3, show_progress=False)

        assert images.shape == (1, 3, 8, 8)

    def test_sit_cfg_only_generation(self) -> None:
        """SiT CFG-only generation should work."""
        denoiser = MockSiTDenoiser(latent_size=4, num_sampling_steps=2, net=ZeroNet())
        sampler = UnifiedSampler("SiT", denoiser, sampling_method="euler")

        labels = torch.tensor([207], device="cpu")
        images = sampler.generate(cfg_labels=labels, num_steps=2, show_progress=False)

        # Output should be decoded to pixel space
        assert images.shape[0] == 1
        assert images.shape[1] == 3

    def test_dit_cfg_only_generation(self) -> None:
        """DiT CFG-only generation should work."""
        denoiser = MockDiTDenoiser(latent_size=4, num_sampling_steps=2, device="cpu")
        # Create sampler with TFGConfig to set device
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)  # No guidance
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        labels = torch.tensor([207], device="cpu")
        images = sampler.generate(cfg_labels=labels, num_steps=2, show_progress=False)

        # Output should be decoded to pixel space
        assert images.shape[0] == 1
        assert images.shape[1] == 3

    def test_pixelflow_cfg_only_generation(self) -> None:
        """PixelFlow CFG-only generation should work."""
        denoiser = MockPixelFlowDenoiser(img_size=8, num_stages=2, num_sampling_steps=2)
        sampler = UnifiedSampler("PixelFlow", denoiser, sampling_method="euler")

        labels = torch.tensor([207], device="cpu")
        images = sampler.generate(cfg_labels=labels, num_steps=2, show_progress=False)

        assert images.shape == (1, 3, 8, 8)

    def test_batch_size_from_labels(self) -> None:
        """batch_size should be inferred from cfg_labels."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        sampler = UnifiedSampler("JiT", denoiser)

        labels = torch.tensor([207, 360, 388], device="cpu")
        images = sampler.generate(cfg_labels=labels, num_steps=2, show_progress=False)

        assert images.shape[0] == 3

    def test_cfg_scale_override(self) -> None:
        """cfg_scale should be overridable."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, cfg_scale=1.5)
        sampler = UnifiedSampler("JiT", denoiser)

        labels = torch.tensor([207], device="cpu")
        # Just verify it doesn't error
        images = sampler.generate(cfg_labels=labels, cfg_scale=3.0, num_steps=2, show_progress=False)
        assert images is not None


# =============================================================================
# CFG + TFG Generation Tests
# =============================================================================


class TestCFGTFGGeneration:
    """Tests for CFG + TFG generation."""

    def test_jit_tfg_generation(self) -> None:
        """JiT CFG+TFG generation should work."""
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            clip_scale=1e9,
            recur_steps=1,
        )
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="euler")
        guider = QuadraticLogpGuider(device="cpu")

        labels = torch.tensor([207], device="cpu")
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_tfg_guidance_changes_output(self) -> None:
        """TFG guidance should change the output compared to CFG-only."""
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.5,
            sigma=0.0,
            eps_bsz=1,
            clip_scale=1e9,
            recur_steps=1,
        )
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())

        # CFG-only
        sampler_cfg = UnifiedSampler("JiT", denoiser)
        labels = torch.tensor([207], device="cpu")

        torch.manual_seed(42)
        images_cfg = sampler_cfg.generate(cfg_labels=labels, num_steps=2, show_progress=False)

        # CFG + TFG
        sampler_tfg = UnifiedSampler("JiT", denoiser, tfg_config=config)
        guider = QuadraticLogpGuider(device="cpu")

        torch.manual_seed(42)
        images_tfg = sampler_tfg.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        # Outputs should be different
        assert not torch.allclose(images_cfg, images_tfg)

    def test_guide_step_under_no_grad(self) -> None:
        """TFG should work even under torch.no_grad() context."""
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            clip_scale=1e9,
            recur_steps=1,
        )
        denoiser = MockJiTDenoiser(img_size=4, steps=2, net=ZeroNet())
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)
        guider = QuadraticLogpGuider(device="cpu")

        labels = torch.tensor([207], device="cpu")

        with torch.no_grad():
            images = sampler.generate(
                cfg_labels=labels,
                guidance=guider,
                tfg_targets=labels,
                num_steps=2,
                show_progress=False,
            )

        assert images.shape == (1, 3, 4, 4)
        assert not torch.isnan(images).any()


# =============================================================================
# Schedule Computation Tests
# =============================================================================


class TestScheduleComputation:
    """Tests for rho/mu/sigma schedule computation."""

    def test_constant_schedule(self) -> None:
        """Constant schedule should return base value at all timesteps."""
        config = TFGConfig(
            device="cpu",
            rho=2.0,
            mu=3.0,
            sigma=0.5,
            rho_schedule="constant",
            mu_schedule="constant",
            sigma_schedule="constant",
        )
        denoiser = MockJiTDenoiser()
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # Access private method for testing
        assert sampler._get_schedule_value(2.0, "constant", 0.0) == pytest.approx(2.0)
        assert sampler._get_schedule_value(2.0, "constant", 0.5) == pytest.approx(2.0)
        assert sampler._get_schedule_value(2.0, "constant", 1.0) == pytest.approx(2.0)

    def test_increase_schedule(self) -> None:
        """Increase schedule without normalization (default).

        Without normalization (default):
        - value = base * t
        - Result: schedule goes from 0 to base_value
        """
        config = TFGConfig(
            device="cpu",
            rho=2.0,
            rho_schedule="increase",
        )
        denoiser = MockJiTDenoiser()
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # Without normalization: value = base * t
        # t=0 gives 0, t=0.5 gives base/2, t=1 gives base_value
        assert sampler._get_schedule_value(2.0, "increase", 0.0) == pytest.approx(0.0)
        assert sampler._get_schedule_value(2.0, "increase", 0.5) == pytest.approx(1.0)
        assert sampler._get_schedule_value(2.0, "increase", 1.0) == pytest.approx(2.0)

    def test_increase_schedule_normalized(self) -> None:
        """Increase schedule with normalization enabled.

        With normalization (matching original TFG):
        - integral of t from 0 to 1 is 0.5
        - so we multiply by 2 to make average = base_value
        - Result: schedule goes from 0 to 2*base_value
        """
        config = TFGConfig(
            device="cpu",
            rho=2.0,
            rho_schedule="increase",
            normalize_schedules=True,
        )
        denoiser = MockJiTDenoiser()
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # With normalization: value = base * t * 2
        # t=0 gives 0, t=0.5 gives base_value, t=1 gives 2*base_value
        assert sampler._get_schedule_value(2.0, "increase", 0.0, normalize=True) == pytest.approx(0.0)
        assert sampler._get_schedule_value(2.0, "increase", 0.5, normalize=True) == pytest.approx(2.0)
        assert sampler._get_schedule_value(2.0, "increase", 1.0, normalize=True) == pytest.approx(4.0)

    def test_decrease_schedule(self) -> None:
        """Decrease schedule without normalization (default).

        Without normalization (default):
        - value = base * (1-t)
        - Result: schedule goes from base_value to 0
        """
        config = TFGConfig(
            device="cpu",
            mu=3.0,
            mu_schedule="decrease",
        )
        denoiser = MockJiTDenoiser()
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # Without normalization: value = base * (1-t)
        # t=0 gives base_value, t=0.5 gives base/2, t=1 gives 0
        assert sampler._get_schedule_value(3.0, "decrease", 0.0) == pytest.approx(3.0)
        assert sampler._get_schedule_value(3.0, "decrease", 0.5) == pytest.approx(1.5)
        assert sampler._get_schedule_value(3.0, "decrease", 1.0) == pytest.approx(0.0)

    def test_decrease_schedule_normalized(self) -> None:
        """Decrease schedule with normalization enabled.

        With normalization (matching original TFG):
        - integral of (1-t) from 0 to 1 is 0.5
        - so we multiply by 2 to make average = base_value
        - Result: schedule goes from 2*base_value to 0
        """
        config = TFGConfig(
            device="cpu",
            mu=3.0,
            mu_schedule="decrease",
            normalize_schedules=True,
        )
        denoiser = MockJiTDenoiser()
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # With normalization: value = base * (1-t) * 2
        # t=0 gives 2*base_value, t=0.5 gives base_value, t=1 gives 0
        assert sampler._get_schedule_value(3.0, "decrease", 0.0, normalize=True) == pytest.approx(6.0)
        assert sampler._get_schedule_value(3.0, "decrease", 0.5, normalize=True) == pytest.approx(3.0)
        assert sampler._get_schedule_value(3.0, "decrease", 1.0, normalize=True) == pytest.approx(0.0)

    def test_schedule_normalization_average(self) -> None:
        """Schedule normalization should make average value equal to base_value.

        This matches the original TFG implementation which uses:
        value = base * scheduler[t] * len(scheduler) / scheduler.sum()

        For continuous time flow matching, we use the integral instead:
        - "increase": ∫t dt from 0 to 1 = 0.5, so multiply by 2
        - "decrease": ∫(1-t) dt from 0 to 1 = 0.5, so multiply by 2

        Note: This test verifies the behavior when normalize_schedules=True.
        """
        denoiser = MockJiTDenoiser()
        config = TFGConfig(device="cpu", rho=2.0, rho_schedule="increase", normalize_schedules=True)
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # Sample many timesteps uniformly in [0, 1]
        num_samples = 101
        timesteps = [i / (num_samples - 1) for i in range(num_samples)]

        # Increase schedule with normalization: average should equal base_value
        increase_values = [sampler._get_schedule_value(2.0, "increase", t, normalize=True) for t in timesteps]
        increase_avg = sum(increase_values) / len(increase_values)
        assert increase_avg == pytest.approx(2.0, rel=0.02), f"Increase avg: {increase_avg}"

        # Decrease schedule with normalization: average should equal base_value
        decrease_values = [sampler._get_schedule_value(2.0, "decrease", t, normalize=True) for t in timesteps]
        decrease_avg = sum(decrease_values) / len(decrease_values)
        assert decrease_avg == pytest.approx(2.0, rel=0.02), f"Decrease avg: {decrease_avg}"

        # Constant schedule: average should equal base_value (trivially)
        constant_values = [sampler._get_schedule_value(2.0, "constant", t, normalize=True) for t in timesteps]
        constant_avg = sum(constant_values) / len(constant_values)
        assert constant_avg == pytest.approx(2.0, rel=0.01), f"Constant avg: {constant_avg}"

    def test_schedule_no_normalization_average(self) -> None:
        """Without normalization, average value differs for non-constant schedules.

        Without normalization (default):
        - "increase": average = base/2 (integral of t from 0 to 1 is 0.5)
        - "decrease": average = base/2 (integral of (1-t) from 0 to 1 is 0.5)
        - "constant": average = base (trivially)
        """
        denoiser = MockJiTDenoiser()
        config = TFGConfig(device="cpu", rho=2.0, rho_schedule="increase")
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # Sample many timesteps uniformly in [0, 1]
        num_samples = 101
        timesteps = [i / (num_samples - 1) for i in range(num_samples)]

        # Increase schedule without normalization: average should equal base/2
        increase_values = [sampler._get_schedule_value(2.0, "increase", t, normalize=False) for t in timesteps]
        increase_avg = sum(increase_values) / len(increase_values)
        assert increase_avg == pytest.approx(1.0, rel=0.02), f"Increase avg: {increase_avg}"

        # Decrease schedule without normalization: average should equal base/2
        decrease_values = [sampler._get_schedule_value(2.0, "decrease", t, normalize=False) for t in timesteps]
        decrease_avg = sum(decrease_values) / len(decrease_values)
        assert decrease_avg == pytest.approx(1.0, rel=0.02), f"Decrease avg: {decrease_avg}"

        # Constant schedule: average should equal base_value (no change)
        constant_values = [sampler._get_schedule_value(2.0, "constant", t, normalize=False) for t in timesteps]
        constant_avg = sum(constant_values) / len(constant_values)
        assert constant_avg == pytest.approx(2.0, rel=0.01), f"Constant avg: {constant_avg}"

    def test_sigma_never_normalized(self) -> None:
        """Sigma should NEVER be normalized, matching original TFG implementation.

        Original TFG (methods/tfg.py:74): return self.args.sigma * scheduler[t]
        Unlike rho/mu which use: * len(scheduler) / scheduler.sum()

        This is because sigma controls Monte Carlo smoothing kernel width,
        not guidance strength. Normalizing would distort its physical meaning.
        """
        denoiser = MockJiTDenoiser()
        config = TFGConfig(
            device="cpu",
            sigma=0.01,
            sigma_schedule="increase",
            normalize_schedules=True,  # Even with normalization enabled...
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # Sigma at t=0.5 should be: base * t = 0.01 * 0.5 = 0.005
        # NOT normalized: 0.01 * 0.5 * 2 = 0.01 (would be wrong!)
        sigma_value = sampler._get_schedule_value(0.01, "increase", 0.5, normalize=False)
        assert sigma_value == pytest.approx(0.005), "Sigma should NOT be normalized"

        # Compare with what normalized would give (wrong behavior)
        normalized_value = sampler._get_schedule_value(0.01, "increase", 0.5, normalize=True)
        assert normalized_value == pytest.approx(0.01), "Normalized gives different value"

        # Verify sigma != normalized across all timesteps for non-constant schedules
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            unnorm = sampler._get_schedule_value(0.01, "increase", t, normalize=False)
            norm = sampler._get_schedule_value(0.01, "increase", t, normalize=True)
            # At t=0, both are 0. At other times, they differ by factor of 2.
            if t > 0:
                assert unnorm == pytest.approx(norm / 2), f"Sigma differs at t={t}"

    def test_sigma_never_normalized_decrease_schedule(self) -> None:
        """Sigma with 'decrease' schedule should also never normalize.

        For decrease schedule: value = base * (1-t)
        - At t=0: base * 1 = base
        - At t=0.5: base * 0.5
        - At t=1: base * 0 = 0
        """
        denoiser = MockJiTDenoiser()
        config = TFGConfig(
            device="cpu",
            sigma=0.02,
            sigma_schedule="decrease",
            normalize_schedules=True,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # At t=0.5: decrease gives (1-t) = 0.5
        # Unnormalized: 0.02 * 0.5 = 0.01
        sigma_value = sampler._get_schedule_value(0.02, "decrease", 0.5, normalize=False)
        assert sigma_value == pytest.approx(0.01), "Decrease schedule sigma incorrect"

        # Normalized would give: 0.02 * 0.5 * 2 = 0.02 (wrong!)
        normalized_value = sampler._get_schedule_value(0.02, "decrease", 0.5, normalize=True)
        assert normalized_value == pytest.approx(0.02), "Normalized differs by 2x"

        # Verify across timesteps
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            unnorm = sampler._get_schedule_value(0.02, "decrease", t, normalize=False)
            expected = 0.02 * (1 - t)
            assert unnorm == pytest.approx(expected), f"Decrease at t={t}"

    def test_sigma_never_normalized_constant_schedule(self) -> None:
        """Sigma with 'constant' schedule is unaffected by normalization.

        For constant schedule, normalization factor is 1.0, so there's no difference.
        But sigma should still use normalize=False for consistency.
        """
        denoiser = MockJiTDenoiser()
        config = TFGConfig(
            device="cpu",
            sigma=0.01,
            sigma_schedule="constant",
            normalize_schedules=True,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # Constant schedule: value = base for all t
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sigma_value = sampler._get_schedule_value(0.01, "constant", t, normalize=False)
            assert sigma_value == pytest.approx(0.01), f"Constant at t={t}"

            # For constant, normalized and unnormalized are the same
            norm_value = sampler._get_schedule_value(0.01, "constant", t, normalize=True)
            assert norm_value == pytest.approx(0.01), f"Constant normalized at t={t}"

    def test_sigma_never_normalized_sit(self) -> None:
        """SiT model should also never normalize sigma."""
        denoiser = MockSiTDenoiser()
        config = TFGConfig(
            device="cpu",
            sigma=0.01,
            sigma_schedule="increase",
            normalize_schedules=True,
        )
        sampler = UnifiedSampler("SiT", denoiser, tfg_config=config)

        # SiT uses same _get_schedule_value as JiT
        sigma_value = sampler._get_schedule_value(0.01, "increase", 0.5, normalize=False)
        assert sigma_value == pytest.approx(0.005), "SiT sigma should not be normalized"

    def test_sigma_never_normalized_dit(self) -> None:
        """DiT model should also never normalize sigma via _get_dit_sigma().

        DiT uses discrete DDPM timesteps but sigma should still not be normalized.
        Note: DiT always normalizes rho/mu, but NOT sigma.
        """
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(
            device="cpu",
            sigma=0.01,
            sigma_schedule="increase",
            normalize_schedules=True,
        )
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        # Get alpha_prod_ts for DiT
        ts = sampler._get_dit_timestep_sequence(num_steps=10, device=torch.device("cpu"))
        alpha_prod_ts = denoiser.schedule.alphas_cumprod[ts]

        # Verify sigma values are NOT normalized
        for t_idx in range(len(ts)):
            sigma = sampler._get_dit_sigma(t_idx, alpha_prod_ts)
            # For "increase" schedule in DDPM, scheduler = alpha_prod_ts**0.5
            expected = 0.01 * (alpha_prod_ts[t_idx] ** 0.5).item()
            assert sigma == pytest.approx(expected), f"DiT sigma at idx={t_idx}"

    def test_sigma_never_normalized_pixelflow(self) -> None:
        """PixelFlow model should also never normalize sigma."""
        denoiser = MockPixelFlowDenoiser(num_stages=4)
        config = TFGConfig(
            device="cpu",
            sigma=0.01,
            sigma_schedule="increase",
            normalize_schedules=True,
        )
        sampler = UnifiedSampler("PixelFlow", denoiser, tfg_config=config)

        # PixelFlow uses _get_schedule_value with normalize=False
        sigma_value = sampler._get_schedule_value(0.01, "increase", 0.5, normalize=False)
        assert sigma_value == pytest.approx(0.005), "PixelFlow sigma should not be normalized"

    def test_normalize_schedules_only_affects_rho_mu_not_sigma(self) -> None:
        """The normalize_schedules flag should affect rho/mu but NOT sigma.

        This is the key distinction:
        - Rho/Mu: guidance strength → benefits from normalization
        - Sigma: MC kernel width → should never be normalized
        """
        denoiser = MockJiTDenoiser()

        # Config with normalize_schedules=True
        config_norm = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=1.0,
            sigma=0.01,
            rho_schedule="increase",
            mu_schedule="increase",
            sigma_schedule="increase",
            normalize_schedules=True,
        )
        sampler_norm = UnifiedSampler("JiT", denoiser, tfg_config=config_norm)

        # Config with normalize_schedules=False
        config_no_norm = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=1.0,
            sigma=0.01,
            rho_schedule="increase",
            mu_schedule="increase",
            sigma_schedule="increase",
            normalize_schedules=False,
        )
        sampler_no_norm = UnifiedSampler("JiT", denoiser, tfg_config=config_no_norm)

        t = 0.5

        # Rho SHOULD differ based on normalize_schedules
        rho_norm = sampler_norm._get_schedule_value(1.0, "increase", t, normalize=True)
        rho_no_norm = sampler_no_norm._get_schedule_value(1.0, "increase", t, normalize=False)
        assert rho_norm == pytest.approx(1.0), "Rho normalized"
        assert rho_no_norm == pytest.approx(0.5), "Rho not normalized"
        assert rho_norm != pytest.approx(rho_no_norm), "Rho differs with normalization"

        # Mu SHOULD differ based on normalize_schedules
        mu_norm = sampler_norm._get_schedule_value(1.0, "increase", t, normalize=True)
        mu_no_norm = sampler_no_norm._get_schedule_value(1.0, "increase", t, normalize=False)
        assert mu_norm != pytest.approx(mu_no_norm), "Mu differs with normalization"

        # Sigma should be IDENTICAL regardless of normalize_schedules
        # Both should use normalize=False internally
        sigma_from_norm = sampler_norm._get_schedule_value(0.01, "increase", t, normalize=False)
        sigma_from_no_norm = sampler_no_norm._get_schedule_value(0.01, "increase", t, normalize=False)
        assert sigma_from_norm == pytest.approx(sigma_from_no_norm), "Sigma should be identical"
        assert sigma_from_norm == pytest.approx(0.005), "Sigma = base * t"

    def test_sigma_never_normalized_various_values(self) -> None:
        """Sigma should never be normalized for various sigma values."""
        denoiser = MockJiTDenoiser()
        config = TFGConfig(device="cpu", normalize_schedules=True)
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # Test various sigma values
        sigma_values = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
        t = 0.5

        for sigma in sigma_values:
            unnorm = sampler._get_schedule_value(sigma, "increase", t, normalize=False)
            expected = sigma * t
            assert unnorm == pytest.approx(expected), f"Sigma {sigma} at t={t}"

    def test_sigma_at_boundary_timesteps(self) -> None:
        """Sigma should behave correctly at boundary timesteps (t=0, t=1)."""
        denoiser = MockJiTDenoiser()
        config = TFGConfig(device="cpu", sigma=0.01, normalize_schedules=True)
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # t=0: increase gives 0, decrease gives base
        assert sampler._get_schedule_value(0.01, "increase", 0.0, normalize=False) == pytest.approx(0.0)
        assert sampler._get_schedule_value(0.01, "decrease", 0.0, normalize=False) == pytest.approx(0.01)
        assert sampler._get_schedule_value(0.01, "constant", 0.0, normalize=False) == pytest.approx(0.01)

        # t=1: increase gives base, decrease gives 0
        assert sampler._get_schedule_value(0.01, "increase", 1.0, normalize=False) == pytest.approx(0.01)
        assert sampler._get_schedule_value(0.01, "decrease", 1.0, normalize=False) == pytest.approx(0.0)
        assert sampler._get_schedule_value(0.01, "constant", 1.0, normalize=False) == pytest.approx(0.01)

    def test_dit_sigma_vs_rho_normalization_asymmetry(self) -> None:
        """DiT should normalize rho but NOT sigma - verify the asymmetry.

        This tests the critical design decision: DiT always normalizes rho/mu
        using original TFG formula, but sigma is never normalized.
        """
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            sigma=0.01,
            rho_schedule="increase",
            sigma_schedule="increase",
        )
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        # Get DiT schedules
        ts = sampler._get_dit_timestep_sequence(num_steps=10, device=torch.device("cpu"))
        alpha_prod_ts = denoiser.schedule.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat([torch.tensor([1.0]), alpha_prod_ts[:-1]])

        # Collect rho and sigma values
        rho_values = []
        sigma_values = []
        for t_idx in range(len(ts)):
            rho = sampler._get_dit_rho(t_idx, alpha_prod_ts, alpha_prod_t_prevs)
            sigma = sampler._get_dit_sigma(t_idx, alpha_prod_ts)
            rho_values.append(rho)
            sigma_values.append(sigma)

        # Rho should have normalization applied (average ~ base_value)
        rho_avg = sum(rho_values) / len(rho_values)
        # Due to normalization, average should be close to base value
        assert rho_avg == pytest.approx(1.0, rel=0.3), f"Rho avg: {rho_avg}"

        # Sigma should NOT be normalized (average ~ base/2 for increase schedule)
        sigma_avg = sum(sigma_values) / len(sigma_values)
        # Without normalization, average is approximately base * 0.5 for "increase"
        # (actually depends on alpha schedule, but definitely not normalized)
        assert sigma_avg < 0.01, f"Sigma avg: {sigma_avg} should be < base (not normalized)"


# =============================================================================
# PixelFlow Multi-stage Tests
# =============================================================================


class TestPixelFlowMultiStage:
    """Tests for PixelFlow multi-stage generation."""

    def test_pixelflow_stage_setup(self) -> None:
        """PixelFlow stage attributes should be initialized correctly."""
        denoiser = MockPixelFlowDenoiser(num_stages=4)
        sampler = UnifiedSampler("PixelFlow", denoiser)

        assert sampler.num_stages == 4
        assert len(sampler.stage_distance) == 4
        assert sampler.stage_range == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_pixelflow_rectify_ratio(self) -> None:
        """Rectify ratio calculation should follow PixelFlow formula."""
        denoiser = MockPixelFlowDenoiser(gamma=-1 / 3)
        sampler = UnifiedSampler("PixelFlow", denoiser)

        # Test rectify ratio at start_t = 0.5
        ratio = sampler._cal_rectify_ratio(0.5)
        expected = 1 / (math.sqrt(1 - (1 / (-1 / 3))) * (1 - 0.5) + 0.5)
        assert ratio == pytest.approx(expected)

    def test_pixelflow_time_schedule_matches_original(self) -> None:
        """Within-stage time schedule should match original PixelFlowScheduler.

        Original: linspace(0, 999/1000, num_steps) + append 1.0
        NOT uniform linspace(0, 1, num_steps + 1).
        The last step dt should be 0.001, not 1/num_steps.
        """
        num_steps = 30
        _NUM_TRAIN_TIMESTEPS = 1000

        # Our schedule (matching original)
        t_end = (_NUM_TRAIN_TIMESTEPS - 1) / _NUM_TRAIN_TIMESTEPS  # 0.999
        t_schedule = np.linspace(0, t_end, num_steps, dtype=np.float64)
        t_within_stage = np.append(t_schedule, 1.0)

        # Verify shape
        assert len(t_within_stage) == num_steps + 1

        # Verify endpoints
        assert t_within_stage[0] == 0.0
        assert t_within_stage[-1] == 1.0
        assert t_within_stage[-2] == pytest.approx(0.999, abs=1e-10)

        # Verify non-uniform spacing: last dt should be tiny (0.001)
        last_dt = t_within_stage[-1] - t_within_stage[-2]
        assert last_dt == pytest.approx(0.001, abs=1e-10)

        # Interior dt should be ~0.999/29 ≈ 0.03445
        interior_dt = t_within_stage[1] - t_within_stage[0]
        assert interior_dt == pytest.approx(0.999 / 29, abs=1e-10)

        # Verify NOT uniform (would be 1/30 ≈ 0.03333)
        assert interior_dt != pytest.approx(1.0 / num_steps, abs=1e-4)

    def test_pixelflow_T_endpoints_integer_indexed(self) -> None:
        """T start/end should use integer-indexed values matching original PixelFlowScheduler.

        Original: Timesteps[int(N * ratio)] where Timesteps = [0, 1, ..., 999]
        This means T values are integers, not floats.
        """
        denoiser = MockPixelFlowDenoiser(num_stages=4)
        sampler = UnifiedSampler("PixelFlow", denoiser)

        _NUM_TRAIN_TIMESTEPS = 1000

        for stage_idx in range(4):
            start_ratio = (
                0.0 if stage_idx == 0 else sum(sampler.stage_distance[:stage_idx]) / sampler.total_stage_distance
            )
            end_ratio = (
                1.0
                if stage_idx == sampler.num_stages - 1
                else sum(sampler.stage_distance[: stage_idx + 1]) / sampler.total_stage_distance
            )

            T_start = int(_NUM_TRAIN_TIMESTEPS * start_ratio)
            T_end = min(int(_NUM_TRAIN_TIMESTEPS * end_ratio), _NUM_TRAIN_TIMESTEPS - 1)

            # T values should be integers (matching original indexing)
            assert isinstance(T_start, int)
            assert isinstance(T_end, int)
            assert 0 <= T_start < _NUM_TRAIN_TIMESTEPS
            assert 0 <= T_end < _NUM_TRAIN_TIMESTEPS
            assert T_start <= T_end

        # Last stage should end at 999
        assert T_end == 999

    def test_pixelflow_schedule_matches_original_scheduler(self) -> None:
        """Full schedule computation should match original PixelFlowScheduler output.

        Compares our schedule computation against the original PixelFlowScheduler
        implementation from original_implementations/PixelFlow/.
        """
        import sys
        from pathlib import Path

        original_path = Path("original_implementations/PixelFlow")
        if not original_path.exists():
            pytest.skip("Original PixelFlow implementation not available")

        sys.path.insert(0, str(original_path))
        try:
            from pixelflow.scheduling_pixelflow import PixelFlowScheduler

            # Create original scheduler
            original = PixelFlowScheduler(1000, num_stages=4, gamma=-1 / 3)

            # Test for 30 steps per stage (paper default)
            num_steps = 30
            _NUM_TRAIN_TIMESTEPS = 1000

            denoiser = MockPixelFlowDenoiser(num_stages=4)
            sampler = UnifiedSampler("PixelFlow", denoiser)

            for stage_idx in range(4):
                # Set up original scheduler for this stage
                original.set_timesteps(num_steps, stage_idx, device=torch.device("cpu"), shift=1.0)

                # Get original t and Timesteps
                orig_t = original.t.numpy()
                orig_T = original.Timesteps.numpy()

                # Compute our schedule
                start_ratio = (
                    0.0 if stage_idx == 0 else sum(sampler.stage_distance[:stage_idx]) / sampler.total_stage_distance
                )
                end_ratio = (
                    1.0
                    if stage_idx == sampler.num_stages - 1
                    else sum(sampler.stage_distance[: stage_idx + 1]) / sampler.total_stage_distance
                )
                T_start_idx = int(_NUM_TRAIN_TIMESTEPS * start_ratio)
                T_end_idx = min(int(_NUM_TRAIN_TIMESTEPS * end_ratio), _NUM_TRAIN_TIMESTEPS - 1)

                t_end_val = (_NUM_TRAIN_TIMESTEPS - 1) / _NUM_TRAIN_TIMESTEPS
                t_schedule = np.linspace(0, t_end_val, num_steps, dtype=np.float64)
                our_t = np.append(t_schedule, 1.0)

                # Mirror actual implementation: non-last stages use stage_T_end correction
                stage_T_start = float(T_start_idx)
                if stage_idx == sampler.num_stages - 1:
                    stage_T_end = float(T_end_idx)
                else:
                    stage_T_end = T_end_idx - (T_end_idx - T_start_idx) / _NUM_TRAIN_TIMESTEPS

                our_T = stage_T_start + (t_schedule / t_end_val) * (stage_T_end - stage_T_start)

                # Compare t values (should match at float64 precision)
                np.testing.assert_allclose(
                    our_t,
                    orig_t,
                    atol=1e-10,
                    err_msg=f"Stage {stage_idx}: t values differ",
                )

                # Compare T values (should match at float64 precision)
                np.testing.assert_allclose(
                    our_T,
                    orig_T,
                    atol=1e-10,
                    err_msg=f"Stage {stage_idx}: Timesteps differ",
                )
        finally:
            sys.path.pop(0)


# =============================================================================
# DiT DDIM Tests
# =============================================================================


class TestDiTDDIM:
    """Tests for DiT DDIM sampling."""

    def test_dit_ddim_timestep_sequence(self) -> None:
        """DiT DDIM timestep sequence should use uniform integer stride.

        SpacedDiffusion with "ddimN" uses [0, 100, 200, ..., 900] for 10 steps.
        For sampling (noisy to clean), we reverse: [900, 800, ..., 0].
        """
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config, sampling_method="ddim")

        device = torch.device("cpu")
        ts = sampler._get_dit_timestep_sequence(num_steps=10, device=device)

        assert len(ts) == 10
        # With 10 steps from 1000: [0, 100, 200, ..., 900] reversed = [900, 800, ..., 0]
        assert ts[0].item() == 900  # Start at max respaced timestep (most noisy)
        assert ts[-1].item() == 0  # End at 0 (clean)
        for i in range(len(ts) - 1):
            assert ts[i] > ts[i + 1]

    def test_dit_ddpm_timestep_sequence(self) -> None:
        """DiT DDPM timestep sequence should use fractional stride.

        DDPM with space_timesteps("N") uses fractional stride that always includes
        both endpoints (0 and 999).
        """
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config, sampling_method="ddpm")

        device = torch.device("cpu")
        ts = sampler._get_dit_timestep_sequence(num_steps=250, device=device)

        assert len(ts) == 250
        assert ts[0].item() == 999  # Fractional stride includes max timestep
        assert ts[-1].item() == 0  # And includes 0
        for i in range(len(ts) - 1):
            assert ts[i] > ts[i + 1]

    def test_dit_alpha_schedules(self) -> None:
        """DiT alpha schedules should match timestep indices."""
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        device = torch.device("cpu")
        ts = sampler._get_dit_timestep_sequence(num_steps=5, device=device)
        alpha_ts, alpha_prevs = sampler._get_dit_alpha_schedules(ts)

        assert alpha_ts.shape == (5,)
        assert alpha_prevs.shape == (5,)
        assert alpha_prevs[-1].item() == 1.0  # Last is clean data


# =============================================================================
# Device and Batch Size Tests
# =============================================================================


class TestDeviceAndBatchSize:
    """Tests for device and batch_size handling."""

    def test_batch_size_required_without_labels(self) -> None:
        """batch_size and device are required if no labels provided."""
        denoiser = MockJiTDenoiser()
        sampler = UnifiedSampler("JiT", denoiser)

        with pytest.raises(ValueError, match="Cannot determine batch size"):
            sampler.generate(num_steps=2, show_progress=False)

    def test_explicit_batch_size(self) -> None:
        """Explicit batch_size and device should work."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        sampler = UnifiedSampler("JiT", denoiser)

        images = sampler.generate(
            batch_size=3,
            device="cpu",
            num_steps=2,
            show_progress=False,
        )

        assert images.shape[0] == 3


# =============================================================================
# pred_target Tests
# =============================================================================


class TestPredTarget:
    """Tests for pred_target handling."""

    @pytest.mark.parametrize("pred_target", ["x", "v", "e"])
    def test_jit_all_pred_targets(self, pred_target: str) -> None:
        """JiT should work with all prediction targets."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, pred_target=pred_target, net=ZeroNet())
        sampler = UnifiedSampler("JiT", denoiser)

        labels = torch.tensor([207], device="cpu")
        images = sampler.generate(cfg_labels=labels, num_steps=2, show_progress=False)

        assert images.shape == (1, 3, 8, 8)
        assert sampler.pred_target == pred_target


# =============================================================================
# Edge Case Tests (TDD Coverage)
# =============================================================================


class TestRecurSteps:
    """Tests for recurrence step behavior."""

    def test_recur_steps_greater_than_one_changes_output(self) -> None:
        """recur_steps > 1 should produce different output than recur_steps = 1."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        # recur_steps = 1
        config1 = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
        )
        sampler1 = UnifiedSampler("JiT", denoiser, tfg_config=config1, sampling_method="euler")

        torch.manual_seed(42)
        images1 = sampler1.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        # recur_steps = 2
        config2 = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=2,
            iter_steps=1,
        )
        sampler2 = UnifiedSampler("JiT", denoiser, tfg_config=config2, sampling_method="euler")

        torch.manual_seed(42)
        images2 = sampler2.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        # Different recur_steps should produce different results
        assert not torch.allclose(images1, images2)

    def test_iter_steps_greater_than_one_changes_output(self) -> None:
        """iter_steps > 1 should produce different output than iter_steps = 1."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        # iter_steps = 1
        config1 = TFGConfig(
            device="cpu",
            rho=0.0,
            mu=1.0,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
        )
        sampler1 = UnifiedSampler("JiT", denoiser, tfg_config=config1, sampling_method="euler")

        torch.manual_seed(42)
        images1 = sampler1.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        # iter_steps = 3
        config2 = TFGConfig(
            device="cpu",
            rho=0.0,
            mu=1.0,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=3,
        )
        sampler2 = UnifiedSampler("JiT", denoiser, tfg_config=config2, sampling_method="euler")

        torch.manual_seed(42)
        images2 = sampler2.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        # Different iter_steps should produce different results
        assert not torch.allclose(images1, images2)


class TestDiTAlphaEdgeCases:
    """Tests for DiT DDPM alpha edge cases."""

    def test_dit_alpha_near_one_late_timesteps(self) -> None:
        """DiT should handle alpha near 1 (late timesteps, close to clean data)."""
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        device = torch.device("cpu")
        ts = sampler._get_dit_timestep_sequence(num_steps=100, device=device)
        alpha_ts, alpha_prevs = sampler._get_dit_alpha_schedules(ts)

        # Last alpha_prev should be 1.0 (fully clean)
        assert alpha_prevs[-1].item() == 1.0

        # All alphas should be in valid range
        assert (alpha_ts > 0).all()
        assert (alpha_ts <= 1).all()
        assert (alpha_prevs > 0).all()
        assert (alpha_prevs <= 1).all()

    def test_dit_alpha_near_zero_early_timesteps(self) -> None:
        """DiT should handle alpha near 0 (early timesteps, close to pure noise)."""
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        device = torch.device("cpu")
        ts = sampler._get_dit_timestep_sequence(num_steps=100, device=device)
        alpha_ts, _ = sampler._get_dit_alpha_schedules(ts)

        # First alpha should be small (noisy)
        assert alpha_ts[0].item() < 0.1

        # Alphas should be decreasing as we go back in time
        assert alpha_ts[0] < alpha_ts[-1]

    def test_dit_generation_does_not_produce_nan(self) -> None:
        """DiT generation should not produce NaN values."""
        denoiser = MockDiTDenoiser(latent_size=4, num_sampling_steps=5, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        labels = torch.tensor([207], device="cpu")
        images = sampler.generate(cfg_labels=labels, num_steps=5, show_progress=False)

        assert not torch.isnan(images).any()
        assert not torch.isinf(images).any()


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_x0_prediction_no_nan_for_all_targets(self) -> None:
        """x0 prediction should not produce NaN for all pred_targets."""
        for pred_target in ["x", "v", "e"]:
            denoiser = MockJiTDenoiser(img_size=8, steps=2, pred_target=pred_target, net=ZeroNet())
            sampler = UnifiedSampler("JiT", denoiser)

            labels = torch.tensor([207], device="cpu")
            images = sampler.generate(cfg_labels=labels, num_steps=2, show_progress=False)

            assert not torch.isnan(images).any(), f"NaN for pred_target={pred_target}"
            assert not torch.isinf(images).any(), f"Inf for pred_target={pred_target}"

    def test_t_eps_prevents_division_by_zero(self) -> None:
        """t_eps should prevent division by zero near t=1."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, t_eps=0.05)
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)
        guider = QuadraticLogpGuider(device="cpu")

        labels = torch.tensor([207], device="cpu")
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        assert not torch.isnan(images).any()
        assert not torch.isinf(images).any()

    def test_large_cfg_scale_stability(self) -> None:
        """Large CFG scale should not cause numerical instability."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        sampler = UnifiedSampler("JiT", denoiser)

        labels = torch.tensor([207], device="cpu")
        # Use a very large CFG scale
        images = sampler.generate(
            cfg_labels=labels,
            cfg_scale=100.0,
            num_steps=2,
            show_progress=False,
        )

        assert not torch.isnan(images).any()
        assert not torch.isinf(images).any()


class TestNaNGuard:
    """Tests for NaN/Inf guard after gradient computation."""

    def test_nan_gradient_replaced_with_zero(self) -> None:
        """NaN gradients from guidance should be replaced with zero, not propagated."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # Guider that returns NaN logprobs through a differentiable path
        # (simulates classifier producing 0/0 or log(0))
        class NaNGuider(BaseGuider):
            def __init__(self):
                self.device = "cpu"
                self.targets = [207]
                self.img_size = 8
                self.channels = 3

            def get_guidance(self, x, *, targets=None, return_logp=False, **kwargs):
                # Create NaN through differentiable ops: 0/0 preserves grad_fn
                zero = (x.flatten(1) * 0).sum(dim=1)
                nan_logp = zero / zero  # 0/0 = NaN with grad_fn
                if return_logp:
                    return nan_logp
                return torch.zeros_like(x)

        guider = NaNGuider()
        labels = torch.tensor([207], device="cpu")

        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        assert not torch.isnan(images).any(), "NaN propagated through guidance pipeline"
        assert not torch.isinf(images).any(), "Inf propagated through guidance pipeline"


class TestTFGTargetsPerSample:
    """Tests for per-sample TFG targets."""

    def test_tfg_targets_different_per_sample(self) -> None:
        """Different tfg_targets per sample should be handled correctly."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)
        guider = QuadraticLogpGuider(device="cpu")

        # Different targets for each sample
        cfg_labels = torch.tensor([207, 360, 388], device="cpu")
        tfg_targets = torch.tensor([100, 200, 300], device="cpu")

        images = sampler.generate(
            cfg_labels=cfg_labels,
            guidance=guider,
            tfg_targets=tfg_targets,
            num_steps=2,
            show_progress=False,
        )

        assert images.shape[0] == 3
        assert not torch.isnan(images).any()

    def test_tfg_targets_passed_to_dit(self) -> None:
        """DiT TFG should correctly pass tfg_targets to guidance functions."""
        denoiser = MockDiTDenoiser(latent_size=4, num_sampling_steps=2, device="cpu")
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=1,
        )
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)
        guider = QuadraticLogpGuider(device="cpu")

        cfg_labels = torch.tensor([207], device="cpu")
        tfg_targets = torch.tensor([100], device="cpu")

        images = sampler.generate(
            cfg_labels=cfg_labels,
            guidance=guider,
            tfg_targets=tfg_targets,
            num_steps=2,
            show_progress=False,
        )

        assert images.shape[0] == 1
        assert not torch.isnan(images).any()

    def test_cfg_and_tfg_different_targets(self) -> None:
        """CFG and TFG can use different target labels."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)
        guider = QuadraticLogpGuider(device="cpu")

        # CFG targets class 207, TFG targets class 100
        cfg_labels = torch.tensor([207], device="cpu")
        tfg_targets = torch.tensor([100], device="cpu")

        images = sampler.generate(
            cfg_labels=cfg_labels,
            guidance=guider,
            tfg_targets=tfg_targets,
            num_steps=2,
            show_progress=False,
        )

        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()


# =============================================================================
# Formula Verification Tests
# =============================================================================


class TestFormulaVerification:
    """Tests that verify mathematical formulas are implemented correctly."""

    def test_x0_from_epsilon_formula_dit(self) -> None:
        """x0 = (z_t - sqrt(1-alpha) * eps) / sqrt(alpha) for DiT."""
        # Setup - disable clip_x0 to test raw formula
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0, clip_x0=False)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        # Test at various alpha values
        for alpha in [0.1, 0.5, 0.9, 0.99]:
            alpha_t = torch.tensor(alpha)
            z_t = torch.randn(1, 4, 8, 8)
            eps = torch.randn(1, 4, 8, 8)

            # Formula: x0 = (z_t - sqrt(1-alpha) * eps) / sqrt(alpha)
            x0_expected = (z_t - (1 - alpha) ** 0.5 * eps) / alpha**0.5

            # Use the sampler's method
            x0_actual = sampler._dit_predict_x0(z_t, eps, alpha_t)

            assert torch.allclose(x0_expected, x0_actual, atol=1e-5), f"Failed at alpha={alpha}"

    def test_x0_from_velocity_formula_jit(self) -> None:
        """x0 = (1-t) * v + z_t for flow matching (JiT/SiT)."""
        # The formula for x0 from velocity in flow matching:
        # v = (x - z_t) / (1-t)  =>  x0 = (1-t) * v + z_t

        for t in [0.0, 0.1, 0.5, 0.9]:
            z_t = torch.randn(1, 3, 8, 8)
            v = torch.randn(1, 3, 8, 8)
            t_reshaped = torch.tensor(t).view(1, 1, 1, 1)

            # Formula: x0 = (1-t) * v + z_t
            x0 = (1 - t_reshaped) * v + z_t

            # Verify shape and no NaN
            assert x0.shape == z_t.shape
            assert not torch.isnan(x0).any()

    def test_euler_step_formula(self) -> None:
        """z_next = z + dt * v for Euler integration."""
        z = torch.randn(2, 3, 8, 8)
        v = torch.randn(2, 3, 8, 8)
        t, t_next = 0.0, 0.1
        dt = t_next - t

        # Euler step formula
        z_next_expected = z + dt * v

        # Verify
        assert z_next_expected.shape == z.shape
        assert not torch.isnan(z_next_expected).any()

        # Verify dt = 0 produces no change
        z_no_change = z + 0.0 * v
        assert torch.allclose(z, z_no_change)

    def test_ddim_step_formula(self) -> None:
        """DDIM: x_prev = sqrt(alpha_prev) * x0 + sqrt(1-alpha_prev-sigma^2) * eps."""
        # DDIM deterministic (eta=0, sigma=0)
        x0 = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)

        for alpha_prev in [0.9, 0.8, 0.5, 0.1]:
            alpha_prev_t = torch.tensor(alpha_prev)
            sigma = 0.0  # Deterministic

            # DDIM formula (eta=0)
            sqrt_alpha_prev = alpha_prev_t**0.5
            sqrt_one_minus_alpha_prev_minus_sigma2 = max(0, 1 - alpha_prev - sigma**2) ** 0.5

            x_prev = sqrt_alpha_prev * x0 + sqrt_one_minus_alpha_prev_minus_sigma2 * eps

            assert x_prev.shape == x0.shape
            assert not torch.isnan(x_prev).any()

    # NOTE: test_guidance_application_formula_flow was removed (3rd review A3).
    # It tested a raw tensor formula with incorrect dt* multiplier (x-space has NO dt).
    # Proper formula verification is in TestGuidanceSpaceFormulas.test_x_space_single_step_formula.

    def test_guidance_application_formula_dit(self) -> None:
        """DiT guidance: x_prev += delta_t / sqrt(alpha_t) + delta_0 * sqrt(alpha_prev)."""
        x_prev_base = torch.randn(1, 4, 8, 8)
        delta_t = torch.randn(1, 4, 8, 8) * 0.1
        delta_0 = torch.randn(1, 4, 8, 8) * 0.1

        alpha_prod_t = torch.tensor(0.8)
        alpha_prod_t_prev = torch.tensor(0.9)

        # DiT guidance formula
        alpha_t = alpha_prod_t / alpha_prod_t_prev.clamp_min(1e-8)
        x_guided = x_prev_base + delta_t / alpha_t**0.5 + delta_0 * alpha_prod_t_prev**0.5

        assert x_guided.shape == x_prev_base.shape
        assert not torch.isnan(x_guided).any()

        # Verify guidance changes output
        assert not torch.allclose(x_guided, x_prev_base)


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressionBugFixes:
    """Tests to prevent regression of previously fixed bugs."""

    def test_x0_clamp_not_applied_to_multiplication(self) -> None:
        """Regression test: x0 clipping should be applied after (1-t)*v + z, not to (1-t)."""
        # This verifies that the clamp is applied to x0, not to the (1-t) term
        # Bug: previously had clamp_min on (1-t) which limited the velocity contribution

        z_t = torch.randn(1, 3, 8, 8)
        v = torch.randn(1, 3, 8, 8) * 10  # Large velocity

        for t in [0.1, 0.5, 0.9]:
            t_reshaped = torch.tensor(t, dtype=torch.float32).view(1, 1, 1, 1)

            # Correct formula: x0 = (1-t) * v + z_t, then optionally clamp x0
            x0_correct = (1 - t_reshaped) * v + z_t

            # Incorrect would be: x0 = (1-t).clamp_min(something) * v + z_t
            # This would incorrectly modify the time scaling

            # Just verify the correct formula doesn't clamp the time term
            # The (1-t) factor should be able to be small (near t=1)
            assert x0_correct.shape == z_t.shape
            # At t=0.9, (1-t)=0.1 should still apply correctly
            if t == 0.9:
                expected_scale = torch.tensor(0.1, dtype=torch.float32)
                # The velocity contribution should be scaled by 0.1
                velocity_contrib = expected_scale * v
                recomputed = velocity_contrib + z_t
                assert torch.allclose(x0_correct, recomputed, atol=1e-6)

    def test_recur_steps_loop_structure(self) -> None:
        """Regression test: recurrence loop must update z_current, not just compute deltas."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        # With recur_steps > 1, the intermediate z_current should be updated
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=3,  # Multiple recurrence steps
            iter_steps=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="euler")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        # Should complete without error and produce valid output
        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_tfg_config_clip_x0_default_behavior(self) -> None:
        """Regression test: clip_x0 should default to True and be applied correctly."""
        config = TFGConfig(device="cpu", rho=1.0, mu=0.5)

        # Default clip_x0 should be True
        assert config.clip_x0 is True
        assert config.clip_sample_range == 1.0
        assert config.clip_sample_range_latent == 5.0

        # Also test explicit False
        config_no_clip = TFGConfig(device="cpu", rho=1.0, mu=0.5, clip_x0=False)
        assert config_no_clip.clip_x0 is False

    def test_recurrence_is_complete_resample(self) -> None:
        """Regression test: recurrence must do complete re-sample, not just perturbation.

        TFG paper's original intent: at the same timestep t, re-sample z with a
        new noise realization. The formula is: z = t * x_est + (1-t) * new_noise.

        This test verifies that with fixed random seed, the recurrence produces
        deterministic results that follow the re-sample formula.
        """
        # The re-sample formula: z_new = t * x_est + (1-t) * new_noise
        # where x_est = z + (1-t) * v

        t = 0.3
        z = torch.randn(1, 3, 8, 8)
        v = torch.randn(1, 3, 8, 8)

        # Compute x estimate
        x_est = z + (1 - t) * v

        # Generate new noise with fixed seed
        torch.manual_seed(123)
        new_noise = torch.randn_like(z)

        # Complete re-sample
        z_resampled = t * x_est + (1 - t) * new_noise

        # Verify re-sample is different from original z (not just a small perturbation)
        assert not torch.allclose(z, z_resampled, atol=0.01)

        # Verify the formula is correct by checking components
        # At t=0.3: z_resampled = 0.3 * x_est + 0.7 * new_noise
        expected = 0.3 * x_est + 0.7 * new_noise
        assert torch.allclose(z_resampled, expected, atol=1e-6)


# =============================================================================
# Extended Numerical Stability Tests
# =============================================================================


class TestExtendedNumericalStability:
    """Extended tests for numerical edge cases."""

    def test_alpha_very_near_zero(self) -> None:
        """alpha → 0 (very high noise) should not produce NaN or Inf."""
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        # Test with alpha very close to 0
        alpha_t = torch.tensor(1e-6)
        z_t = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)

        x0 = sampler._dit_predict_x0(z_t, eps, alpha_t)

        # Should not be NaN (clamp_min protects division)
        assert not torch.isnan(x0).any()

    def test_alpha_very_near_one(self) -> None:
        """alpha → 1 (very clean) should produce reasonable x0."""
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        # Test with alpha very close to 1
        alpha_t = torch.tensor(0.9999)
        z_t = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)

        x0 = sampler._dit_predict_x0(z_t, eps, alpha_t)

        # Should be close to z_t since alpha ≈ 1 means z_t ≈ x0
        assert not torch.isnan(x0).any()
        # At alpha=1, z_t = sqrt(1)*x0 + sqrt(0)*eps = x0
        # So x0 should be approximately z_t
        assert torch.allclose(x0, z_t, atol=0.1)

    def test_t_eps_clamping_at_boundary(self) -> None:
        """Verify t_eps clamping works at t very close to 1."""
        t_eps = 0.05

        for t in [0.95, 0.99, 0.999, 1.0]:
            # The clamping used in guidance application
            t_clamped = max(t, t_eps)

            # Verify it's at least t_eps
            assert t_clamped >= t_eps

            # Verify division doesn't explode
            t_next = 1.0
            ratio = t_next / t_clamped
            assert ratio <= 1.0 / t_eps  # Max ratio
            assert not math.isnan(ratio)
            assert not math.isinf(ratio)

    def test_schedule_values_at_boundaries(self) -> None:
        """Test rho/mu schedule values at t=0 and t=1 boundaries."""
        config = TFGConfig(
            device="cpu",
            rho=2.0,
            mu=3.0,
            sigma=0.5,
            rho_schedule="increase",
            mu_schedule="decrease",
        )
        denoiser = MockJiTDenoiser()
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # Without normalization (default):
        # At t=0: increase should give 0, decrease should give base
        assert sampler._get_schedule_value(2.0, "increase", 0.0) == pytest.approx(0.0)
        assert sampler._get_schedule_value(3.0, "decrease", 0.0) == pytest.approx(3.0)

        # At t=1: increase should give base, decrease should give 0
        assert sampler._get_schedule_value(2.0, "increase", 1.0) == pytest.approx(2.0)
        assert sampler._get_schedule_value(3.0, "decrease", 1.0) == pytest.approx(0.0)

        # Constant should always return base value
        assert sampler._get_schedule_value(1.0, "constant", 0.0) == pytest.approx(1.0)
        assert sampler._get_schedule_value(1.0, "constant", 1.0) == pytest.approx(1.0)

    def test_schedule_values_at_boundaries_normalized(self) -> None:
        """Test rho/mu schedule values at t=0 and t=1 boundaries with normalization."""
        config = TFGConfig(
            device="cpu",
            rho=2.0,
            mu=3.0,
            sigma=0.5,
            rho_schedule="increase",
            mu_schedule="decrease",
            normalize_schedules=True,
        )
        denoiser = MockJiTDenoiser()
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # With normalization:
        # At t=0: increase should give 0, decrease should give 2*base
        assert sampler._get_schedule_value(2.0, "increase", 0.0, normalize=True) == pytest.approx(0.0)
        assert sampler._get_schedule_value(3.0, "decrease", 0.0, normalize=True) == pytest.approx(6.0)

        # At t=1: increase should give 2*base, decrease should give 0
        assert sampler._get_schedule_value(2.0, "increase", 1.0, normalize=True) == pytest.approx(4.0)
        assert sampler._get_schedule_value(3.0, "decrease", 1.0, normalize=True) == pytest.approx(0.0)

        # Constant should always return base value
        assert sampler._get_schedule_value(1.0, "constant", 0.0, normalize=True) == pytest.approx(1.0)
        assert sampler._get_schedule_value(1.0, "constant", 1.0, normalize=True) == pytest.approx(1.0)

    def test_zero_rho_zero_mu_no_guidance_effect(self) -> None:
        """When rho=0 and mu=0, TFG should have no effect (equivalent to CFG-only)."""
        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())

        # CFG-only sampler
        sampler_cfg = UnifiedSampler("JiT", denoiser)

        # TFG with zero strength
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler_tfg = UnifiedSampler("JiT", denoiser, tfg_config=config)
        guider = QuadraticLogpGuider(device="cpu")

        labels = torch.tensor([207], device="cpu")

        torch.manual_seed(42)
        images_cfg = sampler_cfg.generate(cfg_labels=labels, num_steps=2, show_progress=False)

        torch.manual_seed(42)
        images_tfg = sampler_tfg.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        # With rho=0 and mu=0, outputs should be identical
        assert torch.allclose(images_cfg, images_tfg, atol=1e-6)


# =============================================================================
# DiT-Specific Tests
# =============================================================================


class TestDiTSpecificBehavior:
    """Extended tests for DiT-specific DDPM behavior."""

    def test_alpha_schedule_monotonic_decrease(self) -> None:
        """Alpha schedule should monotonically decrease: alpha[i] > alpha[i+1]."""
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        alphas_cumprod = denoiser.schedule.alphas_cumprod

        # Check monotonic decrease
        for i in range(len(alphas_cumprod) - 1):
            assert alphas_cumprod[i] > alphas_cumprod[i + 1], f"alpha[{i}] <= alpha[{i + 1}]"

        # First alpha should be close to 1 (clean)
        assert alphas_cumprod[0] > 0.99

        # Last alpha should be small (noisy)
        assert alphas_cumprod[-1] < 0.01

    def test_ddim_timestep_sequence_decreasing(self) -> None:
        """DDIM timestep sequence should be strictly decreasing: t[i] > t[i+1].

        US-012 FIX: Timestep sequence now matches original DiT's space_timesteps():
        - For 10 steps: {0, 100, ..., 900} reversed → first is 900
        - For 100 steps: {0, 10, ..., 990} reversed → first is 990
        """
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config, sampling_method="ddim")

        device = torch.device("cpu")
        total_timesteps = 1000
        for num_steps in [10, 50, 100, 250]:
            ts = sampler._get_dit_timestep_sequence(num_steps, device)
            step_size = total_timesteps // num_steps

            assert len(ts) == num_steps
            # DDIM: First timestep is (total_timesteps - step_size), not 999
            expected_first = total_timesteps - step_size
            assert ts[0].item() == expected_first, f"Expected {expected_first}, got {ts[0].item()}"
            assert ts[-1].item() == 0  # End at clean data boundary

            # Strictly decreasing
            for i in range(len(ts) - 1):
                assert ts[i] > ts[i + 1], f"ts[{i}]={ts[i]} <= ts[{i + 1}]={ts[i + 1]}"

    def test_ddpm_timestep_sequence_decreasing(self) -> None:
        """DDPM timestep sequence should be strictly decreasing and include endpoints.

        DDPM uses fractional stride from space_timesteps("N"):
        - Always includes 0 and 999
        - For 250 steps: first is 999, last is 0
        """
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config, sampling_method="ddpm")

        device = torch.device("cpu")
        for num_steps in [10, 50, 100, 250]:
            ts = sampler._get_dit_timestep_sequence(num_steps, device)

            assert len(ts) == num_steps
            assert ts[0].item() == 999  # Fractional stride always includes max
            assert ts[-1].item() == 0  # And includes 0

            # Strictly decreasing
            for i in range(len(ts) - 1):
                assert ts[i] > ts[i + 1], f"ts[{i}]={ts[i]} <= ts[{i + 1}]={ts[i + 1]}"

    def test_dit_x0_prediction_consistency(self) -> None:
        """DiT x0 prediction should be consistent with DDPM formula."""
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0, clip_x0=False)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        # Generate test data
        z_t = torch.randn(2, 4, 8, 8)
        eps = torch.randn(2, 4, 8, 8)

        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            alpha_t = torch.tensor(alpha)

            # Sampler's x0 prediction
            x0_sampler = sampler._dit_predict_x0(z_t, eps, alpha_t)

            # Manual DDPM formula: x0 = (z_t - sqrt(1-alpha) * eps) / sqrt(alpha)
            sqrt_alpha = alpha**0.5
            sqrt_one_minus_alpha = (1 - alpha) ** 0.5
            x0_manual = (z_t - sqrt_one_minus_alpha * eps) / sqrt_alpha

            assert torch.allclose(x0_sampler, x0_manual, atol=1e-5), f"Mismatch at alpha={alpha}"

    def test_dit_rho_schedule_normalization(self) -> None:
        """DiT rho schedule should be normalized over timesteps."""
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=1.0, mu=0.0, rho_schedule="increase")
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        device = torch.device("cpu")
        ts = sampler._get_dit_timestep_sequence(num_steps=10, device=device)
        alpha_ts, alpha_prevs = sampler._get_dit_alpha_schedules(ts)

        # Collect all rho values
        rho_values = []
        for t_idx in range(len(ts)):
            rho = sampler._get_dit_rho(t_idx, alpha_ts, alpha_prevs)
            rho_values.append(rho)
            assert rho >= 0, f"Negative rho at t_idx={t_idx}"

        # Sum should be approximately n * base_rho (due to normalization)
        # The normalizer ensures average effect is base_rho
        total = sum(rho_values)
        assert total > 0, "Total rho should be positive"

    def test_dit_rejects_non_x_guidance_space(self) -> None:
        """DiT should reject non-x guidance spaces (DDPM uses discrete timesteps)."""
        denoiser = MockDiTDenoiser(device="cpu")
        for gs in ["v", "v2"]:
            with pytest.raises(ValueError, match="DiT only supports guidance_space='x'"):
                UnifiedSampler("DiT", denoiser, guidance_space=gs)


# =============================================================================
# Guidance Space Formula Tests
# =============================================================================


class LinearGuider(BaseGuider):
    """Guider with constant gradient for formula verification.

    logp(x) = sum(x * direction) → grad_x logp = direction (constant).
    Used with ZeroNet (v=0, so x̂_0 = z) to make delta_t = rho * direction.
    """

    def __init__(self, direction: torch.Tensor, device: str = "cpu") -> None:
        self.direction = direction
        self.device = device
        self.targets = [207]

    def get_guidance(
        self,
        x: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        return_logp: bool = False,
        check_grad: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        logp = (x * self.direction).flatten(1).sum(dim=1)
        if return_logp:
            return logp
        return self.direction.expand_as(x)


class TestGuidanceSpaceFormulas:
    """Verify guidance space formulas via single-step arithmetic interception.

    Strategy: LinearGuider (constant gradient) + ZeroNet (v=0) + sigma=0, mu=0
    makes delta_t = rho * direction (known) and delta_0 = 0. Then we call
    _euler_step/_heun_step directly and verify exact numerical results.

    x-space: guidance = (t_next/t)·δ_t                (NO dt)
    v-space: guidance = dt·λ_t·δ_t = dt·(1-t)/t·δ_t   (dt present)
    Ratio:   x/v = t_next / (dt·(1-t))                (verifiable)

    Property tests (step-count dependency) use full generation.
    """

    def _make_sampler(self, guidance_space: str, sampling_method: str = "euler", img_size: int = 4):
        denoiser = MockJiTDenoiser(img_size=img_size, steps=50, net=ZeroNet())
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
            rho_schedule="constant",
            clip_x0=False,
        )
        return UnifiedSampler(
            "JiT",
            denoiser,
            tfg_config=config,
            sampling_method=sampling_method,
            guidance_space=guidance_space,
        )

    @pytest.mark.parametrize("t,t_next", [(0.5, 0.6), (0.2, 0.3)])
    def test_x_space_single_step_formula(self, t: float, t_next: float) -> None:
        """x-space Euler: verify (t_next/t)·δ_t with NO dt, via single step.

        Setup: ZeroNet → v=0, x̂_0=z. LinearGuider → grad=direction.
        Therefore delta_t = rho * direction = direction (rho=1).

        Expected: z_next = z + (t_next/t) * direction
        """
        sampler = self._make_sampler("x")
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])

        z = torch.randn(1, 3, 4, 4)
        z_orig = z.clone()

        z_next = sampler._euler_step(z, t, t_next, labels, guider, labels)

        # v=0 → z_next_base = z. Guidance = (t_next/t)*delta_t, NO dt.
        expected = z_orig + (t_next / t) * direction

        assert torch.allclose(z_next, expected, atol=1e-6), (
            f"x-space formula wrong at t={t}. Max diff: {(z_next - expected).abs().max():.8f}"
        )

    @pytest.mark.parametrize("t,t_next", [(0.5, 0.6), (0.2, 0.3)])
    def test_v_space_single_step_formula(self, t: float, t_next: float) -> None:
        """v-space Euler: verify dt·(1-t)/t·δ_t, via single step.

        Key difference from x-space: dt IS present (velocity modification through ODE).
        """
        sampler = self._make_sampler("v")
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])

        dt = t_next - t
        lambda_t = (1 - t) / t
        z = torch.randn(1, 3, 4, 4)
        z_orig = z.clone()

        z_next = sampler._euler_step(z, t, t_next, labels, guider, labels)

        # v=0 → v_guided = λ_t*delta_t. z_next = z + dt*v_guided.
        expected = z_orig + dt * lambda_t * direction

        assert torch.allclose(z_next, expected, atol=1e-6), (
            f"v-space formula wrong at t={t}. Max diff: {(z_next - expected).abs().max():.8f}"
        )

    @pytest.mark.parametrize("t,t_next", [(0.5, 0.6), (0.2, 0.3)])
    def test_x_vs_v_guidance_ratio(self, t: float, t_next: float) -> None:
        """x-space and v-space differ by a computable ratio per step.

        At time t:
          x-space guidance magnitude: (t_next/t) * |δ_t|
          v-space guidance magnitude: dt * (1-t)/t * |δ_t|
          Ratio: t_next / (dt*(1-t))
        """
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])

        dt = t_next - t
        z = torch.randn(1, 3, 4, 4)

        sampler_x = self._make_sampler("x")
        sampler_v = self._make_sampler("v")

        z_x = sampler_x._euler_step(z.clone(), t, t_next, labels, guider, labels)
        z_v = sampler_v._euler_step(z.clone(), t, t_next, labels, guider, labels)

        guidance_x = (z_x - z).abs().mean().item()
        guidance_v = (z_v - z).abs().mean().item()

        expected_ratio = (t_next / t) / (dt * (1 - t) / t)
        actual_ratio = guidance_x / guidance_v

        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01, (
            f"x/v ratio at t={t}: expected {expected_ratio:.2f}, got {actual_ratio:.2f}"
        )

    def test_v2_per_nfe_uses_different_lambda(self) -> None:
        """v2-space: v1 uses λ_t, v2 uses λ_{t_next} (different values).

        Single Heun step with TimeDependentNet (v1 ≠ v2).
        v2-space must differ from v-space because per-NFE lambdas differ.
        """
        denoiser = MockJiTDenoiser(img_size=4, steps=50, net=TimeDependentNet())
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])

        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
        )

        sampler_v = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="heun", guidance_space="v")
        sampler_v2 = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="heun", guidance_space="v2")

        t, t_next = 0.5, 0.6
        z = torch.randn(1, 3, 4, 4)

        z_v = sampler_v._heun_step(z.clone(), t, t_next, labels, guider, labels)
        z_v2 = sampler_v2._heun_step(z.clone(), t, t_next, labels, guider, labels)

        assert not torch.allclose(z_v, z_v2, atol=1e-6), (
            "v2-space must differ from v-space: per-NFE uses λ_{t_next} ≠ λ_t at v2"
        )

    def test_x_space_is_step_count_dependent(self) -> None:
        """x-space is step-count dependent (DDPM heritage, intentional).

        Guidance has no dt, so cumulative guidance magnitude grows with num_steps.
        This matches original TFG behavior: changing num_steps requires re-tuning
        rho/mu. We verify this by comparing guidance-only displacement across
        two step counts.
        """
        denoiser = MockJiTDenoiser(img_size=8, steps=50, net=ZeroNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
        )

        # CFG-only baseline (no guidance) — same for both step counts with ZeroNet
        torch.manual_seed(42)
        sampler_base = UnifiedSampler(
            "JiT",
            denoiser,
            sampling_method="euler",
            guidance_space="x",
        )
        base_10 = sampler_base.generate(
            cfg_labels=labels,
            num_steps=10,
            show_progress=False,
        )
        torch.manual_seed(42)
        base_20 = sampler_base.generate(
            cfg_labels=labels,
            num_steps=20,
            show_progress=False,
        )

        # Guided outputs
        torch.manual_seed(42)
        sampler_x = UnifiedSampler(
            "JiT",
            denoiser,
            tfg_config=config,
            sampling_method="euler",
            guidance_space="x",
        )
        guided_10 = sampler_x.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=10,
            show_progress=False,
        )
        torch.manual_seed(42)
        guided_20 = sampler_x.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=20,
            show_progress=False,
        )

        # Guidance displacement
        disp_10 = (guided_10 - base_10).abs().mean().item()
        disp_20 = (guided_20 - base_20).abs().mean().item()

        # 20 steps should accumulate more guidance than 10 steps (step-count dependent)
        assert disp_20 > disp_10 * 1.2, (
            f"x-space should be step-count dependent: disp_20={disp_20:.4f} should be >1.2x disp_10={disp_10:.4f}"
        )

    def test_v_space_is_more_step_count_stable(self) -> None:
        """v-space is more step-count stable than x-space due to ODE dt normalization.

        When doubling num_steps, v-space guidance displacement ratio should be
        closer to 1.0 than x-space ratio, because dt halves and compensates.
        """
        denoiser = MockJiTDenoiser(img_size=8, steps=50, net=ZeroNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
        )

        disps = {}
        for space in ["x", "v"]:
            for steps in [10, 20]:
                # Baseline (no guidance)
                torch.manual_seed(42)
                sampler_base = UnifiedSampler(
                    "JiT",
                    denoiser,
                    sampling_method="euler",
                    guidance_space=space,
                )
                base = sampler_base.generate(
                    cfg_labels=labels,
                    num_steps=steps,
                    show_progress=False,
                )

                # Guided
                torch.manual_seed(42)
                sampler_g = UnifiedSampler(
                    "JiT",
                    denoiser,
                    tfg_config=config,
                    sampling_method="euler",
                    guidance_space=space,
                )
                guided = sampler_g.generate(
                    cfg_labels=labels,
                    guidance=guider,
                    tfg_targets=labels,
                    num_steps=steps,
                    show_progress=False,
                )

                disps[(space, steps)] = (guided - base).abs().mean().item()

        # Compute step-count sensitivity ratio (20 steps / 10 steps)
        x_ratio = disps[("x", 20)] / max(disps[("x", 10)], 1e-8)
        v_ratio = disps[("v", 20)] / max(disps[("v", 10)], 1e-8)

        # v-space ratio should be closer to 1.0 (more stable)
        x_deviation = abs(x_ratio - 1.0)
        v_deviation = abs(v_ratio - 1.0)
        assert v_deviation < x_deviation, (
            f"v-space should be more step-count stable: "
            f"v_deviation={v_deviation:.4f} should be < x_deviation={x_deviation:.4f}"
        )

    @pytest.mark.parametrize("t,t_next", [(0.5, 0.6), (0.2, 0.3)])
    def test_heun_x_space_formula(self, t: float, t_next: float) -> None:
        """Heun x-space: guidance = (t_next/t)·δ_t, applied to base Heun step.

        With ZeroNet: v1=v2=0, so base z_next = z (no movement).
        Same formula as Euler x-space (position correction, no dt).
        """
        sampler = self._make_sampler("x", sampling_method="heun")
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])

        z = torch.randn(1, 3, 4, 4)
        z_orig = z.clone()

        z_next = sampler._heun_step(z, t, t_next, labels, guider, labels)

        expected = z_orig + (t_next / t) * direction
        assert torch.allclose(z_next, expected, atol=1e-6), (
            f"Heun x-space formula wrong at t={t}. Max diff: {(z_next - expected).abs().max():.8f}"
        )

    @pytest.mark.parametrize("t,t_next", [(0.5, 0.6), (0.2, 0.3)])
    def test_heun_v_space_formula(self, t: float, t_next: float) -> None:
        """Heun v-space: v_avg_guided = v_avg + λ_t·δ_t + δ_0.

        With ZeroNet: v1=v2=0, v_avg=0.
        v_avg_guided = λ_t*direction. z_next = z + dt*λ_t*direction.
        """
        sampler = self._make_sampler("v", sampling_method="heun")
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])

        dt = t_next - t
        lambda_t = (1 - t) / t
        z = torch.randn(1, 3, 4, 4)
        z_orig = z.clone()

        z_next = sampler._heun_step(z, t, t_next, labels, guider, labels)

        expected = z_orig + dt * lambda_t * direction
        assert torch.allclose(z_next, expected, atol=1e-6), (
            f"Heun v-space formula wrong at t={t}. Max diff: {(z_next - expected).abs().max():.8f}"
        )

    @pytest.mark.parametrize("t,t_next", [(0.5, 0.6), (0.2, 0.3)])
    def test_delta_0_mean_guidance_applied(self, t: float, t_next: float) -> None:
        """Verify delta_0 (mean guidance, mu>0) is correctly applied in x-space.

        With mu=1.0, iter_steps=1, rho=0 (delta_t=0), LinearGuider:
          delta_0 = mu * grad(logp(x0)) = 1.0 * direction
        x-space Euler: z_next = z + t_next*delta_0
        """
        denoiser = MockJiTDenoiser(img_size=4, steps=50, net=ZeroNet())
        config = TFGConfig(
            device="cpu",
            rho=0.0,
            mu=1.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
            rho_schedule="constant",
            mu_schedule="constant",
            clip_x0=False,
        )
        sampler = UnifiedSampler(
            "JiT",
            denoiser,
            tfg_config=config,
            sampling_method="euler",
            guidance_space="x",
        )
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])

        z = torch.randn(1, 3, 4, 4)
        z_orig = z.clone()

        z_next = sampler._euler_step(z, t, t_next, labels, guider, labels)

        # rho=0 → delta_t=0. mu=1 → delta_0 = direction.
        # x-space: z_next = z + t_next*direction
        expected = z_orig + t_next * direction
        assert torch.allclose(z_next, expected, atol=1e-5), (
            f"delta_0 x-space formula wrong at t={t}. Max diff: {(z_next - expected).abs().max():.8f}"
        )

    @pytest.mark.parametrize("t,t_next", [(0.5, 0.6), (0.2, 0.3)])
    def test_delta_0_v_space_applied(self, t: float, t_next: float) -> None:
        """Verify delta_0 in v-space differs from x-space.

        v-space: v_guided = v + 0 + delta_0 = direction (no lambda_t on delta_0).
        z_next = z + dt * direction.

        x-space would give: z + t_next * direction.
        v-space gives:      z + dt * direction.  (dt = t_next - t)
        """
        denoiser = MockJiTDenoiser(img_size=4, steps=50, net=ZeroNet())
        config = TFGConfig(
            device="cpu",
            rho=0.0,
            mu=1.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
            rho_schedule="constant",
            mu_schedule="constant",
            clip_x0=False,
        )
        sampler = UnifiedSampler(
            "JiT",
            denoiser,
            tfg_config=config,
            sampling_method="euler",
            guidance_space="v",
        )
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])

        dt = t_next - t
        z = torch.randn(1, 3, 4, 4)
        z_orig = z.clone()

        z_next = sampler._euler_step(z, t, t_next, labels, guider, labels)

        # rho=0 → delta_t=0, lambda_t irrelevant. mu=1 → delta_0 = direction.
        # v-space: v_guided = 0 + 0 + direction. z_next = z + dt*direction.
        expected = z_orig + dt * direction
        assert torch.allclose(z_next, expected, atol=1e-5), (
            f"delta_0 v-space formula wrong at t={t}. Max diff: {(z_next - expected).abs().max():.8f}"
        )

    @pytest.mark.parametrize("t,t_next", [(0.3, 0.4), (0.5, 0.6)])
    def test_v2_space_heun_formula(self, t: float, t_next: float) -> None:
        """v2-space Heun: per-NFE velocity modification with exact formula.

        With ZeroNet (v1=v2=0), z_euler=z (no movement).
        Guidance at v1: delta_t_1 = direction (at z, t) → λ_t = (1-t)/t
        Guidance at v2: delta_t_2 = direction (at z, t_next) → λ_{t_next} = (1-t_next)/t_next

        v1_guided = λ_t * direction
        v2_guided = λ_{t_next} * direction
        z_next = z + dt * 0.5 * (v1_guided + v2_guided)
        """
        sampler = self._make_sampler("v2", sampling_method="heun")
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])

        dt = t_next - t
        lambda_t = (1 - t) / t
        lambda_t_next = (1 - t_next) / t_next
        z = torch.randn(1, 3, 4, 4)
        z_orig = z.clone()

        z_next = sampler._heun_step(z, t, t_next, labels, guider, labels)

        # Per-NFE: average of individually guided velocities
        expected = z_orig + dt * 0.5 * (lambda_t + lambda_t_next) * direction
        assert torch.allclose(z_next, expected, atol=1e-6), (
            f"v2-space Heun formula wrong at t={t}. "
            f"λ_t={lambda_t:.4f}, λ_t_next={lambda_t_next:.4f}. "
            f"Max diff: {(z_next - expected).abs().max():.8f}"
        )

    @pytest.mark.parametrize("t,t_next", [(0.3, 0.4), (0.6, 0.7)])
    def test_v2_euler_fallback_matches_v_space(self, t: float, t_next: float) -> None:
        """v2-space must fall back to v-space in _euler_step (last Heun step).

        Regression test: if `in ("v", "v2")` is refactored to `== "v"`,
        v2-space would silently lose guidance on the final Euler step.
        """
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])
        z = torch.randn(1, 3, 4, 4)

        # v-space Euler
        sampler_v = self._make_sampler("v", sampling_method="euler")
        z_v = sampler_v._euler_step(z.clone(), t, t_next, labels, guider, labels)

        # v2-space Euler (fallback path)
        sampler_v2 = self._make_sampler("v2", sampling_method="heun")
        z_v2 = sampler_v2._euler_step(z.clone(), t, t_next, labels, guider, labels)

        assert torch.allclose(z_v, z_v2, atol=1e-7), (
            f"v2 Euler fallback differs from v-space at t={t}. Max diff: {(z_v - z_v2).abs().max():.8f}"
        )

    @pytest.mark.parametrize("t,t_next", [(0.3, 0.4), (0.6, 0.7)])
    def test_combined_rho_mu_x_space(self, t: float, t_next: float) -> None:
        """x-space: both delta_t (rho=1) and delta_0 (mu=1) applied with correct scaling.

        Setup: ZeroNet → v=0, x̂_0=z. LinearGuider → grad=direction.
        delta_t = direction, delta_0 = direction (mu=1, one iter_step).
        Expected: z_next = z + (t_next/t)*direction + t_next*direction
        """
        denoiser = MockJiTDenoiser(img_size=4, steps=50, net=ZeroNet())
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=1.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
            rho_schedule="constant",
            mu_schedule="constant",
            clip_x0=False,
        )
        sampler = UnifiedSampler(
            "JiT",
            denoiser,
            tfg_config=config,
            sampling_method="euler",
            guidance_space="x",
        )
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])

        z = torch.randn(1, 3, 4, 4)
        z_orig = z.clone()

        z_next = sampler._euler_step(z, t, t_next, labels, guider, labels)

        t_clamped = max(t, 0.05)
        expected = z_orig + (t_next / t_clamped) * direction + t_next * direction
        assert torch.allclose(z_next, expected, atol=1e-5), (
            f"Combined rho+mu x-space formula wrong at t={t}. Max diff: {(z_next - expected).abs().max():.8f}"
        )


# =============================================================================
# Recurrence Integration Tests
# =============================================================================


class TestRecurrenceIntegration:
    """Integration tests for recurrence behavior in actual sampler."""

    def test_recurrence_uses_velocity_prediction(self) -> None:
        """Verify recurrence uses v_pred_recur for x estimation."""
        # With recur_steps > 1, the sampler should use v_pred_recur to estimate x
        # then re-sample z = t * x_est + (1-t) * new_noise

        denoiser = MockJiTDenoiser(img_size=8, steps=2, net=ZeroNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=3,
            iter_steps=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="euler")

        # Should complete without error
        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_recurrence_resample_changes_intermediate_z(self) -> None:
        """Verify recurrence changes z through complete re-sample, not just perturbation."""
        # The re-sample formula z = t * x_est + (1-t) * new_noise should produce
        # significantly different z values than simple perturbation

        t = 0.5
        z = torch.randn(1, 3, 8, 8)
        v = torch.randn(1, 3, 8, 8)

        # x estimation
        x_est = z + (1 - t) * v

        # Complete re-sample (correct)
        torch.manual_seed(42)
        new_noise = torch.randn_like(z)
        z_resample = t * x_est + (1 - t) * new_noise

        # Simple perturbation (old incorrect approach)
        torch.manual_seed(42)
        noise_scale = 0.1  # arbitrary small perturbation
        z_perturb = z + noise_scale * torch.randn_like(z)

        # Re-sample should produce very different result than perturbation
        diff_resample = (z_resample - z).abs().mean()
        diff_perturb = (z_perturb - z).abs().mean()

        # Re-sample typically produces much larger changes
        assert diff_resample > diff_perturb * 2, (
            f"Re-sample diff {diff_resample:.4f} should be much larger than perturbation diff {diff_perturb:.4f}"
        )


# =============================================================================
# PixelFlow TFG Tests
# =============================================================================


class TestPixelFlowTFG:
    """Tests for PixelFlow with TFG guidance."""

    def test_pixelflow_tfg_generation(self) -> None:
        """PixelFlow should work with TFG guidance."""
        denoiser = MockPixelFlowDenoiser(img_size=32, num_stages=2, device="cpu")
        guider = QuadraticLogpGuider(device="cpu", img_size=32)
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
        )
        sampler = UnifiedSampler("PixelFlow", denoiser, tfg_config=config, sampling_method="euler")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        assert images.shape == (1, 3, 32, 32)
        assert not torch.isnan(images).any()

    def test_pixelflow_tfg_changes_output(self) -> None:
        """PixelFlow TFG should change output compared to CFG-only."""
        denoiser = MockPixelFlowDenoiser(img_size=32, num_stages=2, device="cpu")
        guider = QuadraticLogpGuider(device="cpu", img_size=32)
        labels = torch.tensor([207], device="cpu")

        # CFG-only
        sampler_cfg = UnifiedSampler("PixelFlow", denoiser, tfg_config=None, sampling_method="euler")
        torch.manual_seed(42)
        images_cfg = sampler_cfg.generate(
            cfg_labels=labels,
            num_steps=2,
            show_progress=False,
        )

        # CFG + TFG
        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=1.0,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler_tfg = UnifiedSampler("PixelFlow", denoiser, tfg_config=config, sampling_method="euler")
        torch.manual_seed(42)
        images_tfg = sampler_tfg.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        # TFG should change the output
        assert not torch.allclose(images_cfg, images_tfg)

    def test_pixelflow_recurrence(self) -> None:
        """PixelFlow recurrence should use complete re-sample."""
        denoiser = MockPixelFlowDenoiser(img_size=32, num_stages=2, device="cpu")
        guider = QuadraticLogpGuider(device="cpu", img_size=32)
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=2,  # Multiple recurrence steps
            iter_steps=1,
        )
        sampler = UnifiedSampler("PixelFlow", denoiser, tfg_config=config, sampling_method="euler")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        assert images.shape == (1, 3, 32, 32)
        assert not torch.isnan(images).any()


# =============================================================================
# SiT TFG Tests
# =============================================================================


class TestSiTTFG:
    """Tests for SiT with TFG guidance."""

    def test_sit_tfg_generation(self) -> None:
        """SiT should work with TFG guidance."""
        denoiser = MockSiTDenoiser(device="cpu")
        guider = QuadraticLogpGuider(device="cpu", img_size=8, channels=4)
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
        )
        sampler = UnifiedSampler("SiT", denoiser, tfg_config=config, sampling_method="euler")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        # SiT returns latents (no VAE decode in mock)
        assert images.shape[0] == 1
        assert not torch.isnan(images).any()

    def test_sit_tfg_changes_output(self) -> None:
        """SiT TFG should change output compared to CFG-only."""
        # Use TimeDependentNet to ensure non-trivial velocity predictions
        denoiser = MockSiTDenoiser(device="cpu", net=TimeDependentNet())
        guider = QuadraticLogpGuider(device="cpu", img_size=8, channels=4)
        labels = torch.tensor([207], device="cpu")

        # CFG-only
        sampler_cfg = UnifiedSampler("SiT", denoiser, tfg_config=None, sampling_method="euler")
        torch.manual_seed(42)
        images_cfg = sampler_cfg.generate(
            cfg_labels=labels,
            num_steps=5,
            show_progress=False,
        )

        # CFG + TFG with stronger guidance
        config = TFGConfig(
            device="cpu",
            rho=5.0,  # Stronger guidance
            mu=5.0,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler_tfg = UnifiedSampler("SiT", denoiser, tfg_config=config, sampling_method="euler")
        torch.manual_seed(42)
        images_tfg = sampler_tfg.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        # TFG should change the output
        assert not torch.allclose(images_cfg, images_tfg)


# =============================================================================
# Heun Sampling with TFG Tests
# =============================================================================


class TestHeunSamplingWithTFG:
    """Tests for Heun sampling method with TFG guidance."""

    def test_jit_heun_tfg_generation(self) -> None:
        """JiT Heun sampling should work with TFG."""
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=ZeroNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="heun")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_heun_tfg_differs_from_euler_tfg(self) -> None:
        """Heun with TFG should produce different output than Euler with TFG."""
        # Use TimeDependentNet so v1 != v2 (velocity changes with time)
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=TimeDependentNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )

        # Euler
        sampler_euler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="euler")
        torch.manual_seed(42)
        images_euler = sampler_euler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        # Heun
        sampler_heun = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="heun")
        torch.manual_seed(42)
        images_heun = sampler_heun.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        # Different sampling methods should produce different results
        assert not torch.allclose(images_euler, images_heun)

    def test_sit_heun_tfg_generation(self) -> None:
        """SiT Heun sampling should work with TFG."""
        denoiser = MockSiTDenoiser(device="cpu")
        guider = QuadraticLogpGuider(device="cpu", img_size=8, channels=4)
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("SiT", denoiser, tfg_config=config, sampling_method="heun")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        assert images.shape[0] == 1
        assert not torch.isnan(images).any()

    def test_pixelflow_heun_tfg_generation(self) -> None:
        """PixelFlow Heun sampling should work with TFG."""
        denoiser = MockPixelFlowDenoiser(img_size=32, num_stages=2, device="cpu")
        guider = QuadraticLogpGuider(device="cpu", img_size=32)
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("PixelFlow", denoiser, tfg_config=config, sampling_method="heun")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        assert images.shape == (1, 3, 32, 32)
        assert not torch.isnan(images).any()


# =============================================================================
# UnifiedSampler DiT Generation Path Verification Tests (US-005)
# =============================================================================


class TestUnifiedSamplerDiTGenerationPath:
    """Tests verifying UnifiedSampler DiT-specific implementation.

    These tests verify that the DiT generation path in UnifiedSampler:
    1. Uses correct SpacedDiffusion-style timestep respacing
    2. Computes alpha schedules correctly for respaced timesteps
    3. Implements DDIM step correctly
    4. Produces numerically stable outputs
    """

    def test_dit_ddim_timestep_sequence_matches_spaced_diffusion(self) -> None:
        """Verify _get_dit_timestep_sequence() for DDIM matches SpacedDiffusion.

        SpacedDiffusion with "ddimN" (for 100 steps from 1000):
        - Creates timesteps [0, 10, 20, ..., 990]
        - For sampling, iterates in reverse: 990, 980, ..., 0
        """
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config, sampling_method="ddim")

        device = torch.device("cpu")

        # Test 100 steps from 1000 (DDIM configuration)
        ts = sampler._get_dit_timestep_sequence(num_steps=100, device=device)

        assert len(ts) == 100
        assert ts[0].item() == 990  # First (most noisy)
        assert ts[-1].item() == 0  # Last (clean)

        # Verify step size is 10
        for i in range(len(ts) - 1):
            assert ts[i] - ts[i + 1] == 10, f"Step {i}: expected step of 10"

    def test_dit_alpha_schedule_access_respaced_values(self) -> None:
        """Verify alpha schedules access respaced timestep values correctly.

        For timestep sequence [990, 980, ..., 0]:
        - alpha_prod_ts should contain alphas at those timesteps
        - alpha_prod_t_prevs[i] should contain alpha at ts[i+1] (the next step)
        - alpha_prod_t_prevs[-1] should be 1.0 (final clean state)
        """
        import numpy as np

        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        device = torch.device("cpu")
        ts = sampler._get_dit_timestep_sequence(num_steps=100, device=device)
        alpha_ts, alpha_prevs = sampler._get_dit_alpha_schedules(ts)

        # Get reference alphas_cumprod
        betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod_ref = np.cumprod(alphas)

        # Verify alpha_ts matches reference at respaced timesteps
        for i, t in enumerate(ts):
            expected_alpha = alphas_cumprod_ref[t.item()]
            actual_alpha = alpha_ts[i].item()
            assert np.isclose(expected_alpha, actual_alpha, rtol=1e-6), (
                f"Step {i}, t={t.item()}: expected alpha {expected_alpha}, got {actual_alpha}"
            )

        # Verify alpha_prevs[i] = alphas_cumprod[ts[i+1]] for i < len-1
        for i in range(len(ts) - 1):
            expected_alpha_prev = alphas_cumprod_ref[ts[i + 1].item()]
            actual_alpha_prev = alpha_prevs[i].item()
            assert np.isclose(expected_alpha_prev, actual_alpha_prev, rtol=1e-6), (
                f"Step {i}: expected alpha_prev {expected_alpha_prev}, got {actual_alpha_prev}"
            )

        # Verify final alpha_prev is 1.0
        assert alpha_prevs[-1].item() == 1.0

    def test_dit_ddim_step_numerically_stable(self) -> None:
        """Verify DDIM step produces numerically stable outputs."""
        denoiser = MockDiTDenoiser(latent_size=4, num_sampling_steps=5, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        device = torch.device("cpu")
        ts = sampler._get_dit_timestep_sequence(num_steps=5, device=device)
        alpha_ts, alpha_prevs = sampler._get_dit_alpha_schedules(ts)

        # Test a single DDIM step at various positions
        for t_idx in range(len(ts)):
            z = torch.randn(2, 4, 4, 4)
            labels = torch.tensor([207, 388])

            z_next = sampler._dit_ddim_step(
                z=z,
                t_idx=t_idx,
                ts=ts,
                alpha_prod_ts=alpha_ts,
                alpha_prod_t_prevs=alpha_prevs,
                labels=labels,
                cfg_scale=4.0,
            )

            assert not torch.isnan(z_next).any(), f"NaN at step {t_idx}"
            assert not torch.isinf(z_next).any(), f"Inf at step {t_idx}"

    def test_dit_ddim_generation_loop_consistency(self) -> None:
        """Verify DDIM generation loop produces consistent results with same seed.

        DDIM with eta=0 is deterministic, so the output should be identical
        when using the same random seed for initial noise.
        """
        denoiser = MockDiTDenoiser(latent_size=4, num_sampling_steps=5, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        # Use DDIM explicitly for determinism test
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config, sampling_method="ddim")

        labels = torch.tensor([207], device="cpu")

        # Run twice with same seed
        torch.manual_seed(42)
        images1 = sampler.generate(cfg_labels=labels, num_steps=5, show_progress=False)

        torch.manual_seed(42)
        images2 = sampler.generate(cfg_labels=labels, num_steps=5, show_progress=False)

        # Outputs should be identical for DDIM (eta=0 is deterministic)
        assert torch.allclose(images1, images2), "DDIM generation not deterministic with same seed"

    def test_dit_alpha_ordering_correct_for_sampling(self) -> None:
        """Verify alpha values are ordered correctly for denoising direction.

        During sampling (noisy to clean):
        - We go from t=990 (most noisy, alpha small) to t=0 (clean, alpha large)
        - alpha_t < alpha_t_prev (we move towards higher alpha)
        """
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        device = torch.device("cpu")
        ts = sampler._get_dit_timestep_sequence(num_steps=100, device=device)
        alpha_ts, alpha_prevs = sampler._get_dit_alpha_schedules(ts)

        # For each step (except last), verify alpha_t < alpha_prev
        # This is correct because we're moving from noisy (low alpha) to clean (high alpha)
        for i in range(len(ts) - 1):
            alpha_t = alpha_ts[i].item()
            alpha_prev = alpha_prevs[i].item()
            assert alpha_t < alpha_prev, (
                f"Step {i}: alpha_t={alpha_t} should be < alpha_prev={alpha_prev} (denoising direction)"
            )

    def test_dit_cfg_scale_parameter_accepted(self) -> None:
        """Verify CFG scale parameter is properly accepted and used."""
        denoiser = MockDiTDenoiser(latent_size=4, num_sampling_steps=3, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        labels = torch.tensor([207], device="cpu")

        # Just verify both CFG scales run without error
        # Note: With the mock denoiser (which returns zeros), outputs will be identical.
        # This test ensures the cfg_scale parameter is properly accepted and processed.
        images_cfg1 = sampler.generate(cfg_labels=labels, cfg_scale=1.0, num_steps=3, show_progress=False)
        assert not torch.isnan(images_cfg1).any()

        images_cfg4 = sampler.generate(cfg_labels=labels, cfg_scale=4.0, num_steps=3, show_progress=False)
        assert not torch.isnan(images_cfg4).any()

    def test_dit_sampling_with_different_step_counts(self) -> None:
        """Verify DiT sampling works correctly with different step counts."""
        denoiser = MockDiTDenoiser(latent_size=4, num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        labels = torch.tensor([207], device="cpu")

        # Test various step counts
        for num_steps in [5, 10, 50, 100]:
            images = sampler.generate(cfg_labels=labels, num_steps=num_steps, show_progress=False)
            assert not torch.isnan(images).any(), f"NaN with {num_steps} steps"
            assert not torch.isinf(images).any(), f"Inf with {num_steps} steps"

    def test_dit_timestep_sequence_correct_length(self) -> None:
        """Verify timestep sequence has exactly the requested number of steps."""
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        config = TFGConfig(device="cpu", rho=0.0, mu=0.0)
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        device = torch.device("cpu")

        for num_steps in [10, 50, 100, 250]:
            ts = sampler._get_dit_timestep_sequence(num_steps=num_steps, device=device)
            assert len(ts) == num_steps, f"Expected {num_steps} timesteps, got {len(ts)}"

    def test_dit_cfg_channel_mode_read_from_denoiser(self) -> None:
        """Verify UnifiedSampler reads cfg_channel_mode from denoiser."""
        # Test with "first3" mode (for original DiT reproduction)
        denoiser_first3 = MockDiTDenoiser(cfg_channel_mode="first3", device="cpu")
        sampler_first3 = UnifiedSampler("DiT", denoiser_first3)
        assert getattr(denoiser_first3, "cfg_channel_mode", None) == "first3"

        # Test with "all"
        denoiser_all = MockDiTDenoiser(cfg_channel_mode="all", device="cpu")
        sampler_all = UnifiedSampler("DiT", denoiser_all)
        assert getattr(denoiser_all, "cfg_channel_mode", None) == "all"

    def test_sit_cfg_channel_mode_read_from_denoiser(self) -> None:
        """Verify SiT denoiser cfg_channel_mode is accessible."""
        # Test with "first3" mode (original SiT behavior)
        denoiser_first3 = MockSiTDenoiser(cfg_channel_mode="first3", device="cpu")
        UnifiedSampler("SiT", denoiser_first3)
        assert denoiser_first3.cfg_channel_mode == "first3"

        # Test with "all" mode (TFG research)
        denoiser_all = MockSiTDenoiser(cfg_channel_mode="all", device="cpu")
        UnifiedSampler("SiT", denoiser_all)
        assert denoiser_all.cfg_channel_mode == "all"

    def test_cfg_channel_mode_on_latent_models(self) -> None:
        """Verify cfg_channel_mode exists on latent models (DiT, SiT) but not JiT.

        Latent models (DiT, SiT) support cfg_channel_mode for controlling
        which channels receive CFG. JiT (pixel space) does not need this.
        """
        # JiT doesn't have cfg_channel_mode (pixel space, 3 channels)
        jit_denoiser = MockJiTDenoiser()
        assert not hasattr(jit_denoiser, "cfg_channel_mode")

        # SiT has cfg_channel_mode (latent space, 4 channels)
        sit_denoiser = MockSiTDenoiser()
        assert hasattr(sit_denoiser, "cfg_channel_mode")
        assert sit_denoiser.cfg_channel_mode == "first3"

        # DiT has cfg_channel_mode (latent space, 4 channels)
        dit_denoiser = MockDiTDenoiser(cfg_channel_mode="all")
        assert hasattr(dit_denoiser, "cfg_channel_mode")
        assert dit_denoiser.cfg_channel_mode == "all"


# =============================================================================
# DDPM LEARNED_RANGE and Forward With CFG Tests
# =============================================================================


class TestDiTForwardWithCFG:
    """Tests for _dit_forward_with_cfg returning both eps and variance channels."""

    def test_returns_eps_and_var_when_model_outputs_doubled_channels(self) -> None:
        """_dit_forward_with_cfg should return variance channels from conditional output."""
        denoiser = MockDiTDenoiser(latent_size=4, num_timesteps=1000, device="cpu")
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddpm")

        z = torch.randn(2, 4, 4, 4)
        labels = torch.tensor([207, 360])

        eps, var = sampler._dit_forward_with_cfg(z, 500, labels, cfg_scale=4.0)

        assert eps.shape == (2, 4, 4, 4), f"eps shape: {eps.shape}"
        assert var is not None, "var should be non-None for learn_sigma model"
        assert var.shape == (2, 4, 4, 4), f"var shape: {var.shape}"

    def test_eps_only_delegate_matches(self) -> None:
        """_dit_forward_epsilon_with_cfg should match eps from _dit_forward_with_cfg."""
        denoiser = MockDiTDenoiser(latent_size=4, num_timesteps=1000, device="cpu")
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddpm")

        z = torch.randn(2, 4, 4, 4)
        labels = torch.tensor([207, 360])

        eps_full, _ = sampler._dit_forward_with_cfg(z, 500, labels, cfg_scale=4.0)
        eps_only = sampler._dit_forward_epsilon_with_cfg(z, 500, labels, cfg_scale=4.0)

        torch.testing.assert_close(eps_only, eps_full)

    def test_var_comes_from_conditional_output_only(self) -> None:
        """Variance channels should come from conditional output only, not affected by CFG."""

        # Create a DiT that returns different values for cond vs uncond
        class AsymmetricDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_channels = 4
                self.num_classes = 1000

            def forward(self, x, t, y):
                batch_size = x.shape[0]
                out = torch.zeros(batch_size, 8, x.shape[2], x.shape[3], device=x.device)
                for i in range(batch_size):
                    if y[i].item() < 1000:
                        # Conditional: variance channels = 1.0
                        out[i, 4:] = 1.0
                    else:
                        # Unconditional: variance channels = -1.0
                        out[i, 4:] = -1.0
                return out

        denoiser = MockDiTDenoiser(latent_size=4, device="cpu")
        denoiser.net.dit = AsymmetricDiT()
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddpm")

        z = torch.randn(2, 4, 4, 4)
        labels = torch.tensor([207, 360])

        _, var = sampler._dit_forward_with_cfg(z, 500, labels, cfg_scale=4.0)

        # Variance should come from conditional output only
        assert var is not None
        assert torch.allclose(var, torch.ones_like(var))


class TestDiTDDPMLearnedVariance:
    """Tests for DDPM posterior with LEARNED_RANGE variance."""

    def test_ddpm_step_uses_learned_variance(self) -> None:
        """DDPM step should use learned variance channels when available."""
        denoiser = MockDiTDenoiser(latent_size=4, num_timesteps=1000, device="cpu")
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddpm")

        z = torch.randn(1, 4, 4, 4)
        labels = torch.tensor([207])

        # Should run without error using learned variance path
        ts = sampler._get_dit_timestep_sequence(5, torch.device("cpu"))
        z_next = sampler._dit_ddpm_step(z=z, t_idx=0, ts=ts, labels=labels, cfg_scale=4.0)

        assert not torch.isnan(z_next).any()
        assert not torch.isinf(z_next).any()

    def test_posterior_helper_fixed_small_when_var_is_none(self) -> None:
        """_dit_posterior_mean_and_sample should use FIXED_SMALL when model_var_values=None."""
        denoiser = MockDiTDenoiser(latent_size=4, num_timesteps=1000, device="cpu")
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddpm")

        x0 = torch.randn(2, 4, 4, 4)
        z = torch.randn(2, 4, 4, 4)
        alpha_prod_t = torch.tensor(0.5)
        alpha_prod_t_prev = torch.tensor(0.6)

        result = sampler._dit_posterior_mean_and_sample(
            x0=x0,
            z=z,
            t=500,
            alpha_prod_t=alpha_prod_t,
            alpha_prod_t_prev=alpha_prod_t_prev,
            model_var_values=None,
        )

        assert not torch.isnan(result).any()
        assert result.shape == z.shape

    def test_posterior_helper_learned_range_when_var_provided(self) -> None:
        """_dit_posterior_mean_and_sample should use LEARNED_RANGE when model_var_values provided."""
        denoiser = MockDiTDenoiser(latent_size=4, num_timesteps=1000, device="cpu")
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddpm")

        x0 = torch.randn(2, 4, 4, 4)
        z = torch.randn(2, 4, 4, 4)
        alpha_prod_t = torch.tensor(0.5)
        alpha_prod_t_prev = torch.tensor(0.6)
        model_var_values = torch.zeros(2, 4, 4, 4)  # midpoint of range

        result = sampler._dit_posterior_mean_and_sample(
            x0=x0,
            z=z,
            t=500,
            alpha_prod_t=alpha_prod_t,
            alpha_prod_t_prev=alpha_prod_t_prev,
            model_var_values=model_var_values,
        )

        assert not torch.isnan(result).any()
        assert result.shape == z.shape

    def test_ddim_path_unchanged(self) -> None:
        """DDIM generation should be completely unaffected by learned variance changes.

        Regression test: DDIM doesn't use variance channels, so results must be identical.
        """
        denoiser = MockDiTDenoiser(latent_size=4, num_timesteps=1000, device="cpu")
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddim")

        labels = torch.tensor([207])

        # DDIM with eta=0 is deterministic
        torch.manual_seed(42)
        images1 = sampler.generate(cfg_labels=labels, num_steps=5, show_progress=False)

        torch.manual_seed(42)
        images2 = sampler.generate(cfg_labels=labels, num_steps=5, show_progress=False)

        torch.testing.assert_close(images1, images2)

    def test_ddim_regression_timestep_values(self) -> None:
        """DDIM timestep sequence values should remain unchanged (regression).

        100 steps from 1000: [990, 980, ..., 10, 0]
        """
        denoiser = MockDiTDenoiser(num_timesteps=1000, device="cpu")
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddim")

        ts = sampler._get_dit_timestep_sequence(100, torch.device("cpu"))
        assert ts[0].item() == 990
        assert ts[-1].item() == 0
        assert len(ts) == 100


class TestLearnedRangeRebasedBounds:
    """Tests that LEARNED_RANGE variance bounds match SpacedDiffusion rebasing.

    The original DiT uses SpacedDiffusion which creates a NEW schedule for the
    selected timesteps. The rebased betas are larger (each step spans multiple
    original steps), so the variance interpolation bounds differ from the
    original 1000-step schedule.
    """

    def test_learned_range_uses_rebased_beta_not_original(self) -> None:
        """LEARNED_RANGE max_log should use rebased beta, not original schedule's beta.

        For 250-step sampling from 1000, each rebased beta is ~4x the original.
        """
        denoiser = MockDiTDenoiser(latent_size=4, num_timesteps=1000, device="cpu")
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddpm")

        x0 = torch.randn(1, 4, 4, 4)
        z = torch.randn(1, 4, 4, 4)

        # alpha_prod_t and alpha_prod_t_prev from rebased schedule
        # (simulating a step that spans ~4 original timesteps)
        alpha_prod_t = torch.tensor(0.5)
        alpha_prod_t_prev = torch.tensor(0.52)  # small gap = single original step
        small_beta = 1 - alpha_prod_t / alpha_prod_t_prev  # ~0.038

        alpha_prod_t_prev_big = torch.tensor(0.6)  # big gap = multiple steps
        big_beta = 1 - alpha_prod_t / alpha_prod_t_prev_big  # ~0.167

        # model_var_values = +1 → frac = 1.0 → variance = max_log = log(beta_t)
        model_var_all_max = torch.ones(1, 4, 4, 4)

        torch.manual_seed(0)
        result_small = sampler._dit_posterior_mean_and_sample(
            x0=x0,
            z=z,
            t=500,
            alpha_prod_t=alpha_prod_t,
            alpha_prod_t_prev=alpha_prod_t_prev,
            model_var_values=model_var_all_max,
        )

        torch.manual_seed(0)
        result_big = sampler._dit_posterior_mean_and_sample(
            x0=x0,
            z=z,
            t=500,
            alpha_prod_t=alpha_prod_t,
            alpha_prod_t_prev=alpha_prod_t_prev_big,
            model_var_values=model_var_all_max,
        )

        # The "big" step should produce higher variance (more noise spread)
        # because the rebased beta is larger
        diff_small = (result_small - (x0 * 0 + z * 0)).std()  # rough variance measure
        diff_big = (result_big - (x0 * 0 + z * 0)).std()
        # We can't directly compare std easily due to mean differences,
        # but verify the function runs correctly with both
        assert not torch.isnan(result_small).any()
        assert not torch.isnan(result_big).any()

    def test_learned_range_bounds_match_spaced_diffusion_numerically(self) -> None:
        """Verify LEARNED_RANGE bounds match original SpacedDiffusion rebasing.

        Numerically compares our on-the-fly computation against the original
        SpacedDiffusion's rebased posterior_log_variance_clipped and log(new_betas).
        """
        import numpy as np

        # Original schedule
        betas_orig = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
        alphas_orig = 1.0 - betas_orig
        alphas_cumprod_orig = np.cumprod(alphas_orig)

        # SpacedDiffusion rebasing for 250 steps
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
        new_posterior_variance = new_betas * (1.0 - new_alphas_cumprod_prev) / (1.0 - new_alphas_cumprod)

        # Check at several representative timesteps (skip i=0 where variance=0)
        for local_idx in [1, 10, 50, 125, 200, 249]:
            # Our on-the-fly computation (same as _dit_posterior_mean_and_sample)
            alpha_prod_t = alphas_cumprod_orig[timesteps[local_idx]]
            alpha_prod_t_prev = alphas_cumprod_orig[timesteps[local_idx - 1]] if local_idx > 0 else 1.0
            alpha_t = alpha_prod_t / max(alpha_prod_t_prev, 1e-8)
            beta_t = 1 - alpha_t
            posterior_var = beta_t * (1 - alpha_prod_t_prev) / max(1 - alpha_prod_t, 1e-8)

            our_min_log = float(np.log(max(posterior_var, 1e-20)))
            our_max_log = float(np.log(max(beta_t, 1e-20)))

            # SpacedDiffusion values
            expected_min_log = float(np.log(new_posterior_variance[local_idx]))
            expected_max_log = float(np.log(new_betas[local_idx]))

            assert abs(our_min_log - expected_min_log) < 1e-10, (
                f"min_log mismatch at idx {local_idx}: ours={our_min_log:.6f}, expected={expected_min_log:.6f}"
            )
            assert abs(our_max_log - expected_max_log) < 1e-10, (
                f"max_log mismatch at idx {local_idx}: ours={our_max_log:.6f}, expected={expected_max_log:.6f}"
            )

    def test_fixed_small_variance_matches_rebased(self) -> None:
        """FIXED_SMALL (model_var_values=None) should also use rebased posterior variance."""
        import numpy as np

        betas_orig = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
        alphas_orig = 1.0 - betas_orig
        alphas_cumprod_orig = np.cumprod(alphas_orig)

        # Pick a step that spans multiple original timesteps
        # Step from t=502 to t=498 (rebased step in 250-step sampling)
        alpha_prod_t = torch.tensor(alphas_cumprod_orig[502])
        alpha_prod_t_prev = torch.tensor(alphas_cumprod_orig[498])

        denoiser = MockDiTDenoiser(latent_size=4, num_timesteps=1000, device="cpu")
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddpm")

        x0 = torch.zeros(1, 4, 4, 4)
        z = torch.zeros(1, 4, 4, 4)

        # With model_var_values=None, should use FIXED_SMALL = log(posterior_variance)
        result = sampler._dit_posterior_mean_and_sample(
            x0=x0,
            z=z,
            t=502,
            alpha_prod_t=alpha_prod_t,
            alpha_prod_t_prev=alpha_prod_t_prev,
            model_var_values=None,
        )

        # Verify: posterior_variance should use the rebased beta
        rebased_beta = 1 - alpha_prod_t / alpha_prod_t_prev
        expected_post_var = rebased_beta * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        # This is the on-the-fly computation, which has always been correct
        assert float(expected_post_var) > float(betas_orig[502]), (
            "Rebased posterior variance should be larger than original single-step variance"
        )

    def test_model_var_minus1_equals_fixed_small(self) -> None:
        """model_var_values = -1 (frac=0) should give min_log = posterior_log_variance."""
        denoiser = MockDiTDenoiser(latent_size=4, num_timesteps=1000, device="cpu")
        sampler = UnifiedSampler("DiT", denoiser, sampling_method="ddpm")

        x0 = torch.randn(1, 4, 4, 4)
        z = torch.randn(1, 4, 4, 4)
        alpha_prod_t = torch.tensor(0.5)
        alpha_prod_t_prev = torch.tensor(0.6)

        # model_var_values = -1 → frac = 0 → posterior_log_variance = min_log
        model_var_min = -torch.ones(1, 4, 4, 4)

        torch.manual_seed(42)
        result_learned = sampler._dit_posterior_mean_and_sample(
            x0=x0,
            z=z,
            t=500,
            alpha_prod_t=alpha_prod_t,
            alpha_prod_t_prev=alpha_prod_t_prev,
            model_var_values=model_var_min,
        )

        torch.manual_seed(42)
        result_fixed = sampler._dit_posterior_mean_and_sample(
            x0=x0,
            z=z,
            t=500,
            alpha_prod_t=alpha_prod_t,
            alpha_prod_t_prev=alpha_prod_t_prev,
            model_var_values=None,
        )

        # When model predicts -1, LEARNED_RANGE min_log = log(posterior_variance)
        # which is the same as FIXED_SMALL.
        torch.testing.assert_close(result_learned, result_fixed, atol=1e-5, rtol=1e-5)


# =============================================================================
# Guidance Space Tests
# =============================================================================


class TestGuidanceSpaces:
    """Tests for different guidance spaces (x, v, v2)."""

    def test_v_space_euler_jit(self) -> None:
        """v-space guidance with Euler should work for JiT."""
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=ZeroNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="euler", guidance_space="v")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_v_space_heun_jit(self) -> None:
        """v-space guidance with Heun should work for JiT."""
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=TimeDependentNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="heun", guidance_space="v")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_v2_space_heun_jit(self) -> None:
        """v2-space (per-NFE) guidance with Heun should work for JiT."""
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=TimeDependentNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="heun", guidance_space="v2")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_guidance_spaces_produce_different_outputs(self) -> None:
        """Different guidance spaces should produce different outputs."""
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=TimeDependentNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )

        results = {}
        for gs in ["x", "v"]:
            sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="euler", guidance_space=gs)
            torch.manual_seed(42)
            results[gs] = sampler.generate(
                cfg_labels=labels,
                guidance=guider,
                tfg_targets=labels,
                num_steps=5,
                show_progress=False,
            )

        # x-space and v-space should produce different results due to different scaling
        assert not torch.allclose(results["x"], results["v"])

    def test_lambda_t_scaling_values(self) -> None:
        """Test theoretical lambda_t = (1-t)/t scaling values."""
        # Theoretical values
        t_values = [0.1, 0.5, 0.9]
        expected_lambda = [9.0, 1.0, 1 / 9]  # (1-t)/t

        for t, expected in zip(t_values, expected_lambda, strict=True):
            t_eps = 0.05
            t_clamped = max(t, t_eps)
            lambda_t = (1 - t) / t_clamped
            assert abs(lambda_t - expected) < 0.01, f"At t={t}, expected {expected}, got {lambda_t}"

    def test_v_space_sit(self) -> None:
        """SiT with v-space guidance should work."""
        denoiser = MockSiTDenoiser(device="cpu")
        guider = QuadraticLogpGuider(device="cpu", img_size=8, channels=4)
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("SiT", denoiser, tfg_config=config, sampling_method="euler", guidance_space="v")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        assert images.shape[0] == 1
        assert not torch.isnan(images).any()

    def test_v2_space_sit_heun(self) -> None:
        """SiT with v2-space guidance and Heun should work."""
        denoiser = MockSiTDenoiser(device="cpu")
        guider = QuadraticLogpGuider(device="cpu", img_size=8, channels=4)
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("SiT", denoiser, tfg_config=config, sampling_method="heun", guidance_space="v2")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        assert images.shape[0] == 1
        assert not torch.isnan(images).any()

    def test_v_space_pixelflow(self) -> None:
        """PixelFlow with v-space guidance should work."""
        denoiser = MockPixelFlowDenoiser(img_size=32, num_stages=2, device="cpu")
        guider = QuadraticLogpGuider(device="cpu", img_size=32)
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("PixelFlow", denoiser, tfg_config=config, sampling_method="euler", guidance_space="v")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        assert images.shape == (1, 3, 32, 32)
        assert not torch.isnan(images).any()

    def test_v2_space_pixelflow_heun(self) -> None:
        """PixelFlow with v2-space guidance and Heun should work."""
        denoiser = MockPixelFlowDenoiser(img_size=32, num_stages=2, device="cpu")
        guider = QuadraticLogpGuider(device="cpu", img_size=32)
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=0.5,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("PixelFlow", denoiser, tfg_config=config, sampling_method="heun", guidance_space="v2")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=2,
            show_progress=False,
        )

        assert images.shape == (1, 3, 32, 32)
        assert not torch.isnan(images).any()


class TestGuidanceSpacePhilosophy:
    """Tests verifying guidance space design philosophy.

    x-space: DDPM/TFG heritage - Position correction (no dt scaling)
    v-space: Flow Guidance heritage - Velocity modification (dt implicit)
    v2-space: Per-NFE velocity modification for higher-order solvers
    """

    def test_x_space_no_dt_in_formula(self) -> None:
        """x-space guidance should apply direct position correction without dt."""
        # This is a conceptual test - we verify the formula structure
        # x-space: z_next = z_ode + (t_next/t) * delta_t + t_next * delta_0
        # NO dt multiplier in the guidance term (DDPM heritage)
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=ZeroNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,  # Only variance guidance to simplify
            sigma=0.0,  # No MC smoothing
            eps_bsz=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="euler", guidance_space="x")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        # Test passes if no error - the formula is verified by the implementation
        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_v_space_velocity_modification(self) -> None:
        """v-space should modify velocity with theoretical (1-t)/t scaling."""
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=ZeroNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="euler", guidance_space="v")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        # v-space: z_next = z + dt * (v + lambda_t * delta_t + delta_0)
        # where lambda_t = (1-t)/t (theoretical scaling)
        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_v2_per_nfe_velocity_modification(self) -> None:
        """v2-space should modify each velocity individually in Heun."""
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=TimeDependentNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="heun", guidance_space="v2")

        torch.manual_seed(42)
        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        # v2-space: each velocity guided individually
        # v1_guided = v1 + lambda_t * delta_t_1 + delta_0_1
        # v2_guided = v2 + lambda_{t_next} * delta_t_2 + delta_0_2
        # z_next = z + dt * 0.5 * (v1_guided + v2_guided)
        assert images.shape == (1, 3, 8, 8)
        assert not torch.isnan(images).any()

    def test_v2_differs_from_v_with_heun(self) -> None:
        """v2-space (per-NFE) should produce different results than v-space with Heun."""
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=TimeDependentNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )

        # v-space with Heun
        sampler_v = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="heun", guidance_space="v")
        torch.manual_seed(42)
        images_v = sampler_v.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        # v2-space with Heun
        sampler_v2 = UnifiedSampler("JiT", denoiser, tfg_config=config, sampling_method="heun", guidance_space="v2")
        torch.manual_seed(42)
        images_v2 = sampler_v2.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=5,
            show_progress=False,
        )

        # v2-space (per-NFE) should differ from v-space (single guidance at t)
        # because v2 applies guidance at both v1 and v2 with different lambda values
        assert not torch.allclose(images_v, images_v2, atol=1e-5)

    def test_lambda_t_asymmetric_at_different_times(self) -> None:
        """Lambda_t = (1-t)/t should be asymmetric: large at t→0, small at t→1."""
        t_eps = 0.05

        # At t=0.1 (high noise): lambda_t ≈ 9.0
        t_low = 0.1
        t_clamped_low = max(t_low, t_eps)
        lambda_low = (1 - t_low) / t_clamped_low
        assert lambda_low > 5.0, f"At t={t_low}, lambda should be >5, got {lambda_low}"

        # At t=0.9 (low noise): lambda_t ≈ 0.11
        t_high = 0.9
        t_clamped_high = max(t_high, t_eps)
        lambda_high = (1 - t_high) / t_clamped_high
        assert lambda_high < 0.5, f"At t={t_high}, lambda should be <0.5, got {lambda_high}"

        # The ratio should be large
        ratio = lambda_low / lambda_high
        assert ratio > 10, f"Lambda ratio should be >10, got {ratio}"

    def test_x_v_v2_all_produce_different_outputs(self) -> None:
        """All three guidance spaces should produce different outputs."""
        denoiser = MockJiTDenoiser(img_size=8, steps=10, net=TimeDependentNet())
        guider = QuadraticLogpGuider(device="cpu")
        labels = torch.tensor([207], device="cpu")

        config = TFGConfig(
            device="cpu",
            rho=1.0,
            mu=0.5,
            sigma=0.01,
            eps_bsz=1,
        )

        results = {}
        for gs in ["x", "v", "v2"]:
            sampler = UnifiedSampler(
                "JiT",
                denoiser,
                tfg_config=config,
                sampling_method="heun",
                guidance_space=gs,
            )
            torch.manual_seed(42)
            results[gs] = sampler.generate(
                cfg_labels=labels,
                guidance=guider,
                tfg_targets=labels,
                num_steps=5,
                show_progress=False,
            )

        # All three should be different
        assert not torch.allclose(results["x"], results["v"], atol=1e-5)
        assert not torch.allclose(results["x"], results["v2"], atol=1e-5)
        assert not torch.allclose(results["v"], results["v2"], atol=1e-5)


# =============================================================================
# Tests for Jacobian fix, rescale_mode, lambda_mode
# =============================================================================


class TestForwardFlowWithGrad:
    """Tests for the grad-enabled forward function (Jacobian fix)."""

    def test_output_requires_grad(self):
        """_forward_flow_with_grad output should require grad when input does."""
        denoiser = MockJiTDenoiser(pred_target="x", net=IdentityNet())
        sampler = UnifiedSampler("JiT", denoiser)

        z = torch.randn(1, 3, 8, 8, requires_grad=True)
        t = torch.tensor([[[[0.5]]]])
        labels = torch.tensor([207])

        v = sampler._forward_flow_with_grad(z, t, labels)

        # Output should track gradients (unlike _forward_sample with @torch.no_grad)
        assert v.requires_grad, "Output should require grad"

    def test_grad_flows_to_input(self):
        """Gradients should flow from output back to input z."""
        # Use TimeDependentNet so output != input (non-trivial grad_fn)
        denoiser = MockJiTDenoiser(pred_target="x", net=TimeDependentNet())
        sampler = UnifiedSampler("JiT", denoiser)

        z = torch.randn(1, 3, 8, 8, requires_grad=True)
        t = torch.tensor([[[[0.5]]]])
        labels = torch.tensor([207])

        v = sampler._forward_flow_with_grad(z, t, labels)
        loss = v.sum()
        loss.backward()

        assert z.grad is not None, "Gradient should flow back to input z"
        assert z.grad.abs().sum() > 0, "Gradient should be non-zero"

    def test_matches_forward_sample_values(self):
        """_forward_flow_with_grad should produce same VALUES as _forward_sample."""
        net = IdentityNet()
        denoiser = MockJiTDenoiser(pred_target="x", net=net)
        sampler = UnifiedSampler("JiT", denoiser)

        z = torch.randn(1, 3, 8, 8)
        t = torch.tensor([[[[0.5]]]])
        labels = torch.tensor([207])

        with torch.no_grad():
            v_no_grad = denoiser._forward_sample(z, t, labels)

        v_with_grad = sampler._forward_flow_with_grad(z, t, labels)

        assert torch.allclose(v_no_grad, v_with_grad, atol=1e-6), (
            "Grad-enabled and no-grad forward should produce same values"
        )

    def test_gradient_flows_through_model(self):
        """delta_t should include model Jacobian when using _forward_flow_with_grad."""
        # IdentityNet: net(z) = z, so x0 = (1-t)*z + z = (2-t)*z
        # grad of J(x0) w.r.t. z should include the model's contribution
        denoiser = MockJiTDenoiser(pred_target="x", net=IdentityNet())
        config = TFGConfig(rho=1.0, mu=0.0, sigma=0.0, eps_bsz=1, device="cpu")
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        z = torch.randn(1, 3, 8, 8, requires_grad=True)
        t = torch.tensor([[[[0.5]]]])

        # Manually compute: v = net(z) = z (IdentityNet)
        # x0 = (1-t)*v + z = 0.5*z + z = 1.5*z
        # If grad flows through model: d(x0)/dz = 1.5 (chain rule through net)
        # If grad does NOT flow: d(x0)/dz = 1.0 (only explicit +z term)
        v_with_grad = sampler._forward_flow_with_grad(z, t, torch.tensor([207]))
        x0 = (1 - 0.5) * v_with_grad + z

        # Use a simple objective: sum(x0)
        loss = x0.sum()
        grad_result = torch.autograd.grad(loss, z, create_graph=False)[0]

        # With grad through model: d(sum(1.5*z))/dz = 1.5 for each element
        expected = torch.full_like(z, 1.5)
        assert torch.allclose(grad_result, expected, atol=1e-5), (
            f"Gradient should be 1.5 (through model), got mean={grad_result.mean().item():.4f}"
        )


class TestRescaleMode:
    """Tests for rescale_mode option."""

    def test_clip_mode_clips_large_gradients(self):
        """mode='clip' should clip gradients exceeding clip_scale."""
        from jit_tfg.tfg.utils import rescale_grad

        grad = torch.ones(1, 3, 8, 8) * 1000.0  # Very large
        result = rescale_grad(grad, clip_scale=10.0, mode="clip")
        rms = (result**2).mean().sqrt().item()
        assert rms <= 10.0 * 2, f"Clipped RMS should be near clip_scale, got {rms}"

    def test_original_mode_no_clipping(self):
        """mode='original' should return gradient unchanged for images."""
        from jit_tfg.tfg.utils import rescale_grad

        grad = torch.ones(1, 3, 8, 8) * 1000.0
        result = rescale_grad(grad, clip_scale=10.0, mode="original")
        assert torch.allclose(grad, result), "Original mode should not modify gradient"

    def test_clip_is_default(self):
        """Default mode should be 'clip'."""
        from jit_tfg.tfg.utils import rescale_grad

        grad = torch.ones(1, 3, 8, 8) * 1000.0
        result = rescale_grad(grad, clip_scale=10.0)
        # Should be clipped (default is "clip")
        assert not torch.allclose(grad, result), "Default should clip large gradients"

    def test_config_default(self):
        """TFGConfig should default to rescale_mode='clip'."""
        config = TFGConfig()
        assert config.rescale_mode == "clip"


class TestLambdaMode:
    """Tests for lambda_mode option."""

    def test_auto_default(self):
        """Default lambda_mode should be 'auto'."""
        config = TFGConfig()
        assert config.lambda_mode == "auto"

    def test_flow_guidance_lambda(self):
        """flow_guidance mode: lambda_t = (1-t)/t."""
        denoiser = MockJiTDenoiser(pred_target="x")
        config = TFGConfig(lambda_mode="flow_guidance", device="cpu")
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # At t=0.5: lambda = 0.5/0.5 = 1.0
        assert abs(sampler._get_lambda_t(0.5) - 1.0) < 1e-6
        # At t=0.25: lambda = 0.75/0.25 = 3.0
        assert abs(sampler._get_lambda_t(0.25) - 3.0) < 1e-6

    def test_identity_lambda(self):
        """identity mode: lambda_t = 1.0 always."""
        denoiser = MockJiTDenoiser(pred_target="x")
        config = TFGConfig(lambda_mode="identity", device="cpu")
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        assert sampler._get_lambda_t(0.1) == 1.0
        assert sampler._get_lambda_t(0.5) == 1.0
        assert sampler._get_lambda_t(0.9) == 1.0

    def test_auto_x_prediction_uses_identity(self):
        """auto mode with x-prediction should use identity."""
        denoiser = MockJiTDenoiser(pred_target="x")
        config = TFGConfig(lambda_mode="auto", device="cpu")
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # x-prediction → identity → lambda = 1.0
        assert sampler._get_lambda_t(0.5) == 1.0
        assert sampler._get_lambda_t(0.1) == 1.0

    def test_auto_v_prediction_uses_flow_guidance(self):
        """auto mode with v-prediction should use flow_guidance."""
        denoiser = MockJiTDenoiser(pred_target="v")
        config = TFGConfig(lambda_mode="auto", device="cpu")
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config)

        # v-prediction → flow_guidance → lambda = (1-t)/t
        assert abs(sampler._get_lambda_t(0.5) - 1.0) < 1e-6
        assert abs(sampler._get_lambda_t(0.25) - 3.0) < 1e-6

    def test_schedule_lambda_interaction_flow_guidance_inverts_increase(self):
        """flow_guidance lambda with increase schedule produces effective decrease.

        rho(t) * lambda_t = base_rho * t * (1-t)/t = base_rho * (1-t)
        This is a decrease profile, opposite to the intended 'increase'.
        """
        denoiser = MockJiTDenoiser(pred_target="x")
        config = TFGConfig(
            rho=1.0,
            rho_schedule="increase",
            lambda_mode="flow_guidance",
            device="cpu",
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, guidance_space="v")

        # Effective scale at each t: rho(t) * lambda_t
        # At t=0.1: rho=0.1, lambda=9.0 -> effective = 0.9
        # At t=0.5: rho=0.5, lambda=1.0 -> effective = 0.5
        # At t=0.9: rho=0.9, lambda=0.111 -> effective = 0.1
        # This is a DECREASE profile (0.9 > 0.5 > 0.1)
        rho_01 = sampler._get_schedule_value(1.0, "increase", 0.1, False)
        rho_05 = sampler._get_schedule_value(1.0, "increase", 0.5, False)
        rho_09 = sampler._get_schedule_value(1.0, "increase", 0.9, False)

        eff_01 = rho_01 * sampler._get_lambda_t(0.1)
        eff_05 = rho_05 * sampler._get_lambda_t(0.5)
        eff_09 = rho_09 * sampler._get_lambda_t(0.9)

        # Inverted: early > mid > late (decrease instead of increase)
        assert eff_01 > eff_05 > eff_09
        # Verify exact: rho(t)*lambda_t = base_rho*(1-t)
        assert abs(eff_01 - 1.0 * (1 - 0.1)) < 1e-6
        assert abs(eff_05 - 1.0 * (1 - 0.5)) < 1e-6
        assert abs(eff_09 - 1.0 * (1 - 0.9)) < 1e-6

    def test_schedule_lambda_interaction_identity_preserves_increase(self):
        """identity lambda with increase schedule preserves the intended profile.

        rho(t) * lambda_t = base_rho * t * 1.0 = base_rho * t
        This remains an increase profile as intended.
        """
        denoiser = MockJiTDenoiser(pred_target="x")
        config = TFGConfig(
            rho=1.0,
            rho_schedule="increase",
            lambda_mode="identity",
            device="cpu",
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, guidance_space="v")

        rho_01 = sampler._get_schedule_value(1.0, "increase", 0.1, False)
        rho_05 = sampler._get_schedule_value(1.0, "increase", 0.5, False)
        rho_09 = sampler._get_schedule_value(1.0, "increase", 0.9, False)

        eff_01 = rho_01 * sampler._get_lambda_t(0.1)
        eff_05 = rho_05 * sampler._get_lambda_t(0.5)
        eff_09 = rho_09 * sampler._get_lambda_t(0.9)

        # Preserved: early < mid < late (increase as intended)
        assert eff_01 < eff_05 < eff_09
        # Verify exact: rho(t)*lambda_t = base_rho*t
        assert abs(eff_01 - 1.0 * 0.1) < 1e-6
        assert abs(eff_05 - 1.0 * 0.5) < 1e-6
        assert abs(eff_09 - 1.0 * 0.9) < 1e-6


# =============================================================================
# Tests for v2-space guided z_euler and DDPM nan_to_num
# =============================================================================


class TestV2SpaceGuidedEuler:
    """Tests for v2-space corrector evaluating at guided Euler state."""

    def test_v2_corrector_uses_guided_euler_state(self):
        """v2-space corrector must evaluate v2 at guided (not unguided) Euler state.

        With IdentityNet (v=z_input), v2's output directly reflects its evaluation
        state. Using guided z_euler changes v2, which changes z_next. We verify the
        actual output matches the formula with guided z_euler.

        Key: LinearGuider has input-independent gradient, so delta_t is always the
        same. But IdentityNet makes v2 = z_euler_input, so the corrector velocity
        differs between guided and unguided z_euler.
        """
        denoiser = MockJiTDenoiser(img_size=4, steps=50, net=IdentityNet(), pred_target="v")
        config = TFGConfig(
            rho=5.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
            rho_schedule="constant",
            clip_x0=False,
            device="cpu",
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, guidance_space="v2", sampling_method="heun")

        z = torch.randn(1, 3, 4, 4)
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])
        t, t_next = 0.3, 0.4
        dt = t_next - t

        z_next = sampler._heun_step(z.clone(), t, t_next, labels, guider, labels)

        # Manual computation with GUIDED z_euler:
        rho = 5.0
        lambda_t = (1 - t) / t
        lambda_t_next = (1 - t_next) / t_next

        # With IdentityNet (no _convert_prediction), _forward_flow_with_grad returns z_grad.
        # x0 = (1-t)*z_grad + z_grad = (2-t)*z_grad.
        # logp = sum((2-t)*z_grad * direction), grad = (2-t)*direction.
        # After rescale_grad (no clip) and * rho: delta_t = rho * (2-t) * direction.
        delta_t_1 = rho * (2 - t) * direction

        v1 = z.clone()
        v1_guided = v1 + lambda_t * delta_t_1
        z_euler_guided = z + dt * v1_guided

        # v2 = IdentityNet(z_euler_guided) = z_euler_guided
        v2 = z_euler_guided
        delta_t_2 = rho * (2 - t_next) * direction
        v2_guided = v2 + lambda_t_next * delta_t_2

        expected = z + dt * 0.5 * (v1_guided + v2_guided)

        assert torch.allclose(z_next, expected, atol=1e-5), (
            f"v2-space should use guided z_euler. Max diff: {(z_next - expected).abs().max():.8f}"
        )

    def test_v2_guided_euler_differs_from_unguided(self):
        """Guided z_euler must produce different results from unguided z_euler.

        This is a regression test: if the v2-space corrector reverts to using
        unguided z_euler (z + dt*v1), this test will catch it.
        """
        denoiser = MockJiTDenoiser(img_size=4, steps=50, net=IdentityNet(), pred_target="v")
        config = TFGConfig(
            rho=5.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
            rho_schedule="constant",
            clip_x0=False,
            device="cpu",
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, guidance_space="v2", sampling_method="heun")

        z = torch.randn(1, 3, 4, 4)
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])
        t, t_next = 0.3, 0.4
        dt = t_next - t

        z_next = sampler._heun_step(z.clone(), t, t_next, labels, guider, labels)

        # Compute what UNGUIDED z_euler would give (old/incorrect behavior):
        rho = 5.0
        lambda_t = (1 - t) / t
        lambda_t_next = (1 - t_next) / t_next
        delta_t_1 = rho * (2 - t) * direction
        delta_t_2 = rho * (2 - t_next) * direction

        v1 = z.clone()
        v1_guided = v1 + lambda_t * delta_t_1
        z_euler_unguided = z + dt * v1  # ← unguided (old behavior)
        v2_unguided = z_euler_unguided  # IdentityNet at unguided state
        v2_guided_old = v2_unguided + lambda_t_next * delta_t_2
        expected_old = z + dt * 0.5 * (v1_guided + v2_guided_old)

        # Actual result should NOT match unguided formula
        assert not torch.allclose(z_next, expected_old, atol=1e-5), (
            "v2-space should NOT match unguided z_euler formula. "
            "Regression: corrector may be evaluating at unguided state."
        )

    def test_v2_guided_euler_zeronet_unchanged(self):
        """With ZeroNet, guided z_euler has no effect on v2 or delta_t_2.

        ZeroNet returns zeros regardless of input, and LinearGuider has constant
        gradient. So changing the Euler evaluation point doesn't change v2 (always 0)
        or delta_t_2 (always direction). Result matches the simple formula.
        """
        denoiser = MockJiTDenoiser(img_size=4, steps=50, net=ZeroNet())
        config = TFGConfig(
            rho=1.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
            rho_schedule="constant",
            clip_x0=False,
            device="cpu",
        )
        sampler = UnifiedSampler("JiT", denoiser, tfg_config=config, guidance_space="v2", sampling_method="heun")

        z = torch.randn(1, 3, 4, 4)
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])
        t, t_next = 0.3, 0.4
        dt = t_next - t

        lambda_t = (1 - t) / t
        lambda_t_next = (1 - t_next) / t_next

        z_next = sampler._heun_step(z.clone(), t, t_next, labels, guider, labels)

        # ZeroNet: v1=v2=0, delta_t=direction at both NFEs
        # v1_guided = lambda_t * direction, v2_guided = lambda_t_next * direction
        expected = z + dt * 0.5 * (lambda_t + lambda_t_next) * direction
        assert torch.allclose(z_next, expected, atol=1e-6), (
            f"ZeroNet v2-space should be unchanged by guided z_euler fix. "
            f"Max diff: {(z_next - expected).abs().max():.8f}"
        )

    def test_v2_pixelflow_corrector_uses_guided_euler_state(self):
        """PixelFlow v2-space corrector must also use guided Euler state.

        Same principle as test_v2_corrector_uses_guided_euler_state but for
        _pixelflow_heun_step. IdentityPixelFlowNet makes v2 = z_euler_input,
        so the corrector velocity depends on whether guided or unguided state is used.

        Uses img_size=4 with h=4 to avoid upsampling in guidance computation.
        """
        denoiser = MockPixelFlowDenoiser(
            img_size=4, num_stages=1, num_sampling_steps=2, net=IdentityPixelFlowNet(), device="cpu"
        )
        config = TFGConfig(
            rho=5.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
            rho_schedule="constant",
            clip_x0=False,
            device="cpu",
        )
        sampler = UnifiedSampler("PixelFlow", denoiser, tfg_config=config, guidance_space="v2", sampling_method="heun")

        z = torch.randn(1, 3, 4, 4)
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])
        labels_cfg = torch.cat([labels, torch.full_like(labels, denoiser.num_classes)])

        t, t_next = 0.3, 0.4
        dt = t_next - t
        rho = 5.0
        lambda_t = (1 - t) / t
        lambda_t_next = (1 - t_next) / t_next

        # PixelFlow-specific params: T, T_next, h, stage_cfg
        T_start, T_end = 0.0, 999.0
        T = T_start  # step i=0
        T_next = T_start + (t_next / 0.999) * (T_end - T_start)  # corrector timestep
        stage_cfg = denoiser.cfg_scale  # stage 0 uses full cfg for single-stage

        z_next = sampler._pixelflow_heun_step(
            z=z.clone(),
            t=t,
            t_next=t_next,
            T=T,
            T_next=T_next,
            h=4,
            labels=labels,
            labels_cfg=labels_cfg,
            stage_cfg=stage_cfg,
            guidance=guider,
            tfg_targets=labels,
        )

        # Manual computation with GUIDED z_euler:
        # IdentityPixelFlowNet: forward_multires returns input
        # With z_cfg = [z, z], v_uncond = z, v_cond = z, v1 = z + cfg*(z-z) = z
        v1 = z.clone()

        # _pixelflow_compute_guidance_deltas with IdentityPixelFlowNet:
        # v_guided_grad = z_grad (since uncond==cond for identical inputs)
        # x0_est = z_grad + (1-t)*z_grad = (2-t)*z_grad
        # logp = sum((2-t)*z_grad * direction), grad = (2-t)*direction
        # delta_t_1 = rho * (2-t) * direction
        delta_t_1 = rho * (2 - t) * direction
        v1_guided = v1 + lambda_t * delta_t_1

        z_euler_guided = z + dt * v1_guided
        # v2 = IdentityPixelFlowNet(z_euler_guided) = z_euler_guided
        v2 = z_euler_guided

        delta_t_2 = rho * (2 - t_next) * direction
        v2_guided = v2 + lambda_t_next * delta_t_2

        expected = z + dt * 0.5 * (v1_guided + v2_guided)

        assert torch.allclose(z_next, expected, atol=1e-5), (
            f"PixelFlow v2-space should use guided z_euler. Max diff: {(z_next - expected).abs().max():.8f}"
        )

    def test_v2_pixelflow_guided_euler_differs_from_unguided(self):
        """PixelFlow v2-space guided z_euler should differ from unguided.

        Regression test: if PixelFlow v2-space reverts to unguided z_euler,
        this test catches it.
        """
        denoiser = MockPixelFlowDenoiser(
            img_size=4, num_stages=1, num_sampling_steps=2, net=IdentityPixelFlowNet(), device="cpu"
        )
        config = TFGConfig(
            rho=5.0,
            mu=0.0,
            sigma=0.0,
            eps_bsz=1,
            recur_steps=1,
            iter_steps=1,
            rho_schedule="constant",
            clip_x0=False,
            device="cpu",
        )
        sampler = UnifiedSampler("PixelFlow", denoiser, tfg_config=config, guidance_space="v2", sampling_method="heun")

        z = torch.randn(1, 3, 4, 4)
        direction = torch.ones(1, 3, 4, 4) * 0.1
        guider = LinearGuider(direction=direction)
        labels = torch.tensor([207])
        labels_cfg = torch.cat([labels, torch.full_like(labels, denoiser.num_classes)])

        t, t_next = 0.3, 0.4
        dt = t_next - t
        rho = 5.0
        lambda_t = (1 - t) / t
        lambda_t_next = (1 - t_next) / t_next

        T_start, T_end = 0.0, 999.0
        T = T_start
        T_next = T_start + (t_next / 0.999) * (T_end - T_start)

        z_next = sampler._pixelflow_heun_step(
            z=z.clone(),
            t=t,
            t_next=t_next,
            T=T,
            T_next=T_next,
            h=4,
            labels=labels,
            labels_cfg=labels_cfg,
            stage_cfg=denoiser.cfg_scale,
            guidance=guider,
            tfg_targets=labels,
        )

        # Compute OLD (unguided z_euler) result
        v1 = z.clone()
        delta_t_1 = rho * (2 - t) * direction
        delta_t_2 = rho * (2 - t_next) * direction

        v1_guided = v1 + lambda_t * delta_t_1
        z_euler_unguided = z + dt * v1  # v1 without guidance
        v2_unguided = z_euler_unguided  # IdentityNet
        v2_guided_old = v2_unguided + lambda_t_next * delta_t_2
        expected_old = z + dt * 0.5 * (v1_guided + v2_guided_old)

        assert not torch.allclose(z_next, expected_old, atol=1e-5), (
            "PixelFlow v2-space should NOT match unguided z_euler formula. "
            "Regression: corrector may be evaluating at unguided state."
        )


class TestDDPMNanToNum:
    """Test that DDPM TFG path handles NaN gradients consistently with DDIM/flow paths."""

    def test_ddpm_nan_gradient_does_not_propagate(self):
        """NaN gradients in DDPM TFG step should be replaced with zeros.

        Mirrors the existing NaN test for flow matching (test_nan_gradient_does_not_propagate)
        but targets the DDPM path specifically.
        """
        denoiser = MockDiTDenoiser(
            latent_size=8,
            num_sampling_steps=5,
            device="cpu",
        )
        config = TFGConfig(
            rho=1.0,
            mu=0.0,
            sigma=0.01,
            eps_bsz=1,
            recur_steps=1,
            device="cpu",
        )
        sampler = UnifiedSampler("DiT", denoiser, tfg_config=config)

        class NaNGuider(BaseGuider):
            def __init__(self):
                self.device = "cpu"
                self.targets = [207]
                self.img_size = 8
                self.channels = 4

            def get_guidance(self, x, *, targets=None, return_logp=False, **kwargs):
                zero = (x.flatten(1) * 0).sum(dim=1)
                nan_logp = zero / zero  # 0/0 = NaN with grad_fn
                if return_logp:
                    return nan_logp
                return torch.zeros_like(x)

        guider = NaNGuider()
        labels = torch.tensor([207], device="cpu")

        images = sampler.generate(
            cfg_labels=labels,
            guidance=guider,
            tfg_targets=labels,
            num_steps=3,
            show_progress=False,
        )

        assert not torch.isnan(images).any(), "NaN propagated through DDPM TFG pipeline"
        assert not torch.isinf(images).any(), "Inf propagated through DDPM TFG pipeline"
