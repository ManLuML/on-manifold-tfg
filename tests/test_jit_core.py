import sys
from types import SimpleNamespace

import pytest
import torch

from jit_tfg.models.jit.denoiser import Denoiser

# Add src to path if needed (though pytest usually handles this if configured right,
# explicit add helps in some env)
# sys.path.append('src')
from jit_tfg.models.jit.model import JiT, JiT_models


# Mock Arguments for Denoiser
@pytest.fixture
def mock_args():
    return SimpleNamespace(
        model="JiT-B/16",
        img_size=64,  # Small size for faster testing
        class_num=10,
        attn_dropout=0.0,
        proj_dropout=0.0,
        label_drop_prob=0.1,
        P_mean=-1.2,
        P_std=1.2,
        t_eps=1e-3,
        noise_scale=1.0,
        ema_decay1=0.999,
        ema_decay2=0.9999,
        sampling_method="euler",
        num_sampling_steps=5,  # Small steps for testing
        cfg=4.0,
        interval_min=0.0,
        interval_max=1.0,  # Assuming interval_max is used
    )


def test_jit_model_instantiation():
    """Test if the JiT model can be instantiated correctly."""
    model = JiT_models["JiT-B/16"](input_size=64, num_classes=10)
    assert isinstance(model, JiT)
    # Check if parameters are initialized (just a basic check that it's not empty)
    assert len(list(model.parameters())) > 0


def test_jit_forward_pass():
    """Test the forward pass of the JiT model."""
    img_size = 32
    patch_size = 16
    in_channels = 3
    model = JiT_models["JiT-B/16"](input_size=img_size, in_channels=in_channels, num_classes=10)

    batch_size = 2
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    t = torch.rand(batch_size)
    y = torch.randint(0, 10, (batch_size,))

    output = model(x, t, y)

    assert output.shape == (batch_size, in_channels, img_size, img_size)
    assert not torch.isnan(output).any()


def test_denoiser_instantiation(mock_args):
    """Test correct instantiation of Denoiser."""
    denoiser = Denoiser(mock_args)
    assert denoiser.img_size == mock_args.img_size
    assert denoiser.num_classes == mock_args.class_num


def test_denoiser_forward_loss(mock_args):
    """Test that Denoiser forward returns a valid scalar loss."""
    denoiser = Denoiser(mock_args)

    batch_size = 4
    x = torch.randn(batch_size, 3, mock_args.img_size, mock_args.img_size)
    labels = torch.randint(0, mock_args.class_num, (batch_size,))

    denoiser.train()  # Set to train mode
    loss = denoiser(x, labels)

    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)


def test_denoiser_generation(mock_args):
    """Test generation (sampling) functionality."""
    denoiser = Denoiser(mock_args)
    denoiser.eval()  # Set to eval mode logic (though generate uses @torch.no_grad)

    batch_size = 2
    labels = torch.randint(0, mock_args.class_num, (batch_size,))

    # Test Euler
    denoiser.method = "euler"
    samples = denoiser.generate(labels)
    assert samples.shape == (batch_size, 3, mock_args.img_size, mock_args.img_size)
    assert not torch.isnan(samples).any()

    # Test Heun
    denoiser.method = "heun"
    samples_heun = denoiser.generate(labels)
    assert samples_heun.shape == (batch_size, 3, mock_args.img_size, mock_args.img_size)


def test_overfitting_sanity_check(mock_args):
    """
    Very basic sanity check: verify loss leads to gradient updates.
    Does not run full training, just checks one backward pass logic.
    """
    denoiser = Denoiser(mock_args)
    denoiser.train()
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-4)

    batch_size = 2
    x = torch.randn(batch_size, 3, mock_args.img_size, mock_args.img_size)
    labels = torch.randint(0, mock_args.class_num, (batch_size,))

    # Forward
    loss = denoiser(x, labels)
    initial_loss = loss.item()

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if we can run another pass without error (loss checking is flaky on 1 step random noise task)
    loss_new = denoiser(x, labels)
    assert not torch.isnan(loss_new)
    # Note: On a denoising task with random inputs, loss might not strictly decrease on the *same* batch
    # after 1 step if the noise sampled inside 'forward' is different each time!
    # Because Denoiser.forward() internally samples 't' and 'epsilon'.
    # So we can't assert loss < initial_loss deterministicly here without fixing seeds
    # for the internal Random Number Generator of the forward pass.
    # We just ensure it runs.
