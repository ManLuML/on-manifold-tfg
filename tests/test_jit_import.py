import pytest
import torch


def test_jit_imports():
    """Verify that JiT modules can be imported without error."""
    try:
        from jit_tfg.models.jit import denoiser, model, prepare_ref
        from jit_tfg.models.jit.utils import crop, model_util

        assert hasattr(model, "JiT_models")
        assert hasattr(denoiser, "Denoiser")
    except ImportError as e:
        pytest.fail(f"Failed to import JiT modules: {e}")


def test_jit_model_instantiation():
    """Verify that a small JiT model can be instantiated."""
    from jit_tfg.models.jit.model import JiT_models

    try:
        # Instantiate a small model configuration
        model_fn = JiT_models["JiT-B/16"]
        model = model_fn(input_size=32, num_classes=10)
        assert isinstance(model, torch.nn.Module)
    except Exception as e:
        pytest.fail(f"Failed to instantiate JiT model: {e}")
