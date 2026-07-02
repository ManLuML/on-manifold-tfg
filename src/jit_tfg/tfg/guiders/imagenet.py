"""ImageNet classifier guiders for TFG guidance.

This module provides guiders for ImageNet-1K classification guidance,
supporting both pixel-space models (JiT, PixelFlow) and latent-space
models (DiT, SiT).

Classes:
    - ImageClassifierGuider: Pixel-space classifier guidance
    - LatentClassifierGuider: Latent-space classifier guidance (with VAE decode)
    - LatentMultiTargetGuider: Latent-space with per-sample targets

Functions:
    - create_classifier_guider: Factory function for auto-selecting guider type
    - load_imagenet_classifier: Load pretrained ImageNet classifier
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from jit_tfg.tfg.guiders.base import BaseGuider
from jit_tfg.tfg.utils import check_grad_fn, rescale_grad

if TYPE_CHECKING:
    from jit_tfg.models.dit.vae import VAEHandler


class ImageClassifierGuider(BaseGuider):
    """Guider using an image classifier for class-conditional guidance.

    Computes guidance gradients using the log-probability from a classifier,
    steering generation toward a target class.

    Supports two modes:
        1. Fixed targets: Set at initialization, applied to all samples.
        2. Per-sample targets: Passed at inference time via get_guidance(targets=...).

    Attributes:
        classifier: Pre-trained classifier model.
        default_targets: Default target class indices (used when targets not provided).
    """

    def __init__(
        self,
        classifier: nn.Module,
        targets: list[int] | None = None,
        device: str = "cuda",
        clip_scale: float = 1.0,
        post_process: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize the image classifier guider.

        Args:
            classifier: Pre-trained classifier model.
                Should accept (B, C, H, W) images and return (B, num_classes) logits.
            targets: Default target class indices. If None, targets must be provided
                at inference time via get_guidance(targets=...).
            device: Device for computation.
            clip_scale: Maximum gradient norm for clipping.
            post_process: Optional function to transform x before classification
                (e.g., resize/normalize for classifier input).
        """
        super().__init__(device)
        self.classifier = classifier.to(device).eval()
        self.default_targets = targets
        self.clip_scale = clip_scale
        self.post_process = post_process

        for param in self.classifier.parameters():
            param.requires_grad = False

    @torch.enable_grad()
    def get_guidance(
        self,
        x: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        return_logp: bool = False,
        check_grad: bool = True,
        post_process: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute classifier guidance.

        Computes the gradient of log p(y|x) where y is the target class.

        Args:
            x: Input images of shape (B, C, H, W) requiring gradients.
            targets: Per-sample target classes of shape (B,). If provided,
                each sample is guided toward its corresponding target class.
                If None, uses self.default_targets (fixed targets for all samples).
            return_logp: If True, return log-probability instead of gradient.
            check_grad: If True, verify x.requires_grad is set.
            post_process: Optional function to transform x before classification.
            **kwargs: Additional arguments (ignored).

        Returns:
            If return_logp:
                Log-probability tensor of shape (B,).
            Else:
                Gradient tensor of shape (B, C, H, W).

        Raises:
            ValueError: If neither targets nor self.default_targets is provided.
        """
        if check_grad:
            check_grad_fn(x)

        processor = post_process if post_process is not None else self.post_process
        x_processed = processor(x) if processor is not None else x

        logits = self.classifier(x_processed)
        log_probs = torch.log_softmax(logits, dim=-1)

        if targets is not None:
            target_log_probs = log_probs.gather(1, targets.view(-1, 1)).squeeze(-1)
        elif self.default_targets is not None:
            target_log_probs = log_probs[:, self.default_targets].sum(dim=-1)
        else:
            raise ValueError(
                "No targets provided. Either pass targets to get_guidance() or set default_targets at initialization."
            )

        if return_logp:
            return target_log_probs

        grad = torch.autograd.grad(target_log_probs.sum(), x)[0]
        return rescale_grad(grad, clip_scale=self.clip_scale, **kwargs)


class LatentClassifierGuider(BaseGuider):
    """Classifier guider for latent diffusion models (e.g., DiT, SiT).

    Unlike ImageClassifierGuider which operates directly on pixel outputs,
    this guider:
    1. Receives x_0 prediction in latent space (B, 4, 32, 32)
    2. Decodes to pixel space via VAE (differentiable)
    3. Runs classifier on pixel images
    4. Returns gradients w.r.t. latent input

    This enables TFG guidance for DiT/SiT while keeping standard ImageNet
    classifiers in pixel space.

    Attributes:
        classifier: Pre-trained classifier model.
        vae: VAEHandler for latent -> pixel decoding.
        targets: Target class indices for guidance.
        clip_scale: Maximum gradient norm for clipping.

    Example:
        >>> vae = VAEHandler(device="cuda")
        >>> classifier = torchvision.models.resnet50(pretrained=True)
        >>> guider = LatentClassifierGuider(classifier, vae, targets=[207])
        >>> x_latent = torch.randn(1, 4, 32, 32, device="cuda", requires_grad=True)
        >>> grad = guider.get_guidance(x_latent)
    """

    def __init__(
        self,
        classifier: nn.Module,
        vae: VAEHandler,
        targets: list[int],
        device: str = "cuda",
        clip_scale: float = 1.0,
        classifier_input_size: int = 224,
    ) -> None:
        """Initialize the latent classifier guider.

        Args:
            classifier: Pre-trained classifier model.
                Should accept (B, 3, H, W) images and return (B, num_classes) logits.
            vae: VAEHandler instance for latent -> pixel decoding.
            targets: List of target class indices. Guidance maximizes probability
                of these classes.
            device: Device for computation.
            clip_scale: Maximum gradient norm for clipping.
            classifier_input_size: Input size expected by classifier (default: 224).
        """
        super().__init__(device)
        self.classifier = classifier.to(device).eval()
        self.vae = vae
        self.targets = targets
        self.clip_scale = clip_scale
        self.classifier_input_size = classifier_input_size

        for param in self.classifier.parameters():
            param.requires_grad = False

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _preprocess_for_classifier(self, x_pixel: torch.Tensor) -> torch.Tensor:
        """Preprocess pixel images for classifier input.

        Args:
            x_pixel: Images of shape (B, 3, H, W) in [-1, 1] range.

        Returns:
            Preprocessed images of shape (B, 3, 224, 224) normalized.
        """
        x = (x_pixel + 1) / 2

        if x.shape[-1] != self.classifier_input_size:
            x = F.interpolate(
                x,
                size=self.classifier_input_size,
                mode="bilinear",
                align_corners=False,
            )

        x = self.normalize(x)
        return x

    @torch.enable_grad()
    def get_guidance(
        self,
        x: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        return_logp: bool = False,
        check_grad: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Compute classifier guidance in latent space.

        The guidance is computed by:
        1. Decoding latent to pixel space (differentiable)
        2. Running classifier on pixel images
        3. Computing gradient of log p(y|x) w.r.t. latent

        Args:
            x: Latent x_0 prediction of shape (B, 4, 32, 32) requiring gradients.
            targets: Ignored for this guider (uses self.targets).
            return_logp: If True, return log-probability instead of gradient.
            check_grad: If True, verify x.requires_grad is set.
            **kwargs: Additional arguments (ignored).

        Returns:
            If return_logp:
                Log-probability tensor of shape (B,).
            Else:
                Gradient tensor of shape (B, 4, 32, 32).
        """
        if check_grad:
            check_grad_fn(x)

        x_pixel = self.vae.decode_with_grad(x)
        x_processed = self._preprocess_for_classifier(x_pixel)

        logits = self.classifier(x_processed)
        log_probs = torch.log_softmax(logits, dim=-1)

        target_log_probs = log_probs[:, self.targets].sum(dim=-1)

        if return_logp:
            return target_log_probs

        grad = torch.autograd.grad(target_log_probs.sum(), x)[0]
        return rescale_grad(grad, clip_scale=self.clip_scale, **kwargs)


class LatentMultiTargetGuider(BaseGuider):
    """Latent guider supporting per-sample target classes.

    Unlike LatentClassifierGuider which uses a fixed list of targets,
    this guider allows different target classes for each sample in the batch.

    Useful for class-conditional generation where each sample targets
    a different class.

    Attributes:
        classifier: Pre-trained classifier model.
        vae: VAEHandler for latent -> pixel decoding.
        clip_scale: Maximum gradient norm for clipping.

    Example:
        >>> guider = LatentMultiTargetGuider(classifier, vae)
        >>> x_latent = torch.randn(4, 4, 32, 32, requires_grad=True)
        >>> targets = torch.tensor([207, 360, 85, 543])  # Per-sample targets
        >>> grad = guider.get_guidance(x_latent, targets=targets)
    """

    def __init__(
        self,
        classifier: nn.Module,
        vae: VAEHandler,
        device: str = "cuda",
        clip_scale: float = 1.0,
        classifier_input_size: int = 224,
    ) -> None:
        """Initialize the multi-target guider.

        Args:
            classifier: Pre-trained classifier model.
            vae: VAEHandler instance.
            device: Device for computation.
            clip_scale: Maximum gradient norm for clipping.
            classifier_input_size: Input size for classifier.
        """
        super().__init__(device)
        self.classifier = classifier.to(device).eval()
        self.vae = vae
        self.clip_scale = clip_scale
        self.classifier_input_size = classifier_input_size

        for param in self.classifier.parameters():
            param.requires_grad = False

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _preprocess_for_classifier(self, x_pixel: torch.Tensor) -> torch.Tensor:
        """Preprocess pixel images for classifier input."""
        x = (x_pixel + 1) / 2
        if x.shape[-1] != self.classifier_input_size:
            x = F.interpolate(
                x,
                size=self.classifier_input_size,
                mode="bilinear",
                align_corners=False,
            )
        x = self.normalize(x)
        return x

    @torch.enable_grad()
    def get_guidance(
        self,
        x: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        return_logp: bool = False,
        check_grad: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Compute guidance with per-sample targets.

        Args:
            x: Latent x_0 prediction of shape (B, 4, 32, 32).
            targets: Per-sample target classes of shape (B,). Required.
            return_logp: If True, return log-probability instead of gradient.
            check_grad: If True, verify x.requires_grad is set.
            **kwargs: Additional arguments.

        Returns:
            If return_logp:
                Log-probability tensor of shape (B,).
            Else:
                Gradient tensor of shape (B, 4, 32, 32).
        """
        if check_grad:
            check_grad_fn(x)

        if targets is None:
            raise ValueError("targets must be provided for multi-target guidance")

        x_pixel = self.vae.decode_with_grad(x)
        x_processed = self._preprocess_for_classifier(x_pixel)

        logits = self.classifier(x_processed)
        log_probs = torch.log_softmax(logits, dim=-1)

        batch_size = x.shape[0]
        target_log_probs = log_probs[torch.arange(batch_size, device=x.device), targets]

        if return_logp:
            return target_log_probs

        grad = torch.autograd.grad(target_log_probs.sum(), x)[0]
        return rescale_grad(grad, clip_scale=self.clip_scale, **kwargs)


def create_classifier_guider(
    classifier: nn.Module,
    targets: list[int],
    denoiser: nn.Module,
    device: str = "cuda",
    clip_scale: float = 1.0,
    classifier_input_size: int = 224,
) -> BaseGuider:
    """Create appropriate classifier guider based on denoiser type.

    Automatically selects ImageClassifierGuider for pixel-space models (JiT, PixelFlow)
    or LatentClassifierGuider for latent-space models (DiT, SiT).

    Args:
        classifier: Pre-trained classifier model. Should accept (B, 3, H, W) images
            and return (B, num_classes) logits.
        targets: Target class indices for guidance.
        denoiser: Denoiser instance. Used to determine latent vs pixel space
            via `is_latent_diffusion` attribute and to access `vae` for latent models.
        device: Device for computation.
        clip_scale: Maximum gradient norm for clipping.
        classifier_input_size: Input size expected by classifier (default: 224).

    Returns:
        ImageClassifierGuider for pixel-space models (JiT, PixelFlow).
        LatentClassifierGuider for latent-space models (DiT, SiT).

    Raises:
        ValueError: If denoiser is a latent model but doesn't have 'vae' attribute.

    Example:
        >>> from jit_tfg.tfg.guiders import create_classifier_guider
        >>> from jit_tfg.models.jit.denoiser import Denoiser
        >>> import torchvision.models as models
        >>>
        >>> classifier = models.resnet50(pretrained=True)
        >>> denoiser = load_jit_denoiser(...)
        >>> guider = create_classifier_guider(
        ...     classifier=classifier,
        ...     targets=[207, 360],
        ...     denoiser=denoiser,
        ...     device="cuda",
        ... )
    """
    is_latent = getattr(denoiser, "is_latent_diffusion", False)

    if is_latent:
        vae = getattr(denoiser, "vae", None)
        if vae is None:
            raise ValueError(
                "Latent diffusion model must have 'vae' attribute for "
                "LatentClassifierGuider. Ensure the denoiser has a VAEHandler attached."
            )
        return LatentClassifierGuider(
            classifier=classifier,
            vae=vae,
            targets=targets,
            device=device,
            clip_scale=clip_scale,
            classifier_input_size=classifier_input_size,
        )
    else:
        return ImageClassifierGuider(
            classifier=classifier,
            targets=targets,
            device=device,
            clip_scale=clip_scale,
        )


def load_imagenet_classifier(
    model_name: str = "resnet50",
    device: str = "cuda",
    pretrained: bool = True,
) -> nn.Module:
    """Load a pretrained ImageNet classifier.

    Args:
        model_name: Model name from torchvision.models.
        device: Target device.
        pretrained: Whether to load pretrained weights.

    Returns:
        Classifier model in eval mode.
    """
    import torchvision.models as models

    model_fn = getattr(models, model_name, None)
    if model_fn is None:
        raise ValueError(f"Unknown model: {model_name}")

    if pretrained:
        weights = getattr(models, f"{model_name.upper()}_Weights", None)
        if weights is not None:
            model = model_fn(weights=weights.DEFAULT)
        else:
            model = model_fn(pretrained=True)
    else:
        model = model_fn(pretrained=False)

    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    return model
