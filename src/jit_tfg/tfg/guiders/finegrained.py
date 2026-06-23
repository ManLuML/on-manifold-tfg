"""Fine-grained classifier guiders for TFG guidance.

This module provides guiders for fine-grained classification guidance,
using HuggingFace pre-trained classifiers with specialized class labels
(e.g., bird species).

Reference:
    TFG Paper Appendix D.5: Fine-grained Bird Classification
    Guide Model: dennisjooo/Birds-Classifier-EfficientNetB2 (525 species)
"""

import torch
from torchvision.transforms import Compose, Normalize, Resize
from transformers import AutoModelForImageClassification, AutoProcessor

from jit_tfg.tfg.guiders.base import BaseGuider
from jit_tfg.tfg.utils import check_grad_fn, rescale_grad


class HuggingFaceClassifierGuider(BaseGuider):
    """Guider using a HuggingFace image classifier for fine-grained guidance.

    Wraps AutoModelForImageClassification from HuggingFace Hub, automatically
    loading the appropriate preprocessing from AutoProcessor.

    This is designed for fine-grained classification tasks where the target
    classes are different from ImageNet-1K (e.g., 525 bird species).

    Attributes:
        model: HuggingFace classification model.
        transforms: Preprocessing transforms from AutoProcessor.
        default_targets: Default target class indices.

    Example:
        >>> guider = HuggingFaceClassifierGuider(
        ...     model_name="dennisjooo/Birds-Classifier-EfficientNetB2",
        ...     device="cuda",
        ... )
        >>> x = torch.randn(4, 3, 256, 256, device="cuda", requires_grad=True)
        >>> targets = torch.tensor([111, 222, 333, 444], device="cuda")
        >>> grad = guider.get_guidance(x, targets=targets)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        targets: list[int] | None = None,
        clip_scale: float = 100.0,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the HuggingFace classifier guider.

        Args:
            model_name: HuggingFace model identifier (e.g.,
                "dennisjooo/Birds-Classifier-EfficientNetB2").
            device: Device for computation.
            targets: Default target class indices. If None, targets must be
                provided at inference time via get_guidance(targets=...).
            clip_scale: Maximum gradient norm for clipping.
            cache_dir: Directory to cache downloaded models.
        """
        super().__init__(device)
        self.model_name = model_name
        self.default_targets = targets
        self.clip_scale = clip_scale

        self._load_model(model_name, cache_dir)

    def _load_model(self, model_name: str, cache_dir: str | None) -> None:
        """Load model and processor from HuggingFace Hub.

        Args:
            model_name: HuggingFace model identifier.
            cache_dir: Directory to cache downloaded models.
        """
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = self.model.to(self.device).eval()

        for param in self.model.parameters():
            param.requires_grad = False

        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        self.transforms = Compose([
            Resize([processor.size["height"], processor.size["width"]]),
            Normalize(mean=processor.image_mean, std=processor.image_std),
        ])

        self.num_classes = self.model.config.num_labels

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for the classifier.

        Converts from [-1, 1] range to [0, 1], then applies model-specific
        transforms (resize and normalize).

        Args:
            x: Input images of shape (B, C, H, W) in [-1, 1] range.

        Returns:
            Preprocessed images ready for classifier input.
        """
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        return self.transforms(x)

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
        """Compute guidance gradient from HuggingFace classifier.

        Computes the gradient of log p(y|x) where y is the target class,
        steering generation toward the specified fine-grained class.

        Args:
            x: Input images of shape (B, C, H, W) in [-1, 1] range,
                requiring gradients.
            targets: Per-sample target classes of shape (B,). Each sample
                is guided toward its corresponding target class.
                If None, uses self.default_targets.
            return_logp: If True, return log-probability instead of gradient.
            check_grad: If True, verify x.requires_grad is set.
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

        x_processed = self._preprocess(x)

        logits = self.model(x_processed).logits
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

    def get_label_name(self, label_id: int) -> str:
        """Get the label name for a given label ID.

        Args:
            label_id: Label index.

        Returns:
            Human-readable label name.
        """
        if hasattr(self.model.config, "id2label"):
            return self.model.config.id2label.get(label_id, f"class_{label_id}")
        return f"class_{label_id}"

    def verify_label_compatibility(
        self,
        other_model_name: str,
        sample_labels: list[int] | None = None,
    ) -> dict:
        """Verify that label indices are compatible with another model.

        Useful for ensuring guide and eval models use the same label space.

        Args:
            other_model_name: HuggingFace model name to compare against.
            sample_labels: Optional list of label indices to check.
                If None, checks first 10 labels.

        Returns:
            Dictionary with compatibility information:
                - compatible: Whether labels match
                - mismatches: List of mismatched (index, self_name, other_name)
        """
        from transformers import AutoConfig

        other_config = AutoConfig.from_pretrained(other_model_name)

        if sample_labels is None:
            sample_labels = list(range(min(10, self.num_classes)))

        mismatches = []
        self_id2label = getattr(self.model.config, "id2label", {})
        other_id2label = getattr(other_config, "id2label", {})

        for label_id in sample_labels:
            self_name = self_id2label.get(label_id, f"unknown_{label_id}")
            other_name = other_id2label.get(label_id, f"unknown_{label_id}")
            if self_name != other_name:
                mismatches.append((label_id, self_name, other_name))

        return {
            "compatible": len(mismatches) == 0,
            "mismatches": mismatches,
            "self_num_classes": self.num_classes,
            "other_num_classes": getattr(other_config, "num_labels", "unknown"),
        }
