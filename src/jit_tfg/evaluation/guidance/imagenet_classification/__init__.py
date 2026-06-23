"""ImageNet classification evaluation for guidance validity.

This module provides evaluators for measuring guidance validity on ImageNet
by computing classification accuracy using pre-trained models.

Key Design Choice (from TFG paper):
    - Guide model: google/vit-base-patch16-224 (ViT-B/16)
    - Evaluation model: facebook/deit-small-patch16-224 (DeiT-Small)
    - Using different models prevents over-confidence in validity scores.

Usage:
    from jit_tfg.evaluation.guidance.imagenet_classification import (
        ImageNetClassificationEvaluator,
    )

    evaluator = ImageNetClassificationEvaluator(device="cuda")
    validity = evaluator.compute_validity(images, target_classes=[111, 222])
"""

from jit_tfg.evaluation.guidance.imagenet_classification.evaluator import (
    ImageNetClassificationEvaluator,
)

__all__ = [
    "ImageNetClassificationEvaluator",
]
