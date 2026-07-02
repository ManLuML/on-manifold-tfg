"""Guidance evaluation module for TFG experiments.

This module provides utilities for evaluating guidance quality:
- imagenet_classification: Guidance validity using pre-trained classifiers
- finegrained_classification: Fine-grained bird species validity

Usage:
    from jit_tfg.evaluation.guidance import ImageNetClassificationEvaluator
    from jit_tfg.evaluation.guidance import FinegrainedBirdEvaluator

    evaluator = ImageNetClassificationEvaluator(device="cuda")
    validity = evaluator.compute_validity(images, target_classes)
"""

from jit_tfg.evaluation.guidance.finegrained_classification import (
    FinegrainedBirdEvaluator,
    FinegrainedClassifierEvaluator,
)
from jit_tfg.evaluation.guidance.imagenet_classification import (
    ImageNetClassificationEvaluator,
)

__all__ = [
    "FinegrainedBirdEvaluator",
    "FinegrainedClassifierEvaluator",
    "ImageNetClassificationEvaluator",
]
