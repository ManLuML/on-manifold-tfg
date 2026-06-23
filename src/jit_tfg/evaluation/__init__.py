"""Evaluation module for JiT-TFG.

This module provides utilities for evaluating models across different tasks:
- generation: FID, IS metrics for image generation quality
- guidance: Guidance validity metrics (classification accuracy)

Usage:
    # Generation evaluation
    from jit_tfg.evaluation.generation import (
        calculate_fid,
        calculate_inception_score,
        evaluate_generation,
        InceptionFeatureExtractor,
    )

    # Guidance validity evaluation
    from jit_tfg.evaluation.guidance import ImageNetClassificationEvaluator

    # Or import from top-level for convenience
    from jit_tfg.evaluation import evaluate_generation, ImageNetClassificationEvaluator
"""

from jit_tfg.evaluation.generation import (
    InceptionFeatureExtractor,
    calculate_fid,
    calculate_fid_from_folder,
    calculate_inception_score,
    evaluate_generation,
    get_reference_stats_path,
)
from jit_tfg.evaluation.guidance import ImageNetClassificationEvaluator

__all__ = [
    "ImageNetClassificationEvaluator",
    "InceptionFeatureExtractor",
    "calculate_fid",
    "calculate_fid_from_folder",
    "calculate_inception_score",
    "evaluate_generation",
    "get_reference_stats_path",
]
