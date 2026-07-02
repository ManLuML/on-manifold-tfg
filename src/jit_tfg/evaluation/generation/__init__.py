"""Generation evaluation module for JiT-TFG.

This module provides utilities for evaluating generative models,
including FID (Frechet Inception Distance) and IS (Inception Score).

Usage:
    from jit_tfg.evaluation.generation import calculate_fid, calculate_inception_score
    from jit_tfg.evaluation.generation import InceptionFeatureExtractor

    # Extract features
    extractor = InceptionFeatureExtractor(device="cuda")
    features = extractor.extract_from_folder("path/to/images")

    # Calculate FID with pre-computed statistics (auto-located)
    from jit_tfg.evaluation.generation import get_reference_stats_path
    fid = calculate_fid(
        generated_features=features,
        reference_stats_path=get_reference_stats_path(img_size=256)
    )

    # Calculate Inception Score
    is_mean, is_std = calculate_inception_score(features)
"""

from jit_tfg.evaluation.generation.fid import calculate_fid, calculate_fid_from_folder
from jit_tfg.evaluation.generation.inception import InceptionFeatureExtractor
from jit_tfg.evaluation.generation.metrics import (
    calculate_class_fid,
    calculate_fid_from_features,
    calculate_inception_score,
    calculate_inception_score_from_logits,
    evaluate_generation,
    get_class_stats_path,
    get_finegrained_stats_path,
    get_reference_stats_path,
)

__all__ = [
    "InceptionFeatureExtractor",
    "calculate_class_fid",
    "calculate_fid",
    "calculate_fid_from_features",
    "calculate_fid_from_folder",
    "calculate_inception_score",
    "calculate_inception_score_from_logits",
    "evaluate_generation",
    "get_class_stats_path",
    "get_finegrained_stats_path",
    "get_reference_stats_path",
]
