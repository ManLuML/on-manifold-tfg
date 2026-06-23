"""Fine-grained classification evaluator for TFG guidance validity.

Provides a generic top-1 accuracy evaluator that any fine-grained
benchmark (birds, Stanford Cars, ...) can use by passing its own
HuggingFace model id.

Usage:
    from jit_tfg.evaluation.guidance.finegrained_classification import (
        FinegrainedClassifierEvaluator,    # generic
        FinegrainedBirdEvaluator,          # backward-compatible alias
    )

    evaluator = FinegrainedClassifierEvaluator(
        model_name="tanganke/convnext-base-224_stanford-cars_...",
        device="cuda",
    )
    validity = evaluator.compute_validity_from_folder(...)
"""

from jit_tfg.evaluation.guidance.finegrained_classification.evaluator import (
    FinegrainedBirdEvaluator,
    FinegrainedClassifierEvaluator,
)

__all__ = [
    "FinegrainedBirdEvaluator",
    "FinegrainedClassifierEvaluator",
]
