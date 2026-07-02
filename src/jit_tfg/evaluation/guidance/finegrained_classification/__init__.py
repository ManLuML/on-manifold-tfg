"""Fine-grained classification evaluator for TFG guidance validity.

Provides a generic top-1 accuracy evaluator, used by the fine-grained
bird benchmark (143 species mapped from a 525-class classifier). Other
fine-grained domains can use it by passing their own HuggingFace model id.

Usage:
    from jit_tfg.evaluation.guidance.finegrained_classification import (
        FinegrainedClassifierEvaluator,    # generic
        FinegrainedBirdEvaluator,          # backward-compatible alias
    )

    evaluator = FinegrainedClassifierEvaluator(
        model_name="chriamue/bird-species-classifier",
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
