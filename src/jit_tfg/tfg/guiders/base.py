"""Base class for guidance energy functions.

A guider computes an energy function E(x) given a sample, typically returning
the log-probability or gradient for steering diffusion sampling.

This is adapted from the original TFG implementation (tasks/base.py) for
use with the JiT Flow Matching framework.
"""

from abc import ABC, abstractmethod

import torch


class BaseGuider(ABC):
    """Abstract base class for guidance energy functions.

    A guider wraps an energy function (e.g., classifier, reward model) and
    provides methods to compute guidance signals for TFG.

    Subclasses must implement:
        - get_guidance: Compute guidance gradient or log-probability

    Attributes:
        device: Device for computation.
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize the guider.

        Args:
            device: Device for computation.
        """
        self.device = device

    @abstractmethod
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
        """Compute guidance signal from input.

        Args:
            x: Input tensor requiring gradients. Shape depends on domain:
                - Images: (B, C, H, W)
                - Molecules: (B, N, D) where N is nodes, D is features
            targets: Optional per-sample target tensor of shape (B,).
                If provided, overrides the default targets set at initialization.
            return_logp: If True, return log-probability instead of gradient.
            check_grad: If True, verify x.requires_grad is set.
            **kwargs: Additional domain-specific arguments.

        Returns:
            If return_logp is True:
                Log-probability tensor of shape (B,).
            Else:
                Gradient tensor of same shape as input x.

        Raises:
            AssertionError: If check_grad is True and x.requires_grad is False.
        """
        pass
