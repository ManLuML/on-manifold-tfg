"""Transport module for SiT flow matching.

This module provides interpolation paths and integrators for SiT's
flow matching framework.

Main Components:
    - LinearPath: Linear interpolation path (x_t = t*x + (1-t)*ε)
    - ODE/SDE integrators for sampling
"""

from jit_tfg.models.sit.transport.path import LinearPath

__all__ = ["LinearPath"]
