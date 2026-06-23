"""Model utility functions for positional embeddings and normalization.

This module provides essential building blocks for transformer-based
diffusion models:

1. **Rotary Position Embedding (RoPE)**: 2D positional encoding for
   vision transformers, encoding spatial positions in attention.

2. **RMSNorm**: Root Mean Square Layer Normalization, a simpler and
   sometimes more effective alternative to LayerNorm.

3. **Sinusoidal Position Embedding**: 2D positional embeddings using
   sine/cosine functions (fixed, not learned).

RoPE Background:
    Rotary Position Embedding encodes positions by rotating query and
    key vectors in the attention mechanism. For 2D vision:
    - Separate frequencies for height and width dimensions
    - Concatenated to form full rotary embedding
    - Applied via element-wise multiplication with cos/sin terms

References:
    - RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    - Lightning-DiT: https://github.com/hustvl/LightningDiT
"""

from __future__ import annotations

from math import pi

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn


def broadcat(tensors: list, dim: int = -1) -> torch.Tensor:
    """Concatenate tensors with automatic broadcasting.

    Broadcasts tensors to compatible shapes before concatenation.
    All tensors must have the same number of dimensions, and for each
    dimension (except the concatenation dimension), sizes must be
    either equal or 1 (for broadcasting).

    Args:
        tensors: List of tensors to concatenate. Must all have the
            same number of dimensions.
        dim: Dimension along which to concatenate. Default: -1 (last).

    Returns:
        Concatenated tensor after broadcasting all inputs.

    Raises:
        ValueError: If tensors have different numbers of dimensions,
            or if dimensions are not broadcastable.

    Example:
        >>> a = torch.randn(16, 1, 32)   # Shape: (16, 1, 32)
        >>> b = torch.randn(1, 16, 32)   # Shape: (1, 16, 32)
        >>> c = broadcat([a, b], dim=-1)  # Shape: (16, 16, 64)
    """
    num_tensors = len(tensors)
    shape_lens = {len(t.shape) for t in tensors}

    if len(shape_lens) != 1:
        msg = "tensors must all have the same number of dimensions"
        raise ValueError(msg)

    shape_len = next(iter(shape_lens))
    dim = (dim + shape_len) if dim < 0 else dim  # Handle negative indexing

    # Collect sizes for each dimension
    dims = list(zip(*(list(t.shape) for t in tensors), strict=False))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]

    # Check broadcastability: each dim must have at most 2 unique sizes (size, 1)
    if not all(len(set(t[1])) <= 2 for t in expandable_dims):
        msg = "invalid dimensions for broadcastable concatentation"
        raise ValueError(msg)

    # Compute broadcast target shapes
    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*(t[1] for t in expanded_dims), strict=False))

    # Expand and concatenate
    tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes, strict=False)]
    return torch.cat(tensors, dim=dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate tensor pairs for rotary position embedding.

    Splits the last dimension into pairs (d, r=2) and rotates each pair
    by swapping and negating: (x1, x2) -> (-x2, x1).

    This is a key operation in RoPE, where rotation is applied via:
        x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)

    Tensor Flow:
        Input:  (..., D) where D is even
                    │
                    ▼
        Rearrange: (..., D/2, 2)
                    │
                    ▼
        Split and swap: x1, x2 = unbind → stack(-x2, x1)
                    │
                    ▼
        Output: (..., D)

    Args:
        x: Input tensor with even-sized last dimension.
            Shape: (..., D) where D % 2 == 0.

    Returns:
        Rotated tensor of same shape. For each pair of elements
        (a, b) in the input, output is (-b, a).

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> rotate_half(x)
        tensor([-2., 1., -4., 3.])
    """
    # Reshape to pair elements: (..., D) -> (..., D/2, 2)
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    # Split into pairs
    x1, x2 = x.unbind(dim=-1)
    # Rotate: (x1, x2) -> (-x2, x1)
    x = torch.stack((-x2, x1), dim=-1)
    # Reshape back: (..., D/2, 2) -> (..., D)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbedding(nn.Module):
    """2D Rotary Position Embedding for vision transformers.

    Computes 2D RoPE by creating separate frequency components for
    height and width, then concatenating them. Supports:
    - Different frequency generation strategies (language, pixel, constant)
    - Fine-tuning at different sequence lengths than pre-training

    Frequency Computation (for 'lang' mode):
        freq_i = 1 / (theta^(2i/dim)), i = 0, 1, ..., dim/2 - 1

    Position Encoding:
        For position (h, w), the encoding combines:
        - cos/sin(h * freq) for height positions
        - cos/sin(w * freq) for width positions

    Attributes:
        freqs_cos: Precomputed cosine terms for all positions.
            Shape: (H, W, D) where D is the embedding dimension.
        freqs_sin: Precomputed sine terms for all positions.
    """

    def __init__(
        self,
        dim: int,
        pt_seq_len: int,
        ft_seq_len: int | None = None,
        custom_freqs: torch.Tensor | None = None,
        freqs_for: str = "lang",
        theta: int = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
    ) -> None:
        """Initialize the 2D rotary embedding.

        Args:
            dim: Embedding dimension (must be even).
            pt_seq_len: Pre-training sequence length (grid size).
            ft_seq_len: Fine-tuning sequence length. Default: same as pt_seq_len.
            custom_freqs: Optional custom frequency tensor.
            freqs_for: Frequency generation strategy:
                - "lang": Use language model frequencies (default)
                - "pixel": Linear frequencies from 1 to max_freq/2
                - "constant": All frequencies = 1
            theta: Base for exponential frequency decay (for "lang").
            max_freq: Maximum frequency for "pixel" mode.
            num_freqs: Number of frequencies for "constant" mode.
        """
        super().__init__()

        # Generate frequencies based on strategy
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            # Standard Transformer frequencies: 1/θ^(2i/d)
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            # Linear frequencies for pixel-level features
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            # All frequencies equal (for certain experiments)
            freqs = torch.ones(num_freqs).float()
        else:
            msg = f"unknown modality {freqs_for}"
            raise ValueError(msg)

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        # Create position indices (adjusted for fine-tuning length)
        # t: (ft_seq_len,) normalized to [0, pt_seq_len)
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        # Compute 1D frequencies for height: (ft_seq_len, dim/2)
        freqs_h = torch.einsum("..., f -> ... f", t, freqs)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)  # Duplicate for (cos, sin)

        # Compute 1D frequencies for width: (ft_seq_len, dim/2)
        freqs_w = torch.einsum("..., f -> ... f", t, freqs)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        # Combine into 2D: (H, W, D)
        # Height varies along first dim, width along second
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        # Precompute cos and sin for efficiency
        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t: torch.Tensor, start_index: int = 0) -> torch.Tensor:
        """Apply rotary embedding to input tensor.

        Args:
            t: Input tensor of shape (..., D) where D >= rot_dim.
            start_index: Starting index for applying rotation.
                Useful for applying RoPE to a subset of dimensions.

        Returns:
            Tensor with rotary embedding applied, same shape as input.

        Raises:
            ValueError: If input dimension is too small for rotation.
        """
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim

        if rot_dim > t.shape[-1]:
            msg = f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
            raise ValueError(msg)

        # Split: [before_rotate | rotate | after_rotate]
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]

        # Apply rotation: x * cos + rotate_half(x) * sin
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)

        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    """Optimized 2D Rotary Position Embedding for vision transformers.

    A faster version that precomputes flattened cos/sin tables.
    Supports optional in-context/CLS tokens that don't receive rotation.

    Key Optimization:
        Instead of computing 2D grid indices at runtime, precomputes
        a flattened (N, D) lookup table where N = H * W.

    In-Context Token Handling:
        When num_cls_token > 0, prepends identity rotation (cos=1, sin=0)
        for CLS tokens, so they don't receive positional encoding.

    Attributes:
        freqs_cos: Precomputed cosine terms. Shape: (N, D) where
            N = num_cls_token + H * W.
        freqs_sin: Precomputed sine terms. Shape: (N, D).
    """

    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 16,
        ft_seq_len: int | None = None,
        custom_freqs: torch.Tensor | None = None,
        freqs_for: str = "lang",
        theta: int = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
        num_cls_token: int = 0,
    ) -> None:
        """Initialize the fast 2D rotary embedding.

        Args:
            dim: Embedding dimension per head (half-head-dim for RoPE).
            pt_seq_len: Pre-training grid size (H=W). Default: 16.
            ft_seq_len: Fine-tuning grid size. Default: same as pt_seq_len.
            custom_freqs: Optional custom frequency tensor.
            freqs_for: Frequency generation strategy ("lang", "pixel", "constant").
            theta: Base for exponential frequency decay (for "lang").
            max_freq: Maximum frequency for "pixel" mode.
            num_freqs: Number of frequencies for "constant" mode.
            num_cls_token: Number of CLS/in-context tokens to prepend.
                These receive identity rotation (no position encoding).
        """
        super().__init__()

        # Generate frequencies (same logic as VisionRotaryEmbedding)
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            msg = f"unknown modality {freqs_for}"
            raise ValueError(msg)

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        # Position indices adjusted for fine-tuning
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        # Compute 1D frequencies and create 2D grid
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if num_cls_token > 0:
            # Flatten 2D grid to 1D: (H, W, D) -> (H*W, D)
            freqs_flat = freqs.view(-1, freqs.shape[-1])
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # Create identity rotation for CLS tokens: cos=1, sin=0
            _, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            # Prepend CLS tokens: (num_cls + H*W, D)
            self.register_buffer("freqs_cos", torch.cat([cos_pad, cos_img], dim=0))
            self.register_buffer("freqs_sin", torch.cat([sin_pad, sin_img], dim=0))
        else:
            # No CLS tokens: just flatten 2D to 1D
            self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]))
            self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to input tensor.

        Fast version that uses precomputed flattened lookup tables.

        Args:
            t: Input tensor of shape (B, H, N, D) where:
                - B: Batch size
                - H: Number of attention heads
                - N: Sequence length (num_cls + H*W)
                - D: Head dimension (should match embedding dim)

        Returns:
            Tensor with rotary embedding applied, same shape as input.

        Note:
            The freqs tensors broadcast across batch and head dimensions.
        """
        # Apply rotation: t * cos + rotate_half(t) * sin
        # freqs_cos, freqs_sin: (N, D) broadcast to (B, H, N, D)
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A simplified normalization layer that normalizes by RMS instead of
    mean and variance. Often performs comparably to LayerNorm with
    slightly lower computational cost.

    Formula:
        output = (x / RMS(x)) * weight
        where RMS(x) = sqrt(mean(x^2) + eps)

    Unlike LayerNorm:
        - No mean centering (no bias subtraction)
        - Only rescales by learned weight (no learned bias)

    Attributes:
        weight: Learnable scale parameter of shape (hidden_size,).
        variance_epsilon: Small constant for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm.

        Args:
            hidden_size: Size of the last dimension to normalize over.
            eps: Small constant added to variance for numerical stability.
                Default: 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            hidden_states: Input tensor of shape (..., hidden_size).
                The last dimension is normalized.

        Returns:
            Normalized tensor of same shape as input.
        """
        input_dtype = hidden_states.dtype
        # Compute variance in float32 for stability
        hidden_states = hidden_states.to(torch.float32)
        # Compute mean of squares (no mean centering)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Normalize by RMS: x / sqrt(var + eps)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Apply learned scale and restore dtype
        return (self.weight * hidden_states).to(input_dtype)


# =============================================================================
# Sinusoidal Position Embedding Functions
# =============================================================================


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0
) -> np.ndarray:
    """Create 2D sinusoidal positional embedding for a square grid.

    Generates fixed (non-learned) positional embeddings using sine and
    cosine functions at different frequencies, similar to the original
    Transformer paper but extended to 2D.

    The embedding combines separate encodings for height and width:
        pos_embed = [height_embed, width_embed]

    Each dimension uses half of embed_dim.

    Args:
        embed_dim: Total embedding dimension. Must be divisible by 2.
        grid_size: Size of the square grid (height = width = grid_size).
        cls_token: Whether to include a position for CLS token.
        extra_tokens: Number of extra tokens (e.g., CLS) to prepend.
            Their embeddings are set to zero.

    Returns:
        Positional embeddings as numpy array of shape:
            - (grid_size * grid_size, embed_dim) if no extra tokens
            - (extra_tokens + grid_size * grid_size, embed_dim) with extra tokens

    Example:
        >>> embed = get_2d_sincos_pos_embed(768, grid_size=16)
        >>> embed.shape
        (256, 768)  # 16*16 = 256 positions
    """
    # Create 2D coordinate grid
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first convention
    grid = np.stack(grid, axis=0)  # Shape: (2, H, W)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    # Optionally prepend zeros for extra tokens (e.g., CLS)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Create 2D sinusoidal embedding from a coordinate grid.

    Splits embed_dim in half: first half encodes height (grid[0]),
    second half encodes width (grid[1]).

    Args:
        embed_dim: Total embedding dimension. Must be divisible by 2.
        grid: Coordinate grid of shape (2, 1, H, W) where:
            - grid[0]: Height coordinates
            - grid[1]: Width coordinates

    Returns:
        Positional embeddings of shape (H*W, embed_dim).

    Raises:
        ValueError: If embed_dim is not divisible by 2.
    """
    if embed_dim % 2 != 0:
        msg = "embed_dim must be divisible by 2"
        raise ValueError(msg)

    # Encode height and width separately, each using half the dimensions
    # emb_h: (H*W, embed_dim/2) - height encoding
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    # emb_w: (H*W, embed_dim/2) - width encoding
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    # Concatenate: (H*W, embed_dim)
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Create 1D sinusoidal embedding for a set of positions.

    Uses the Transformer positional encoding formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    This creates a unique encoding for each position with smoothly
    varying patterns that allow the model to learn relative positions.

    Args:
        embed_dim: Output dimension for each position. Must be even.
        pos: Array of positions to encode. Shape: can be any, will be
            flattened. Output shape: (num_positions, embed_dim).

    Returns:
        Positional embeddings of shape (M, embed_dim) where M is the
        number of positions in pos.

    Raises:
        ValueError: If embed_dim is not divisible by 2.

    Example:
        >>> pos = np.array([0, 1, 2, 3])
        >>> embed = get_1d_sincos_pos_embed_from_grid(64, pos)
        >>> embed.shape
        (4, 64)
    """
    if embed_dim % 2 != 0:
        msg = "embed_dim must be divisible by 2"
        raise ValueError(msg)

    # Compute frequency terms: (embed_dim/2,)
    # omega_i = 1 / 10000^(2i/embed_dim)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    # Flatten positions: (M,)
    pos = pos.reshape(-1)

    # Outer product: (M,) × (D/2,) -> (M, D/2)
    out = np.einsum("m,d->md", pos, omega)

    # Apply sin and cos
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    # Concatenate: (M, D)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
