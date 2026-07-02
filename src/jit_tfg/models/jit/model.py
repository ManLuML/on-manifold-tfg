"""JiT (Just image Transformer) model architecture for image generation.

This module implements the JiT transformer architecture, a diffusion model
backbone designed for high-quality image generation. The architecture combines
several modern techniques:

1. **Bottleneck Patch Embedding**: Two-stage convolution for efficient patch
   embedding with dimensionality reduction.

2. **Adaptive Layer Normalization (adaLN)**: Modulates layer outputs based on
   timestep and class conditioning, enabling dynamic feature adjustment.

3. **Rotary Position Embedding (RoPE)**: 2D vision-specific rotary embeddings
   for encoding spatial positions in the attention mechanism.

4. **In-Context Conditioning**: Injects class information as learnable tokens
   partway through the transformer blocks for enhanced conditioning.

5. **SwiGLU FFN**: Swish-gated linear units for improved feed-forward networks.

Architecture Overview:
    Input Image (B, 3, H, W)
         │
         ▼
    Patch Embedding → (B, num_patches, hidden_size)
         │
         ▼
    + Positional Embedding
         │
         ▼
    ┌─── Transformer Blocks (first half) ───┐
    │    - Multi-head Self-Attention + RoPE │
    │    - SwiGLU FFN                        │
    │    - adaLN modulation                  │
    └────────────────────────────────────────┘
         │
         ▼
    + In-Context Class Tokens (injected at in_context_start)
         │
         ▼
    ┌─── Transformer Blocks (second half) ──┐
    │    - Multi-head Self-Attention + RoPE │
    │    - SwiGLU FFN                        │
    │    - adaLN modulation                  │
    └────────────────────────────────────────┘
         │
         ▼
    Remove In-Context Tokens
         │
         ▼
    Final Layer (adaLN + Linear)
         │
         ▼
    Unpatchify → Output Image (B, 3, H, W)

References:
    - JiT Paper: "Back to Basics: Let Denoising Generative Models Denoise"
    - SiT: https://github.com/willisma/SiT
    - Lightning-DiT: https://github.com/hustvl/LightningDiT
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from jit_tfg.models.jit.utils.model_util import RMSNorm, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization (adaLN) modulation.

    Modulates the input tensor using learned scale and shift parameters,
    enabling the model to dynamically adjust layer outputs based on
    conditioning information (timestep and class).

    The modulation follows the formula:
        output = x * (1 + scale) + shift

    Args:
        x: Input tensor of shape (B, L, D) where:
            - B: Batch size
            - L: Sequence length (number of patches)
            - D: Hidden dimension
        shift: Shift parameter of shape (B, D) for additive modulation.
        scale: Scale parameter of shape (B, D) for multiplicative modulation.

    Returns:
        Modulated tensor of shape (B, L, D), same as input.

    Example:
        >>> x = torch.randn(2, 256, 768)      # (B=2, L=256, D=768)
        >>> shift = torch.randn(2, 768)       # (B=2, D=768)
        >>> scale = torch.randn(2, 768)       # (B=2, D=768)
        >>> out = modulate(x, shift, scale)   # (B=2, L=256, D=768)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """Two-stage bottleneck patch embedding for image-to-token conversion.

    Converts input images into a sequence of patch embeddings using a two-stage
    convolution approach. The first stage reduces dimensionality to a bottleneck,
    and the second stage expands to the target embedding dimension. This design
    reduces computation while preserving important features.

    Tensor Flow:
        Input:  (B, in_chans, H, W)
                     │
                     ▼
        Conv2d (patch_size kernel, stride) → (B, pca_dim, H/P, W/P)
                     │
                     ▼
        Conv2d (1x1 kernel) → (B, embed_dim, H/P, W/P)
                     │
                     ▼
        Flatten + Transpose → (B, num_patches, embed_dim)

    Attributes:
        img_size: Tuple of (height, width) for input images.
        patch_size: Tuple of (patch_h, patch_w) for patch extraction.
        num_patches: Total number of patches = (H/P) * (W/P).
        proj1: First convolution layer (bottleneck reduction).
        proj2: Second convolution layer (dimension expansion).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pca_dim: int = 768,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        """Initialize the bottleneck patch embedding layer.

        Args:
            img_size: Input image size (assumed square). Default: 224.
            patch_size: Size of each patch (assumed square). Default: 16.
            in_chans: Number of input channels. Default: 3 (RGB).
            pca_dim: Bottleneck dimension (intermediate feature dim). Default: 768.
            embed_dim: Output embedding dimension. Default: 768.
            bias: Whether to use bias in the second projection. Default: True.
        """
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # First projection: (B, in_chans, H, W) -> (B, pca_dim, H/P, W/P)
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # Second projection: (B, pca_dim, H/P, W/P) -> (B, embed_dim, H/P, W/P)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input images to patch embeddings.

        Args:
            x: Input images of shape (B, C, H, W) where:
                - B: Batch size
                - C: Number of channels (must match in_chans)
                - H: Image height (must match img_size[0])
                - W: Image width (must match img_size[1])

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim) where:
                - num_patches = (H // patch_size) * (W // patch_size)

        Raises:
            ValueError: If input image size doesn't match expected img_size.
        """
        _, _, H, W = x.shape
        if self.img_size[0] != H or self.img_size[1] != W:
            msg = f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            raise ValueError(msg)

        # Two-stage projection with flatten and transpose
        # (B, C, H, W) -> (B, pca_dim, H/P, W/P) -> (B, embed_dim, H/P, W/P)
        # -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations.

    Uses sinusoidal positional encoding (similar to Transformer positional
    embeddings) followed by an MLP to create rich timestep representations
    that condition the denoising process.

    Tensor Flow:
        Input timestep: (B,) scalar values in [0, 1]
              │
              ▼
        Sinusoidal Encoding → (B, frequency_embedding_size)
              │
              ▼
        MLP: Linear → SiLU → Linear → (B, hidden_size)

    Attributes:
        mlp: Two-layer MLP with SiLU activation.
        frequency_embedding_size: Dimension of sinusoidal encoding.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        """Initialize the timestep embedder.

        Args:
            hidden_size: Output dimension of the timestep embedding.
            frequency_embedding_size: Dimension of intermediate sinusoidal
                encoding. Default: 256.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Generates positional encodings using sine and cosine functions at
        different frequencies, similar to the original Transformer paper.

        The encoding for position t and dimension i is:
            PE(t, 2i)   = sin(t / max_period^(2i/dim))
            PE(t, 2i+1) = cos(t / max_period^(2i/dim))

        Args:
            t: 1-D Tensor of N timestep indices, one per batch element.
                Shape: (N,). Values may be fractional (continuous timesteps).
            dim: The dimension of the output embedding.
            max_period: Controls the minimum frequency of the embeddings.
                Larger values → lower frequencies. Default: 10000.

        Returns:
            Positional embeddings of shape (N, dim).

        References:
            https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        half = dim // 2
        # Compute frequency bands: (half,)
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        # Compute arguments for sin/cos: (N, half)
        args = t[:, None].float() * freqs[None]
        # Concatenate sin and cos embeddings: (N, dim)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Pad with zeros if dim is odd
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps into vector representations.

        Args:
            t: Timestep values of shape (B,) where B is batch size.
                Values are typically in [0, 1] for flow matching.

        Returns:
            Timestep embeddings of shape (B, hidden_size).
        """
        # Create sinusoidal encoding: (B,) -> (B, frequency_embedding_size)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # Project through MLP: (B, frequency_embedding_size) -> (B, hidden_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Provides learnable embeddings for class-conditional generation. Includes
    an extra embedding for the "null" class (index = num_classes) used during
    classifier-free guidance when labels are dropped.

    Attributes:
        embedding_table: Embedding layer with (num_classes + 1) entries.
        num_classes: Number of actual classes (excluding null class).
    """

    def __init__(self, num_classes: int, hidden_size: int) -> None:
        """Initialize the label embedder.

        Args:
            num_classes: Number of classes in the dataset (e.g., 1000 for ImageNet).
            hidden_size: Dimension of the class embeddings.
        """
        super().__init__()
        # +1 for the null class embedding (used in classifier-free guidance)
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Embed class labels.

        Args:
            labels: Class indices of shape (B,) where B is batch size.
                Values in [0, num_classes]. Index num_classes represents
                the null/unconditional class for classifier-free guidance.

        Returns:
            Class embeddings of shape (B, hidden_size).
        """
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout_p: float = 0.0
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Implements the standard attention mechanism:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    The attention computation is performed in float32 for numerical stability,
    while the output maintains the input dtype.

    Tensor Flow:
        Q: (B, H, L, D)    K: (B, H, S, D)    V: (B, H, S, D)
              │                  │                  │
              └────── Q @ K^T ───┘                  │
                       │                            │
                       ▼                            │
              (B, H, L, S) × scale                  │
                       │                            │
                       ▼                            │
                   softmax                          │
                       │                            │
                       ▼                            │
              attention_weights @ V ────────────────┘
                       │
                       ▼
              Output: (B, H, L, D)

    Args:
        query: Query tensor of shape (B, H, L, D) where:
            - B: Batch size
            - H: Number of attention heads
            - L: Query sequence length
            - D: Head dimension
        key: Key tensor of shape (B, H, S, D) where S is key sequence length.
        value: Value tensor of shape (B, H, S, D).
        dropout_p: Dropout probability for attention weights. Default: 0.0.

    Returns:
        Attention output of shape (B, H, L, D).
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))

    # Initialize attention bias: (B, 1, L, S)
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)

    # Compute attention scores in float32 for stability
    # (B, H, L, D) @ (B, H, D, S) -> (B, H, L, S)
    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    # Apply attention to values: (B, H, L, S) @ (B, H, S, D) -> (B, H, L, D)
    return attn_weight @ value


class Attention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE).

    Implements multi-head attention with:
    - QK normalization using RMSNorm for training stability
    - 2D Rotary Position Embedding for spatial position encoding
    - Flexible dropout for regularization

    Tensor Flow:
        Input: (B, L, D)
            │
            ▼
        QKV Linear → (B, L, 3*D)
            │
            ▼
        Reshape → Q, K, V each (B, H, L, D/H)
            │
            ▼
        QK Norm (RMSNorm per head)
            │
            ▼
        Apply RoPE to Q and K
            │
            ▼
        Scaled Dot-Product Attention
            │
            ▼
        Reshape → (B, L, D)
            │
            ▼
        Output Projection → (B, L, D)

    Attributes:
        num_heads: Number of attention heads.
        q_norm: RMSNorm for query normalization.
        k_norm: RMSNorm for key normalization.
        qkv: Linear projection for query, key, value (combined).
        attn_drop: Dropout for attention weights.
        proj: Output projection layer.
        proj_drop: Dropout after output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Initialize the attention layer.

        Args:
            dim: Input and output dimension.
            num_heads: Number of attention heads. Default: 8.
            qkv_bias: Whether to include bias in QKV projection. Default: True.
            qk_norm: Whether to apply RMSNorm to Q and K. Default: True.
            attn_drop: Dropout rate for attention weights. Default: 0.0.
            proj_drop: Dropout rate after output projection. Default: 0.0.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        # Combined QKV projection: (B, L, D) -> (B, L, 3*D)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # Output projection: (B, L, D) -> (B, L, D)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        """Apply multi-head self-attention with RoPE.

        Args:
            x: Input tensor of shape (B, L, D) where:
                - B: Batch size
                - L: Sequence length (number of patches + optional cls tokens)
                - D: Hidden dimension
            rope: Vision Rotary Position Embedding module to apply to Q and K.

        Returns:
            Output tensor of shape (B, L, D), same as input.
        """
        B, N, C = x.shape

        # Project to QKV: (B, N, D) -> (B, N, 3*D) -> (B, N, 3, H, D/H) -> (3, B, H, N, D/H)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, D/H)

        # Apply QK normalization for training stability
        q = self.q_norm(q)  # (B, H, N, D/H)
        k = self.k_norm(k)  # (B, H, N, D/H)

        # Apply 2D Rotary Position Embedding
        q = rope(q)  # (B, H, N, D/H)
        k = rope(k)  # (B, H, N, D/H)

        # Compute attention: (B, H, N, D/H) -> (B, H, N, D/H)
        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        # Reshape back: (B, H, N, D/H) -> (B, N, H, D/H) -> (B, N, D)
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Implements the SwiGLU activation function within a feed-forward network,
    which has been shown to improve training stability and model quality
    compared to standard GELU or ReLU activations.

    SwiGLU splits the intermediate representation into two parts:
    - One part goes through SiLU (Swish) activation
    - The other part acts as a gate
    - The two are multiplied element-wise

    Formula:
        SwiGLU(x) = SiLU(W1 @ x) * (W2 @ x)
        output = W3 @ SwiGLU(x)

    Tensor Flow:
        Input: (B, L, D)
            │
            ▼
        Linear → (B, L, 2 * hidden_dim)
            │
            ▼
        Split → x1: (B, L, hidden_dim), x2: (B, L, hidden_dim)
            │
            ▼
        SiLU(x1) * x2 → (B, L, hidden_dim)
            │
            ▼
        Dropout + Linear → (B, L, D)

    Attributes:
        w12: Combined linear layer for SiLU and gate paths.
        w3: Output projection layer.
        ffn_dropout: Dropout applied after activation.
    """

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True) -> None:
        """Initialize the SwiGLU FFN.

        Args:
            dim: Input and output dimension.
            hidden_dim: Intermediate hidden dimension (before 2/3 reduction).
            drop: Dropout rate. Default: 0.0.
            bias: Whether to use bias in linear layers. Default: True.

        Note:
            The actual hidden dimension is reduced to 2/3 of the specified
            hidden_dim for computational efficiency while maintaining capacity.
        """
        super().__init__()
        # Reduce hidden_dim by 2/3 to match parameter count of standard FFN
        hidden_dim = int(hidden_dim * 2 / 3)
        # Combined projection for SiLU and gate paths: (B, L, D) -> (B, L, 2*hidden)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        # Output projection: (B, L, hidden) -> (B, L, D)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU feed-forward transformation.

        Args:
            x: Input tensor of shape (B, L, D).

        Returns:
            Output tensor of shape (B, L, D), same as input.
        """
        # Project and split: (B, L, D) -> (B, L, 2*hidden) -> x1, x2: (B, L, hidden)
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        # SwiGLU: SiLU(x1) * x2
        hidden = F.silu(x1) * x2
        # Output projection with dropout
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """Final output layer of JiT with adaptive layer normalization.

    Applies the final transformation from hidden representations to patch
    predictions, using adaLN modulation for conditioning.

    Tensor Flow:
        Input: x (B, L, hidden_size), c (B, hidden_size)
            │
            ▼
        adaLN Modulation: c → shift, scale
            │
            ▼
        RMSNorm(x) * (1 + scale) + shift
            │
            ▼
        Linear → (B, L, patch_size^2 * out_channels)

    Attributes:
        norm_final: RMSNorm for pre-projection normalization.
        linear: Final linear projection to patch pixels.
        adaLN_modulation: MLP to generate shift and scale from conditioning.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        """Initialize the final layer.

        Args:
            hidden_size: Input hidden dimension.
            patch_size: Size of output patches (spatial dimension).
            out_channels: Number of output channels (e.g., 3 for RGB).
        """
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        # Project to patch pixels: (B, L, hidden) -> (B, L, P*P*C)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # Generate shift and scale for adaLN: (B, hidden) -> (B, 2*hidden)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    @torch.compile
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply final layer transformation.

        Args:
            x: Hidden representations of shape (B, L, hidden_size).
            c: Conditioning vector of shape (B, hidden_size).
                Contains combined timestep and class information.

        Returns:
            Patch predictions of shape (B, L, patch_size^2 * out_channels).
        """
        # Generate modulation parameters: (B, hidden) -> (B, hidden), (B, hidden)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # Apply modulation and normalization: (B, L, hidden)
        x = modulate(self.norm_final(x), shift, scale)
        # Project to output: (B, L, hidden) -> (B, L, P*P*C)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    """Transformer block for JiT with adaLN-Zero.

    A single transformer block combining:
    - Multi-head self-attention with RoPE
    - SwiGLU feed-forward network
    - adaLN-Zero modulation for both sub-layers

    adaLN-Zero applies separate shift, scale, and gate parameters to each
    sub-layer, allowing fine-grained conditioning control. The gates are
    initialized to zero, enabling residual connections to dominate early
    in training.

    Tensor Flow:
        Input: x (B, L, D), c (B, D)
            │
            ▼
        c → adaLN_modulation → 6 params: shift_msa, scale_msa, gate_msa,
                                          shift_mlp, scale_mlp, gate_mlp
            │
            ▼
        x + gate_msa * Attention(modulate(RMSNorm(x), shift_msa, scale_msa))
            │
            ▼
        x + gate_mlp * MLP(modulate(RMSNorm(x), shift_mlp, scale_mlp))
            │
            ▼
        Output: (B, L, D)

    Attributes:
        norm1: RMSNorm before attention.
        attn: Multi-head attention with RoPE.
        norm2: RMSNorm before MLP.
        mlp: SwiGLU feed-forward network.
        adaLN_modulation: MLP to generate 6 modulation parameters from c.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0, proj_drop: float = 0.0
    ) -> None:
        """Initialize a JiT transformer block.

        Args:
            hidden_size: Hidden dimension size.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension = hidden_size * mlp_ratio. Default: 4.0.
            attn_drop: Dropout rate for attention weights. Default: 0.0.
            proj_drop: Dropout rate for projections. Default: 0.0.
        """
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        # Generate 6 modulation params: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    @torch.compile
    def forward(self, x: torch.Tensor, c: torch.Tensor, feat_rope: VisionRotaryEmbeddingFast = None) -> torch.Tensor:
        """Apply transformer block with adaLN modulation.

        Args:
            x: Input tensor of shape (B, L, D).
            c: Conditioning vector of shape (B, D).
            feat_rope: Rotary position embedding module. Optional.

        Returns:
            Output tensor of shape (B, L, D), same as input.
        """
        # Generate all 6 modulation parameters from conditioning
        # (B, D) -> (B, 6*D) -> 6 × (B, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention branch with adaLN modulation
        # gate_msa controls residual contribution (initialized near zero)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)

        # MLP branch with adaLN modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """Just image Transformer (JiT) for diffusion-based image generation.

    JiT is a transformer architecture designed for image generation with
    diffusion/flow matching. Key features include:

    1. **Bottleneck Patch Embedding**: Efficient image tokenization with
       dimensionality reduction.

    2. **In-Context Conditioning**: Class embeddings are injected as learnable
       tokens partway through the network, providing enhanced conditioning.

    3. **RoPE**: 2D Rotary Position Embeddings encode spatial relationships.

    4. **adaLN-Zero**: Adaptive layer normalization with zero-initialized gates
       for stable training.

    5. **Target-Agnostic Output**: The model outputs a tensor of the same shape
       as the input, which can be interpreted as noise, velocity, or clean data
       depending on the training objective.

    Complete Tensor Flow:
        Input: x (B, 3, H, W), t (B,), y (B,)

        1. Embeddings:
           - t_emb = TimestepEmbedder(t)      → (B, hidden_size)
           - y_emb = LabelEmbedder(y)         → (B, hidden_size)
           - c = t_emb + y_emb                → (B, hidden_size)

        2. Patch Embedding:
           - x = PatchEmbed(x)                → (B, num_patches, hidden_size)
           - x = x + pos_embed                → (B, num_patches, hidden_size)

        3. First Half Blocks (0 to in_context_start-1):
           - For each block: x = block(x, c, feat_rope)

        4. In-Context Token Injection (at in_context_start):
           - in_context = y_emb + in_context_posemb  → (B, in_context_len, hidden_size)
           - x = concat([in_context, x])             → (B, in_context_len + num_patches, hidden_size)

        5. Second Half Blocks (in_context_start to end):
           - For each block: x = block(x, c, feat_rope_incontext)

        6. Output:
           - x = x[:, in_context_len:]        → (B, num_patches, hidden_size)
           - x = final_layer(x, c)            → (B, num_patches, P*P*3)
           - output = unpatchify(x)           → (B, 3, H, W)

    Attributes:
        in_channels: Number of input image channels.
        out_channels: Number of output image channels.
        patch_size: Patch size for tokenization.
        num_heads: Number of attention heads.
        hidden_size: Transformer hidden dimension.
        input_size: Expected input image size.
        in_context_len: Number of in-context conditioning tokens.
        in_context_start: Block index where in-context tokens are injected.
        num_classes: Number of classes for conditional generation.
        t_embedder: Timestep embedding module.
        y_embedder: Class label embedding module.
        x_embedder: Patch embedding module.
        pos_embed: Learnable positional embeddings.
        in_context_posemb: Learnable in-context token positional embeddings.
        feat_rope: RoPE for blocks before in-context injection.
        feat_rope_incontext: RoPE for blocks after in-context injection.
        blocks: List of transformer blocks.
        final_layer: Output projection layer.
    """

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 1000,
        bottleneck_dim: int = 128,
        in_context_len: int = 32,
        in_context_start: int = 8,
    ) -> None:
        """Initialize the JiT model.

        Args:
            input_size: Input image size (assumed square). Default: 256.
            patch_size: Patch size for tokenization. Default: 16.
            in_channels: Number of input channels. Default: 3.
            hidden_size: Transformer hidden dimension. Default: 1024.
            depth: Number of transformer blocks. Default: 24.
            num_heads: Number of attention heads. Default: 16.
            mlp_ratio: MLP expansion ratio. Default: 4.0.
            attn_drop: Attention dropout rate. Default: 0.0.
            proj_drop: Projection dropout rate. Default: 0.0.
            num_classes: Number of classes for conditioning. Default: 1000.
            bottleneck_dim: Bottleneck dimension in patch embedding. Default: 128.
            in_context_len: Number of in-context conditioning tokens. Default: 32.
            in_context_start: Block index for in-context injection. Default: 8.

        Note:
            Dropout is only applied to the middle 50% of blocks (from depth//4
            to depth*3//4) as per the JiT paper recommendations.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        # Timestep and class embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # Bottleneck patch embedding
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )

        # Fixed sin-cos positional embedding (not learnable)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Learnable positional embedding for in-context tokens
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=0.02)

        # 2D Rotary Position Embeddings
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        # RoPE for blocks before in-context injection (no cls tokens)
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)
        # RoPE for blocks after in-context injection (with in-context tokens as cls)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len
        )

        # Transformer blocks with selective dropout (middle 50% only)
        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                # Apply dropout only to middle 50% of blocks
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
            )
            for i in range(depth)
        ])

        # Final output layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize model weights following best practices.

        Initialization strategy:
        - Linear layers: Xavier uniform
        - Positional embeddings: Fixed sin-cos (frozen)
        - Patch embedding convolutions: Xavier uniform
        - Label embeddings: Normal(0, 0.02)
        - adaLN modulation outputs: Zero (for residual learning)
        - Final layer: Zero (for residual learning)
        """

        def _basic_init(module: nn.Module) -> None:
            """Apply Xavier initialization to Linear layers."""
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fixed sin-cos positional embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding convolutions with Xavier
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embeddings with small normal
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedder MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-initialize adaLN modulation outputs (for residual learning)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-initialize final layer outputs
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, p: int) -> torch.Tensor:
        """Convert patch predictions back to image format.

        Rearranges the patch-wise predictions into a proper image tensor by:
        1. Reshaping patches into a grid
        2. Rearranging dimensions to reconstruct the image

        Tensor Flow:
            Input:  (B, num_patches, P*P*C)
                         │
                         ▼
            Reshape: (B, H/P, W/P, P, P, C)
                         │
                         ▼
            Einsum rearrange: (B, C, H/P, P, W/P, P)
                         │
                         ▼
            Reshape: (B, C, H, W)

        Args:
            x: Patch predictions of shape (B, T, P*P*C) where:
                - B: Batch size
                - T: Number of patches (must be a perfect square)
                - P: Patch size
                - C: Number of output channels
            p: Patch size.

        Returns:
            Reconstructed images of shape (B, C, H, W).

        Raises:
            ValueError: If number of patches is not a perfect square.
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            msg = f"Input shape {x.shape[1]} is not a square number"
            raise ValueError(msg)

        # (B, T, P*P*C) -> (B, h, w, P, P, C)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        # (B, h, w, P, P, C) -> (B, C, h, P, w, P)
        x = torch.einsum("nhwpqc->nchpwq", x)
        # (B, C, h, P, w, P) -> (B, C, H, W) where H = h*P, W = w*P
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass through the JiT model.

        Processes input images with timestep and class conditioning to produce
        an output of the same shape (e.g., predicted noise, velocity, or x_0).

        Args:
            x: Noisy input images of shape (B, C, H, W) where:
                - B: Batch size
                - C: Number of channels (3 for RGB)
                - H: Image height (must match input_size)
                - W: Image width (must match input_size)
            t: Diffusion timesteps of shape (B,).
                Values typically in [0, 1] for flow matching.
            y: Class labels of shape (B,).
                Values in [0, num_classes] where num_classes is null class.

        Returns:
            Predicted clean images of shape (B, C, H, W), same as input.

        Note:
            The interpretation of the output depends on the training objective
            (e.g., x-prediction, v-prediction, or epsilon-prediction).
        """
        # Compute conditioning embeddings
        # t: (B,) -> t_emb: (B, hidden_size)
        t_emb = self.t_embedder(t)
        # y: (B,) -> y_emb: (B, hidden_size)
        y_emb = self.y_embedder(y)
        # Combined conditioning: (B, hidden_size)
        c = t_emb + y_emb

        # Patch embedding with positional encoding
        # x: (B, C, H, W) -> (B, num_patches, hidden_size)
        x = self.x_embedder(x)
        # Add fixed positional embedding
        x += self.pos_embed

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            # Inject in-context tokens at specified block
            if self.in_context_len > 0 and i == self.in_context_start:
                # Create in-context tokens from class embedding
                # (B, hidden_size) -> (B, in_context_len, hidden_size)
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                # Add learnable positional embedding
                in_context_tokens += self.in_context_posemb
                # Prepend to sequence: (B, in_context_len + num_patches, hidden_size)
                x = torch.cat([in_context_tokens, x], dim=1)

            # Choose RoPE based on whether in-context tokens are present
            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, rope)

        # Remove in-context tokens from sequence
        # (B, in_context_len + num_patches, hidden_size) -> (B, num_patches, hidden_size)
        x = x[:, self.in_context_len :]

        # Final layer and unpatchify
        # (B, num_patches, hidden_size) -> (B, num_patches, P*P*C)
        x = self.final_layer(x, c)
        # (B, num_patches, P*P*C) -> (B, C, H, W)
        output = self.unpatchify(x, self.patch_size)

        return output


# =============================================================================
# Model Factory Functions
# =============================================================================


def JiT_B_16(**kwargs) -> JiT:
    """Create JiT-Base model with patch size 16.

    Configuration:
        - Depth: 12 blocks
        - Hidden size: 768
        - Heads: 12
        - Parameters: ~86M (similar to ViT-B)
        - In-context injection: block 4

    Args:
        **kwargs: Additional arguments passed to JiT constructor.

    Returns:
        JiT model instance.
    """
    return JiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def JiT_B_32(**kwargs) -> JiT:
    """Create JiT-Base model with patch size 32.

    Configuration:
        - Depth: 12 blocks
        - Hidden size: 768
        - Heads: 12
        - Parameters: ~86M (similar to ViT-B)
        - In-context injection: block 4
        - Larger patches → fewer tokens → faster but less detail

    Args:
        **kwargs: Additional arguments passed to JiT constructor.

    Returns:
        JiT model instance.
    """
    return JiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=32,
        **kwargs,
    )


def JiT_L_16(**kwargs) -> JiT:
    """Create JiT-Large model with patch size 16.

    Configuration:
        - Depth: 24 blocks
        - Hidden size: 1024
        - Heads: 16
        - Parameters: ~307M (similar to ViT-L)
        - In-context injection: block 8

    Args:
        **kwargs: Additional arguments passed to JiT constructor.

    Returns:
        JiT model instance.
    """
    return JiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=16,
        **kwargs,
    )


def JiT_L_32(**kwargs) -> JiT:
    """Create JiT-Large model with patch size 32.

    Configuration:
        - Depth: 24 blocks
        - Hidden size: 1024
        - Heads: 16
        - Parameters: ~307M (similar to ViT-L)
        - In-context injection: block 8
        - Larger patches → fewer tokens → faster but less detail

    Args:
        **kwargs: Additional arguments passed to JiT constructor.

    Returns:
        JiT model instance.
    """
    return JiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=32,
        **kwargs,
    )


def JiT_H_16(**kwargs) -> JiT:
    """Create JiT-Huge model with patch size 16.

    Configuration:
        - Depth: 32 blocks
        - Hidden size: 1280
        - Heads: 16
        - Parameters: ~632M (similar to ViT-H)
        - In-context injection: block 10
        - Larger bottleneck dim: 256

    Args:
        **kwargs: Additional arguments passed to JiT constructor.

    Returns:
        JiT model instance.
    """
    return JiT(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=16,
        **kwargs,
    )


def JiT_H_32(**kwargs) -> JiT:
    """Create JiT-Huge model with patch size 32.

    Configuration:
        - Depth: 32 blocks
        - Hidden size: 1280
        - Heads: 16
        - Parameters: ~632M (similar to ViT-H)
        - In-context injection: block 10
        - Larger bottleneck dim: 256
        - Larger patches → fewer tokens → faster but less detail

    Args:
        **kwargs: Additional arguments passed to JiT constructor.

    Returns:
        JiT model instance.
    """
    return JiT(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=32,
        **kwargs,
    )


# Registry of available JiT model variants
JiT_models = {
    "JiT-B/16": JiT_B_16,
    "JiT-B/32": JiT_B_32,
    "JiT-L/16": JiT_L_16,
    "JiT-L/32": JiT_L_32,
    "JiT-H/16": JiT_H_16,
    "JiT-H/32": JiT_H_32,
}
