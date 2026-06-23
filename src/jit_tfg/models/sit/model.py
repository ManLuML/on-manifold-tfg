"""SiT (Scalable Interpolant Transformers) model architecture.

This module provides the SiT model architecture for v-prediction (velocity).
SiT uses the same transformer architecture as DiT but predicts velocity
instead of noise (epsilon).

Key differences from DiT:
1. Prediction target: v (velocity) instead of ε (noise)
2. Time convention: Same as JiT flow matching (t=0: noise, t=1: clean)
3. Sampling: ODE/SDE integration instead of DDPM/DDIM

The SiT architecture is identical to DiT:
- Patch embedding for input images
- Sinusoidal timestep embedding
- Class embedding with dropout for CFG
- adaLN-Zero modulated transformer blocks
- Unpatchify for output

References:
    - SiT paper: "Scalable Interpolant Transformers"
    - DiT paper: "Scalable Diffusion Models with Transformers"
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization modulation.

    Args:
        x: Input tensor of shape (B, N, D).
        shift: Shift parameter of shape (B, D).
        scale: Scale parameter of shape (B, D).

    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Uses sinusoidal position encoding followed by an MLP.

    Note:
        Unlike DiT which takes discrete timesteps [0, T-1],
        SiT takes continuous timesteps in [0, 1].
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        """Initialize timestep embedder.

        Args:
            hidden_size: Output embedding dimension.
            frequency_embedding_size: Intermediate frequency embedding dimension.
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

        Args:
            t: 1D Tensor of N timesteps (may be fractional).
            dim: Output dimension.
            max_period: Controls minimum frequency.

        Returns:
            Tensor of shape (N, dim) with positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: Timesteps of shape (B,).

        Returns:
            Embeddings of shape (B, hidden_size).
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Handles label dropout for classifier-free guidance (CFG).
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float) -> None:
        """Initialize label embedder.

        Args:
            num_classes: Number of classes.
            hidden_size: Embedding dimension.
            dropout_prob: Probability of dropping labels for CFG.
        """
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Drop labels to enable classifier-free guidance.

        Args:
            labels: Class labels of shape (B,).
            force_drop_ids: Optional mask to force dropping.

        Returns:
            Labels with some replaced by num_classes (null class).
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed labels with optional dropout.

        Args:
            labels: Class labels of shape (B,).
            train: Whether in training mode.
            force_drop_ids: Optional mask to force dropping.

        Returns:
            Embeddings of shape (B, hidden_size).
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class SiTBlock(nn.Module):
    """SiT transformer block with adaLN-Zero conditioning.

    Identical to DiTBlock but named for clarity.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs) -> None:
        """Initialize SiT block.

        Args:
            hidden_size: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension multiplier.
            **block_kwargs: Additional kwargs for attention.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass with conditioning.

        Args:
            x: Input of shape (B, N, D).
            c: Conditioning of shape (B, D).

        Returns:
            Output of shape (B, N, D).
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final layer of SiT with adaLN modulation."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        """Initialize final layer.

        Args:
            hidden_size: Hidden dimension.
            patch_size: Patch size for unpatchification.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass with conditioning.

        Args:
            x: Input of shape (B, N, D).
            c: Conditioning of shape (B, D).

        Returns:
            Output of shape (B, N, patch_size^2 * out_channels).
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0
) -> np.ndarray:
    """Generate 2D sinusoidal positional embedding.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Grid height and width.
        cls_token: Whether to include cls token position.
        extra_tokens: Number of extra tokens.

    Returns:
        Positional embedding of shape (grid_size^2, embed_dim) or
        (1 + grid_size^2, embed_dim) with cls token.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    # Split embedding between h and w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embedding from grid.

    Args:
        embed_dim: Output dimension.
        pos: Positions to encode.

    Returns:
        Embedding of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class SiT(nn.Module):
    """Scalable Interpolant Transformer for v-prediction.

    SiT uses flow matching with velocity prediction:
    - Forward: z_t = t*x + (1-t)*ε (same as JiT)
    - Velocity: v = x - ε (the target prediction)
    - Time: t=0 is noise, t=1 is clean data

    Attributes:
        learn_sigma: Whether to predict variance.
        in_channels: Input channels (4 for VAE latents).
        out_channels: Output channels.
        patch_size: Patch size for tokenization.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
    ) -> None:
        """Initialize SiT model.

        Args:
            input_size: Input spatial size (e.g., 32 for latents).
            patch_size: Patch size for tokenization.
            in_channels: Input channels.
            hidden_size: Transformer hidden dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension multiplier.
            class_dropout_prob: Label dropout probability for CFG.
            num_classes: Number of classes.
            learn_sigma: Whether to predict variance.
        """
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches

        # Fixed sin-cos positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize model weights."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize positional embedding with sin-cos
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers (critical for training stability)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patched representation back to spatial.

        Args:
            x: Patched tensor of shape (B, N, patch_size^2 * C).

        Returns:
            Spatial tensor of shape (B, C, H, W).
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of SiT.

        Args:
            x: Spatial inputs of shape (B, C, H, W).
            t: Timesteps of shape (B,) in [0, 1].
            y: Class labels of shape (B,).

        Returns:
            Velocity prediction of shape (B, C, H, W).
            If learn_sigma=True, returns (B, 2*C, H, W) but we extract v.
        """
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        # Extract velocity (first in_channels)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        """Forward with classifier-free guidance (Original SiT/DiT style).

        IMPORTANT: This method applies CFG to first 3 channels only, following
        the original SiT/DiT implementation for exact reproducibility with
        published benchmarks.

        Historical context:
            The 3-channel CFG originates from GLIDE (pixel-space RGB model).
            DiT/SiT adopted this convention for reproducibility with existing
            benchmarks. DiT paper (Appendix A) notes: "three-channel guidance
            and four-channel guidance give similar results when adjusting the
            scale factor" (3-channel scale 1.5 ≈ 4-channel scale 1.375).

        Comparison with SiTWrapper.forward_cfg():
            - SiT.forward_with_cfg() [this method]:
              * 3-channel CFG (original style, for reproducibility)
              * Requires pre-concatenated batch (B*2, C, H, W)
              * Used for: Reproducing original SiT/DiT benchmark numbers
            - SiTWrapper.forward_cfg():
              * Configurable CFG (first3 or all channels via cfg_channel_mode)
              * Takes single batch, runs two forward passes internally
              * Used for: TFG experiments (all our denoisers use this pattern)

        For TFG experiments, use SiTWrapper.forward_cfg() instead.

        Args:
            x: Input of shape (B*2, C, H, W) - concatenated cond and uncond.
            t: Timesteps of shape (B*2,).
            y: Labels of shape (B*2,) - cond then null labels.
            cfg_scale: Guidance scale.

        Returns:
            CFG-guided velocity of shape (B, C, H, W).
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        # Apply CFG on first 3 channels (following original SiT)
        v, rest = model_out[:, :3], model_out[:, 3:]
        cond_v, uncond_v = torch.split(v, len(v) // 2, dim=0)
        half_v = uncond_v + cfg_scale * (cond_v - uncond_v)
        v = torch.cat([half_v, half_v], dim=0)
        return torch.cat([v, rest], dim=1)


# Model configurations
def SiT_XL_2(**kwargs) -> SiT:
    """SiT-XL/2 model (675M parameters)."""
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def SiT_XL_4(**kwargs) -> SiT:
    """SiT-XL/4 model."""
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def SiT_XL_8(**kwargs) -> SiT:
    """SiT-XL/8 model."""
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def SiT_L_2(**kwargs) -> SiT:
    """SiT-L/2 model."""
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def SiT_L_4(**kwargs) -> SiT:
    """SiT-L/4 model."""
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def SiT_L_8(**kwargs) -> SiT:
    """SiT-L/8 model."""
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def SiT_B_2(**kwargs) -> SiT:
    """SiT-B/2 model."""
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def SiT_B_4(**kwargs) -> SiT:
    """SiT-B/4 model."""
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def SiT_B_8(**kwargs) -> SiT:
    """SiT-B/8 model."""
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def SiT_S_2(**kwargs) -> SiT:
    """SiT-S/2 model."""
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def SiT_S_4(**kwargs) -> SiT:
    """SiT-S/4 model."""
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def SiT_S_8(**kwargs) -> SiT:
    """SiT-S/8 model."""
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


# Model registry
SiT_models = {
    "SiT-XL/2": SiT_XL_2,
    "SiT-XL/4": SiT_XL_4,
    "SiT-XL/8": SiT_XL_8,
    "SiT-L/2": SiT_L_2,
    "SiT-L/4": SiT_L_4,
    "SiT-L/8": SiT_L_8,
    "SiT-B/2": SiT_B_2,
    "SiT-B/4": SiT_B_4,
    "SiT-B/8": SiT_B_8,
    "SiT-S/2": SiT_S_2,
    "SiT-S/4": SiT_S_4,
    "SiT-S/8": SiT_S_8,
}
