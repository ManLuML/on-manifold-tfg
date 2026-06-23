"""DiT (Diffusion Transformer) model architecture.

This module contains the DiT model implementation adapted from the original
Meta DiT repository (https://github.com/facebookresearch/DiT).

DiT replaces the U-Net backbone in diffusion models with a transformer,
using adaptive layer normalization (adaLN-Zero) for timestep and class
conditioning.

References:
    - "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
    - https://arxiv.org/abs/2212.09748
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

    Uses sinusoidal positional encoding followed by an MLP.

    Args:
        hidden_size: Output embedding dimension.
        frequency_embedding_size: Dimension of sinusoidal embedding.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
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
            t: 1-D tensor of N indices (can be fractional).
            dim: Output dimension.
            max_period: Controls minimum frequency of embeddings.

        Returns:
            Tensor of shape (N, dim) containing positional embeddings.
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
            t: Tensor of timesteps of shape (B,).

        Returns:
            Embeddings of shape (B, hidden_size).
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Supports label dropout for classifier-free guidance training.

    Args:
        num_classes: Number of classes in the dataset.
        hidden_size: Output embedding dimension.
        dropout_prob: Probability of dropping labels (for CFG).
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Drop labels for classifier-free guidance.

        Args:
            labels: Class labels of shape (B,).
            force_drop_ids: Optional mask to force dropping specific labels.

        Returns:
            Labels with some replaced by num_classes (unconditional token).
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
        """Embed class labels.

        Args:
            labels: Class labels of shape (B,).
            train: Whether in training mode (enables dropout).
            force_drop_ids: Optional mask to force dropping.

        Returns:
            Embeddings of shape (B, hidden_size).
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.

    The block applies attention and MLP with modulation from the
    combined timestep and class embeddings.

    Args:
        hidden_size: Transformer hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension multiplier.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
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
            x: Input tensor of shape (B, N, D).
            c: Conditioning tensor of shape (B, D).

        Returns:
            Output tensor of shape (B, N, D).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final layer of DiT that unpatchifies the output.

    Args:
        hidden_size: Transformer hidden dimension.
        patch_size: Patch size used in embedding.
        out_channels: Number of output channels.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, D).
            c: Conditioning tensor of shape (B, D).

        Returns:
            Output tensor of shape (B, N, patch_size^2 * out_channels).
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion Transformer model.

    A transformer-based diffusion model that operates on patches of images
    or latent representations. Uses adaLN-Zero for conditioning on timesteps
    and class labels.

    Args:
        input_size: Input spatial size (assumes square input).
        patch_size: Patch size for patch embedding.
        in_channels: Number of input channels (4 for VAE latents).
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension multiplier.
        class_dropout_prob: Label dropout probability for CFG.
        num_classes: Number of classes for conditioning.
        learn_sigma: Whether to predict variance (doubles output channels).

    Example:
        >>> model = DiT(input_size=32, in_channels=4, num_classes=1000)
        >>> x = torch.randn(2, 4, 32, 32)  # VAE latents
        >>> t = torch.randint(0, 1000, (2,))  # Timesteps
        >>> y = torch.randint(0, 1000, (2,))  # Class labels
        >>> out = model(x, t, y)  # (2, 8, 32, 32) if learn_sigma=True
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

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize model weights."""

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize positional embedding with sin-cos
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
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
        """Convert patch tokens back to spatial image.

        Args:
            x: Tensor of shape (N, T, patch_size^2 * C).

        Returns:
            Images of shape (N, C, H, W).
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
        """Forward pass of DiT.

        Args:
            x: Spatial inputs of shape (N, C, H, W).
                For latent diffusion: (N, 4, 32, 32) VAE latents.
            t: Diffusion timesteps of shape (N,).
                DDPM convention: integers in [0, num_timesteps-1].
            y: Class labels of shape (N,).
                Values in [0, num_classes-1], or num_classes for unconditional.

        Returns:
            Output of shape (N, out_channels, H, W).
            If learn_sigma=True, out_channels = 2 * in_channels.
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size^2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Forward pass with classifier-free guidance.

        Batches conditional and unconditional forward passes for efficiency.

        Note:
            For reproducibility with original DiT, CFG is applied only to
            the first 3 channels (RGB). This can be changed by modifying
            the channel slicing.

        Args:
            x: Spatial inputs of shape (2N, C, H, W).
                First half: conditional, second half: unconditional.
            t: Timesteps of shape (2N,).
            y: Labels of shape (2N,).
                Second half should be num_classes (unconditional).
            cfg_scale: Guidance scale (1.0 = no guidance).

        Returns:
            CFG-guided output of shape (2N, out_channels, H, W).
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # Apply CFG on first 3 channels only (for reproducibility)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# Positional embedding utilities


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0
) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Grid height/width.
        cls_token: Whether to include class token position.
        extra_tokens: Number of extra tokens.

    Returns:
        Positional embeddings of shape [grid_size^2, embed_dim] or
        [1 + grid_size^2, embed_dim] with cls_token.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate positional embeddings from a 2D grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embeddings.

    Args:
        embed_dim: Output dimension for each position.
        pos: Positions to encode of shape (M,).

    Returns:
        Embeddings of shape (M, embed_dim).
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


# Model configurations


def DiT_XL_2(**kwargs) -> DiT:
    """DiT-XL/2 model (675M parameters)."""
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs) -> DiT:
    """DiT-XL/4 model."""
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs) -> DiT:
    """DiT-XL/8 model."""
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs) -> DiT:
    """DiT-L/2 model (307M parameters)."""
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs) -> DiT:
    """DiT-L/4 model."""
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs) -> DiT:
    """DiT-L/8 model."""
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs) -> DiT:
    """DiT-B/2 model (86M parameters)."""
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs) -> DiT:
    """DiT-B/4 model."""
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs) -> DiT:
    """DiT-B/8 model."""
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs) -> DiT:
    """DiT-S/2 model (65M parameters)."""
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs) -> DiT:
    """DiT-S/4 model."""
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs) -> DiT:
    """DiT-S/8 model."""
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}
