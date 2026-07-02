"""PixelFlow transformer model for class-conditional image generation.

Adapted from the official PixelFlow implementation
(ShoufaChen/PixelFlow-Class2Image). This module contains the full model
architecture including patch embedding, RoPE attention, adaLN conditioning,
and multi-stage pyramid support.

Key architecture details:
    - Patch size 4, XL config: depth=28, heads=16, head_dim=72 (embed=1152)
    - 2D RoPE positional embeddings
    - AdaLayerNorm-Zero conditioning (timestep + class + stage size)
    - v-prediction (velocity) target
"""

from __future__ import annotations

import warnings
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import LabelEmbedding, TimestepEmbedding, Timesteps

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "`flash-attn` is not installed. Training mode may not work properly.",
        UserWarning,
        stacklevel=2,
    )
    flash_attn_varlen_func = None


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Apply rotary positional embeddings to input tensor.

    Args:
        x: Input tensor of shape (B, H, S, D) or (H, S, D).
        freqs_cis: Pre-computed cos/sin frequencies, last dim is 2 (cos, sin).

    Returns:
        Tensor with rotary embeddings applied, same shape as x.
    """
    cos, sin = freqs_cis.unbind(-1)
    cos = cos[None, None]
    sin = sin[None, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out


class PatchEmbed(nn.Module):
    """Patch embedding using convolution projection.

    Args:
        patch_size: Size of each patch.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        bias: Whether to use bias in projection.
    """

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, bias=bias)

    def forward_unfold(self, x: torch.Tensor) -> torch.Tensor:
        """Unfold-based forward for training (sequence packing compatible).

        Args:
            x: Input tensor.

        Returns:
            Projected patches.
        """
        out_unfold = x.matmul(self.proj.weight.view(self.proj.weight.size(0), -1).t())
        if self.proj.bias is not None:
            out_unfold += self.proj.bias.to(out_unfold.dtype)
        return out_unfold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, N, D).
        """
        if self.training:
            return self.forward_unfold(x)
        out = self.proj(x)
        out = out.flatten(2).transpose(1, 2)
        return out


class AdaLayerNorm(nn.Module):
    """Adaptive layer normalization with zero initialization (adaLN-Zero).

    Produces 6 modulation parameters: shift/scale/gate for both MSA and MLP.

    Args:
        embedding_dim: Model embedding dimension.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        seqlen_list: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor of shape (B, N, D) or (total_tokens, D).
            timestep: Conditioning embedding of shape (B, D).
            seqlen_list: Per-sample sequence lengths for packed mode.

        Returns:
            Tuple of (normed_x, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        """
        input_dtype = x.dtype
        emb = self.linear(self.silu(timestep))

        if seqlen_list is not None:
            emb = torch.cat(
                [one_emb[None].expand(repeat_time, -1) for one_emb, repeat_time in zip(emb, seqlen_list)]
            )
        else:
            emb = emb.unsqueeze(1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.float().chunk(6, dim=-1)
        x = self.norm(x).float() * (1 + scale_msa) + shift_msa
        return x.to(input_dtype), gate_msa, shift_mlp, scale_mlp, gate_mlp


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation.

    Args:
        dim: Input dimension.
        dim_out: Output dimension (defaults to dim).
        mult: Hidden dimension multiplier.
        inner_dim: Explicit inner dimension (overrides mult).
        bias: Whether to use bias.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        inner_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult) if inner_dim is None else inner_dim
        dim_out = dim_out if dim_out is not None else dim
        self.fc1 = nn.Linear(dim, inner_dim, bias=bias)
        self.fc2 = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor of shape (..., dim).

        Returns:
            Output tensor of shape (..., dim_out).
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        dim: Normalization dimension.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of shape (..., dim).
        """
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * output).to(x.dtype)


class Attention(nn.Module):
    """Multi-head attention with RoPE and optional flash attention.

    Supports both standard batched mode and sequence-packed mode (flash_attn).

    Args:
        q_dim: Query dimension.
        kv_dim: Key/Value dimension (defaults to q_dim).
        heads: Number of attention heads.
        head_dim: Dimension per head.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int | None = None,
        heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim if kv_dim is not None else q_dim
        self.inner_dim = head_dim * heads
        self.dropout = dropout
        self.head_dim = head_dim
        self.num_heads = heads

        self.q_proj = nn.Linear(self.q_dim, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(self.kv_dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(self.kv_dim, self.inner_dim, bias=bias)
        self.o_proj = nn.Linear(self.inner_dim, self.q_dim, bias=bias)

        self.q_norm = RMSNorm(self.inner_dim)
        self.k_norm = RMSNorm(self.inner_dim)

    def prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> torch.Tensor | None:
        """Prepare attention mask for scaled_dot_product_attention.

        Args:
            attention_mask: Raw attention mask or None.
            target_length: Target sequence length.
            batch_size: Batch size.
            out_dim: Output dimensions (3 or 4).

        Returns:
            Prepared attention mask or None.
        """
        if attention_mask is None:
            return attention_mask

        current_length = attention_mask.shape[-1]
        if current_length != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * self.num_heads:
                attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=1)

        return attention_mask

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        cross_attention: bool = False,
        rope_pos_embed: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional RoPE and flash attention.

        Args:
            inputs_q: Query tensor.
            inputs_kv: Key/Value tensor (defaults to inputs_q for self-attention).
            attention_mask: Optional attention mask.
            cross_attention: Whether this is cross-attention.
            rope_pos_embed: Rotary position embeddings.
            cu_seqlens_q: Cumulative sequence lengths for queries (flash_attn).
            cu_seqlens_k: Cumulative sequence lengths for keys (flash_attn).
            max_seqlen_q: Maximum query sequence length (flash_attn).
            max_seqlen_k: Maximum key sequence length (flash_attn).

        Returns:
            Attention output tensor.
        """
        inputs_kv = inputs_q if inputs_kv is None else inputs_kv

        query_states = self.q_proj(inputs_q)
        key_states = self.k_proj(inputs_kv)
        value_states = self.v_proj(inputs_kv)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if max_seqlen_q is None:
            assert not self.training, "PixelFlow needs sequence packing for training"

            bsz, q_len, _ = inputs_q.shape
            _, kv_len, _ = inputs_kv.shape

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

            query_states = apply_rotary_emb(query_states, rope_pos_embed)
            if not cross_attention:
                key_states = apply_rotary_emb(key_states, rope_pos_embed)

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, kv_len, bsz)
                attention_mask = attention_mask.view(bsz, self.num_heads, -1, attention_mask.shape[-1])

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, self.inner_dim)
            attn_output = self.o_proj(attn_output)
            return attn_output

        else:
            query_states = query_states.view(-1, self.num_heads, self.head_dim)
            key_states = key_states.view(-1, self.num_heads, self.head_dim)
            value_states = value_states.view(-1, self.num_heads, self.head_dim)

            query_states = apply_rotary_emb(
                query_states.permute(1, 0, 2)[None], rope_pos_embed
            )[0].permute(1, 0, 2)
            if not cross_attention:
                key_states = apply_rotary_emb(
                    key_states.permute(1, 0, 2)[None], rope_pos_embed
                )[0].permute(1, 0, 2)

            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )

            attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
            attn_output = self.o_proj(attn_output)
            return attn_output


class TransformerBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning and optional cross-attention.

    Args:
        dim: Model dimension.
        num_attention_heads: Number of attention heads.
        attention_head_dim: Dimension per attention head.
        dropout: Dropout probability.
        cross_attention_dim: Cross-attention dimension (None to disable).
        attention_bias: Whether to use bias in attention projections.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)

        self.attn1 = Attention(
            q_dim=dim,
            kv_dim=None,
            heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        if cross_attention_dim is not None:
            self.norm2 = RMSNorm(dim, eps=1e-6)
            self.attn2 = Attention(
                q_dim=dim,
                kv_dim=cross_attention_dim,
                heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
            )
        else:
            self.attn2 = None

        self.norm3 = RMSNorm(dim, eps=1e-6)
        self.mlp = FeedForward(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        rope_pos_embed: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        seqlen_list_q: list[int] | None = None,
        seqlen_list_k: list[int] | None = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            hidden_states: Input tensor.
            encoder_hidden_states: Cross-attention context.
            encoder_attention_mask: Cross-attention mask.
            timestep: AdaLN conditioning embedding.
            rope_pos_embed: Rotary position embeddings.
            cu_seqlens_q: Cumulative query sequence lengths (flash_attn).
            cu_seqlens_k: Cumulative key sequence lengths (flash_attn).
            seqlen_list_q: Per-sample query lengths.
            seqlen_list_k: Per-sample key lengths.

        Returns:
            Output tensor with same shape as hidden_states.
        """
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, seqlen_list_q
        )

        attn_output = self.attn1(
            inputs_q=norm_hidden_states,
            inputs_kv=None,
            attention_mask=None,
            cross_attention=False,
            rope_pos_embed=rope_pos_embed,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max(seqlen_list_q) if seqlen_list_q is not None else None,
            max_seqlen_k=max(seqlen_list_q) if seqlen_list_q is not None else None,
        )

        attn_output = (gate_msa * attn_output.float()).to(attn_output.dtype)
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                inputs_q=norm_hidden_states,
                inputs_kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                cross_attention=True,
                rope_pos_embed=rope_pos_embed,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max(seqlen_list_q) if seqlen_list_q is not None else None,
                max_seqlen_k=max(seqlen_list_k) if seqlen_list_k is not None else None,
            )
            hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm3(hidden_states)
        norm_hidden_states = (norm_hidden_states.float() * (1 + scale_mlp) + shift_mlp).to(
            norm_hidden_states.dtype
        )
        ff_output = self.mlp(norm_hidden_states)
        ff_output = (gate_mlp * ff_output.float()).to(ff_output.dtype)
        hidden_states = ff_output + hidden_states

        return hidden_states


class PixelFlowModel(nn.Module):
    """PixelFlow transformer for class-conditional image generation.

    Multi-stage pyramid model operating directly in pixel space with
    v-prediction (velocity) target and flow matching.

    Args:
        in_channels: Input image channels (3 for RGB).
        out_channels: Output channels (3 for RGB).
        num_attention_heads: Number of attention heads.
        attention_head_dim: Dimension per attention head.
        depth: Number of transformer blocks.
        patch_size: Patch embedding size.
        dropout: Dropout probability.
        cross_attention_dim: Cross-attention dimension (None to disable).
        attention_bias: Whether to use bias in attention.
        num_classes: Number of classes (0 for unconditional).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        depth: int,
        patch_size: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        attention_bias: bool = True,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.attention_head_dim = attention_head_dim
        self.num_classes = num_classes
        self.out_channels = out_channels

        embed_dim = num_attention_heads * attention_head_dim
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embed_dim)

        self.latent_size_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embed_dim)
        if self.num_classes > 0:
            self.class_embedder = LabelEmbedding(num_classes, embed_dim, dropout_prob=0.1)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout,
                    cross_attention_dim,
                    attention_bias,
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(embed_dim, 2 * embed_dim)
        self.proj_out_2 = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights following the official PixelFlow scheme."""

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        nn.init.normal_(self.timestep_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.linear_2.weight, std=0.02)

        nn.init.normal_(self.latent_size_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.latent_size_embedder.linear_2.weight, std=0.02)

        if self.num_classes > 0:
            nn.init.normal_(self.class_embedder.embedding_table.weight, std=0.02)

        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)

        nn.init.constant_(self.proj_out_1.weight, 0)
        nn.init.constant_(self.proj_out_1.bias, 0)
        nn.init.constant_(self.proj_out_2.weight, 0)
        nn.init.constant_(self.proj_out_2.bias, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        latent_size: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        seqlen_list_q: list[int] | None = None,
        seqlen_list_k: list[int] | None = None,
    ) -> torch.Tensor:
        """Forward pass through the PixelFlow model.

        Args:
            hidden_states: Input images of shape (B, C, H, W).
            encoder_hidden_states: Cross-attention context (optional).
            class_labels: Class labels of shape (B,).
            timestep: Discrete timesteps of shape (B,) in [0, 1000).
            latent_size: Patch grid size of shape (B,) (e.g., 64 for 256/4).
            encoder_attention_mask: Cross-attention mask (optional).
            pos_embed: RoPE positional embeddings.
            cu_seqlens_q: Cumulative query lengths (flash_attn packed mode).
            cu_seqlens_k: Cumulative key lengths (flash_attn packed mode).
            seqlen_list_q: Per-sample query lengths (packed mode).
            seqlen_list_k: Per-sample key lengths (packed mode).

        Returns:
            Velocity prediction of shape (B, C, H, W) in inference mode,
            or flattened patches in training mode.
        """
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        orig_height, orig_width = hidden_states.shape[-2], hidden_states.shape[-1]
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = self.patch_embed(hidden_states)

        timesteps_proj = self.time_proj(timestep)
        conditioning = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))

        if self.num_classes > 0:
            class_embed = self.class_embedder(class_labels)
            conditioning += class_embed

        latent_size_proj = self.time_proj(latent_size)
        latent_size_embed = self.latent_size_embedder(latent_size_proj.to(dtype=hidden_states.dtype))
        conditioning += latent_size_embed

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=conditioning,
                rope_pos_embed=pos_embed,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqlen_list_q=seqlen_list_q,
                seqlen_list_k=seqlen_list_k,
            )

        shift, scale = self.proj_out_1(F.silu(conditioning)).float().chunk(2, dim=1)
        if seqlen_list_q is None:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        else:
            shift = torch.cat(
                [shift_i[None].expand(ri, -1) for shift_i, ri in zip(shift, seqlen_list_q)]
            )
            scale = torch.cat(
                [scale_i[None].expand(ri, -1) for scale_i, ri in zip(scale, seqlen_list_q)]
            )

        hidden_states = (self.norm_out(hidden_states).float() * (1 + scale) + shift).to(
            hidden_states.dtype
        )
        hidden_states = self.proj_out_2(hidden_states)
        if self.training:
            hidden_states = hidden_states.reshape(
                hidden_states.shape[0], self.patch_size, self.patch_size, self.out_channels
            )
            hidden_states = hidden_states.permute(0, 3, 1, 2).flatten(1)
            return hidden_states

        height = orig_height // self.patch_size
        width = orig_width // self.patch_size
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        return output


__all__ = ["PixelFlowModel"]
