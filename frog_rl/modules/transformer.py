# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from frog_rl.networks import EmpiricalNormalization, MLP
from frog_rl.utils import resolve_nn_activation


class _CrossAttentionLayer(nn.Module):
    """Transformer-style block where queries attend to separate key/value inputs."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
        norm_first: bool,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_dim),
            resolve_nn_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            query = self.norm1(q)
            attended, _ = self.attention(
                query=query,
                key=k,
                value=v,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            q = q + self.dropout1(attended)
            q = q + self.dropout2(self.feed_forward(self.norm2(q)))
            return q

        attended, _ = self.attention(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q = self.norm1(q + self.dropout1(attended))
        return self.norm2(q + self.dropout2(self.feed_forward(q)))


class Transformer(nn.Module):
    """Standalone transformer encoder and cross-attention module.

    The module can consume either one tensor or multiple tensors that share the
    same batch/time dimensions. Multiple inputs are concatenated along the last
    dimension before being projected into the transformer space.

    For separate query, key, and value inputs, call ``forward(q=..., k=..., v=...)``.
    In that mode, the query sequence attends to the key/value sequence.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        *,
        input_dims: Sequence[int] | None = None,
        q_dim: int | None = None,
        k_dim: int | None = None,
        v_dim: int | None = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_hidden_dim: int | None = None,
        output_dim: int | None = None,
        input_hidden_sizes: Sequence[int] | None = None,
        output_hidden_sizes: Sequence[int] | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        normalize_input: bool = False,
        normalization_until: int | None = int(1e8),
        max_seq_len: int = 128,
        use_cls_token: bool = True,
        pooling: str = "cls",
        batch_first: bool = True,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
    ) -> None:
        """Initialize the transformer encoder.

        Args:
            input_dim: Total feature dimension of one token. Use this for a single input tensor.
            input_dims: Feature dimensions of multiple input chunks. The chunks are concatenated on the last axis.
            q_dim: Raw feature dimension of the query input in separate-QKV mode. Defaults to ``input_dim``.
            k_dim: Raw feature dimension of the key input in separate-QKV mode. Defaults to ``q_dim``.
            v_dim: Raw feature dimension of the value input in separate-QKV mode. Defaults to ``k_dim``.
            hidden_dim: Transformer embedding dimension.
            num_layers: Number of encoder layers.
            num_heads: Number of attention heads.
            ff_hidden_dim: Feed-forward hidden size. Defaults to ``4 * hidden_dim``.
            output_dim: Final output size. Defaults to ``hidden_dim``.
            input_hidden_sizes: Optional hidden sizes for an input MLP before the transformer.
            output_hidden_sizes: Optional hidden sizes for an output MLP after pooling.
            activation: Activation function name.
            dropout: Dropout rate for transformer layers.
            normalize_input: Whether to apply empirical input normalization.
            normalization_until: Number of samples used to update the input normalizer.
            max_seq_len: Maximum supported sequence length, excluding the CLS token.
            use_cls_token: Whether to prepend a learnable CLS token.
            pooling: Pooling mode. One of ``"cls"``, ``"mean"``, or ``"max"``.
            batch_first: Whether sequence inputs are batch-first.
            layer_norm_eps: Epsilon used by LayerNorm.
            norm_first: Whether to use Pre-LN transformer blocks.
            **kwargs: Extra arguments are accepted for forward compatibility and ignored.
        """
    
        super().__init__()

        if input_dim is None and input_dims is None and q_dim is not None:
            input_dim = q_dim
        self.input_dims = tuple(input_dims) if input_dims is not None else None
        self.input_dim = self._resolve_input_dim(input_dim, self.input_dims)
        self.q_dim = q_dim if q_dim is not None else self.input_dim
        self.k_dim = k_dim if k_dim is not None else self.q_dim
        self.v_dim = v_dim if v_dim is not None else self.k_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.batch_first = batch_first
        self.use_cls_token = use_cls_token
        self.pooling = pooling
        self.max_seq_len = max_seq_len
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first

        if num_layers <= 0:
            raise ValueError("Transformer requires at least one encoder layer.")
        if num_heads <= 0:
            raise ValueError("Transformer requires at least one attention head.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if pooling not in {"cls", "mean", "max"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}. Use 'cls', 'mean', or 'max'.")
        if pooling == "cls" and not use_cls_token:
            raise ValueError("CLS pooling requires use_cls_token=True.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.q_dim <= 0 or self.k_dim <= 0 or self.v_dim <= 0:
            raise ValueError("q_dim, k_dim, and v_dim must be positive.")

        self.input_normalizer = (
            EmpiricalNormalization(shape=[self.input_dim], until=normalization_until) if normalize_input else nn.Identity()
        )

        if input_hidden_sizes:
            self.input_proj = MLP(self.input_dim, hidden_dim, list(input_hidden_sizes), activation=activation)
        else:
            self.input_proj = nn.Linear(self.input_dim, hidden_dim)

        self.q_proj = nn.Linear(self.q_dim, hidden_dim)
        self.k_proj = nn.Linear(self.k_dim, hidden_dim)
        self.v_proj = nn.Linear(self.v_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if use_cls_token else None
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len + (1 if use_cls_token else 0), hidden_dim))

        ff_hidden_dim = ff_hidden_dim if ff_hidden_dim is not None else 4 * hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            activation=resolve_nn_activation(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
        )
        self.cross_attention_layers = nn.ModuleList(
            [
                _CrossAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.cross_final_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        if output_hidden_sizes:
            self.head = MLP(hidden_dim, self.output_dim, list(output_hidden_sizes), activation=activation)
        elif self.output_dim != hidden_dim:
            self.head = MLP(hidden_dim, self.output_dim, [hidden_dim], activation=activation)
        else:
            self.head = nn.Identity()

        self._reset_parameters()

    def _resolve_input_dim(
        self, input_dim: int | None, input_dims: Sequence[int] | None
    ) -> int:
        if input_dims is not None:
            total = int(sum(input_dims))
            if input_dim is not None and input_dim != total:
                raise ValueError(f"input_dim ({input_dim}) does not match sum(input_dims) ({total}).")
            return total
        if input_dim is None:
            raise ValueError("Either input_dim or input_dims must be provided.")
        return input_dim

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    @property
    def feature_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        x: torch.Tensor | Sequence[torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        return_sequence: bool = False,
        *,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a sequence or run cross-attention with separate Q/K/V inputs."""
        if q is not None or k is not None or v is not None:
            if x is not None:
                raise ValueError("Pass either x or q/k/v, not both.")
            if q is None or k is None or v is None:
                raise ValueError("q, k, and v must all be provided together.")
            return self._forward_cross_attention(q, k, v, padding_mask, key_padding_mask, return_sequence)
        if x is None:
            raise ValueError("Provide x or q/k/v.")

        x = self._combine_inputs(x)

        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dimension {self.input_dim}, got {x.shape[-1]}.")
        if x.shape[1] > self.max_seq_len:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_seq_len={self.max_seq_len}.")
        padding_mask = self._prepare_padding_mask(padding_mask, x.shape[:2])

        x = self.input_normalizer(x)
        x = self.input_proj(x)

        if self.use_cls_token:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            if padding_mask is not None:
                cls_mask = torch.zeros((padding_mask.shape[0], 1), dtype=torch.bool, device=padding_mask.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x = x + self.pos_embedding[:, : x.shape[1]]
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        if return_sequence:
            return self.head(x)

        pooled = self._pool_sequence(x, padding_mask)
        return self.head(pooled)

    def _forward_cross_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_padding_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        return_sequence: bool,
    ) -> torch.Tensor:
        q = self._prepare_sequence(q, self.q_dim, "q")
        k = self._prepare_sequence(k, self.k_dim, "k")
        v = self._prepare_sequence(v, self.v_dim, "v")

        if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
            raise ValueError("q, k, and v must have compatible batch dimensions; k and v must share shape.")
        if q.shape[1] > self.max_seq_len or k.shape[1] > self.max_seq_len:
            raise ValueError(f"Sequence lengths must not exceed max_seq_len={self.max_seq_len}.")

        query_padding_mask = self._prepare_padding_mask(query_padding_mask, q.shape[:2])
        key_padding_mask = self._prepare_padding_mask(key_padding_mask, k.shape[:2])

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        if self.use_cls_token:
            cls = self.cls_token.expand(q.shape[0], -1, -1)
            q = torch.cat([cls, q], dim=1)
            if query_padding_mask is not None:
                cls_mask = torch.zeros(
                    (query_padding_mask.shape[0], 1),
                    dtype=torch.bool,
                    device=query_padding_mask.device,
                )
                query_padding_mask = torch.cat([cls_mask, query_padding_mask], dim=1)

        q = q + self.pos_embedding[:, : q.shape[1]]
        k = k + self.pos_embedding[:, : k.shape[1]]
        v = v + self.pos_embedding[:, : v.shape[1]]

        for layer in self.cross_attention_layers:
            q = layer(q, k, v, key_padding_mask=key_padding_mask)
        q = self.cross_final_norm(q)

        if return_sequence:
            return self.head(q)
        return self.head(self._pool_sequence(q, query_padding_mask))

    def update_normalization(self, x: torch.Tensor | Sequence[torch.Tensor]) -> None:
        """Update the input normalizer with raw tokens."""
        if hasattr(self.input_normalizer, "update"):
            combined = self._combine_inputs(x)
            self.input_normalizer.update(combined.reshape(-1, combined.shape[-1]))

    def _combine_inputs(self, x: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            tensors = [x]
        else:
            tensors = list(x)
            if not tensors:
                raise ValueError("Transformer received an empty input sequence.")
            if self.input_dims is not None and len(tensors) != len(self.input_dims):
                raise ValueError(
                    f"Expected {len(self.input_dims)} input chunks from input_dims, got {len(tensors)}."
                )

        normalized: list[torch.Tensor] = []
        batch_shape: tuple[int, int] | None = None

        for idx, tensor in enumerate(tensors):
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            elif tensor.dim() != 3:
                raise ValueError(f"Input chunk {idx} must have shape (batch, time, dim) or (batch, dim).")

            if not self.batch_first:
                tensor = tensor.transpose(0, 1)

            if batch_shape is None:
                batch_shape = tensor.shape[:2]
            elif tensor.shape[:2] != batch_shape:
                raise ValueError("All input chunks must share the same batch and time dimensions.")

            if self.input_dims is not None and len(tensors) > 1 and tensor.shape[-1] != self.input_dims[idx]:
                raise ValueError(
                    f"Input chunk {idx} expected last dimension {self.input_dims[idx]}, got {tensor.shape[-1]}."
                )
            normalized.append(tensor)

        return torch.cat(normalized, dim=-1)

    def _prepare_sequence(self, x: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"{name} must have shape (batch, time, dim) or (batch, dim).")
        if not self.batch_first:
            x = x.transpose(0, 1)
        if x.shape[-1] != expected_dim:
            raise ValueError(f"Expected {name} last dimension {expected_dim}, got {x.shape[-1]}.")
        return x

    def _prepare_padding_mask(
        self,
        padding_mask: torch.Tensor | None,
        expected_shape: torch.Size,
    ) -> torch.Tensor | None:
        if padding_mask is None:
            return None
        if padding_mask.shape == expected_shape:
            return padding_mask
        if not self.batch_first and padding_mask.dim() == 2 and padding_mask.transpose(0, 1).shape == expected_shape:
            return padding_mask.transpose(0, 1)
        raise ValueError("padding_mask must have shape (batch, time).")

    def _pool_sequence(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if self.pooling == "cls":
            return x[:, 0]

        start_idx = 1 if self.use_cls_token else 0
        tokens = x[:, start_idx:]
        mask = padding_mask[:, start_idx:] if padding_mask is not None else None

        if self.pooling == "mean":
            if mask is None:
                return tokens.mean(dim=1)
            valid = (~mask).unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1)
            return (tokens * valid).sum(dim=1) / denom

        if self.pooling == "max":
            if mask is not None:
                tokens = tokens.masked_fill(mask.unsqueeze(-1), float("-inf"))
            pooled = tokens.amax(dim=1)
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
            return pooled

        raise RuntimeError(f"Unknown pooling mode: {self.pooling}")
