# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class MoE(nn.Module):
    """Soft mixture-of-experts network.

    The module evaluates every expert on each forward pass and mixes their outputs with a learned gating network.
    It follows the same output-shaping conventions as :class:`frog_rl.networks.mlp.MLP`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        num_experts: int = 8,
        gate_hidden_dims: tuple[int, ...] | list[int] = (),
    ) -> None:
        """Initialize the MoE network."""
        super().__init__()

        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.num_experts = num_experts
        self.gate_hidden_dims = list(gate_hidden_dims)

        moe_output_dim = output_dim
        self.experts = nn.ModuleList(
            [self._build_network(input_dim, moe_output_dim, self.hidden_dims, activation) for _ in range(num_experts)]
        )
        self.gate = self._build_network(input_dim, num_experts, self.gate_hidden_dims, activation)

    @staticmethod
    def _build_network(
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> nn.Module:
        """Build a dense expert or gate network."""
        if len(hidden_dims) == 0:
            layers: list[nn.Module] = []
            if isinstance(output_dim, int):
                layers.append(nn.Linear(input_dim, output_dim))
            else:
                total_out_dim = reduce(lambda x, y: x * y, output_dim)
                layers.append(nn.Linear(input_dim, total_out_dim))
                layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))
            return nn.Sequential(*layers)
        return MLP(input_dim, output_dim, hidden_dims, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the mixture-of-experts forward pass."""
        gate_scores = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        for _ in range(expert_outputs.dim() - gate_scores.dim()):
            gate_scores = gate_scores.unsqueeze(-1)
        return torch.sum(expert_outputs * gate_scores, dim=1)
