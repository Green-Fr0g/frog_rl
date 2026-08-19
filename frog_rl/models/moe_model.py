# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from frog_rl.models.mlp_model import MLPModel
from frog_rl.modules import MLP
from frog_rl.utils import unpad_trajectories


class MoEModel(MLPModel):
    """Mixture-of-Experts (MoE) based neural model.

    This model is a drop-in replacement for :class:`~frog_rl.models.mlp_model.MLPModel` in which the MLP head is
    replaced by a mixture-of-experts layer. Observations are selected, concatenated, and (optionally) normalized exactly
    as in the MLP model, and the resulting latent is then processed by a set of expert networks whose outputs are mixed
    by a learned gating network. This is a "soft" (dense) MoE: every expert is evaluated and mixed on each forward pass,
    as opposed to a sparse top-k selection. The output of the model can be either deterministic or stochastic, in which
    case a distribution module is used to sample the outputs.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 8,
        gate_hidden_dims: tuple[int, ...] | list[int] = (),
    ) -> None:
        """Initialize the MoE-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of each expert MLP.
            activation: Activation function of the experts and gate.
            obs_normalization: Whether to normalize the observations before feeding them to the experts.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
            num_experts: Number of expert networks in the mixture-of-experts layer.
            gate_hidden_dims: Hidden dimensions of the gating network. Empty results in a plain linear gate.
        """
        self.num_experts = num_experts
        self.gate_hidden_dims = gate_hidden_dims

        # Initialize the parent MLP model (observation handling, normalization and distribution)
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        # Replace the default MLP head with a mixture-of-experts head. The output dimension must match what the
        # distribution expects (e.g. [2, output_dim] for heteroscedastic or Beta distributions).
        moe_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        latent_dim = self._get_latent_dim()
        self.experts = nn.ModuleList(
            [self._build_network(latent_dim, moe_output_dim, hidden_dims, activation) for _ in range(num_experts)]
        )
        self.gate = self._build_network(latent_dim, num_experts, gate_hidden_dims, activation)
        # Discard the plain MLP head built by the parent so its parameters are not optimized nor serialized.
        del self.mlp

        # Apply distribution-specific weight initialization to each expert (e.g. std / beta heads).
        if self.distribution is not None:
            for expert in self.experts:
                self.distribution.init_mlp_weights(expert)

    @staticmethod
    def _build_network(
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> nn.Module:
        """Build an expert or gate network, falling back to a linear layer when no hidden dims are given."""
        if len(hidden_dims) == 0:
            return nn.Linear(input_dim, output_dim)
        return MLP(input_dim, output_dim, hidden_dims, activation)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MoE model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MoE input latent
        latent = self.get_latent(obs, masks, hidden_state)
        # Mixture-of-experts forward pass
        mlp_output = self._moe_forward(latent)
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MoE output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def _moe_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix the expert outputs according to the softmax gating scores."""
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [..., num_experts]
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [..., num_experts, *output_dims]
        # Align the gating scores for broadcasting over the (possibly multi-dimensional) expert outputs.
        for _ in range(expert_outputs.dim() - gate_scores.dim()):
            gate_scores = gate_scores.unsqueeze(-1)
        return torch.sum(expert_outputs * gate_scores, dim=1)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchMoEModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMoEModel(self, verbose)


class _TorchMoEModel(nn.Module):
    """Exportable MoE model for JIT."""

    def __init__(self, model: MoEModel) -> None:
        """Create a TorchScript-friendly copy of an MoEModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.experts = copy.deepcopy(model.experts)
        self.gate = copy.deepcopy(model.gate)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        x = self.obs_normalizer(x)
        gate_scores = F.softmax(self.gate(x), dim=-1)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        for _ in range(expert_outputs.dim() - gate_scores.dim()):
            gate_scores = gate_scores.unsqueeze(-1)
        out = torch.sum(expert_outputs * gate_scores, dim=1)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MoE exports)."""
        pass


class _OnnxMoEModel(nn.Module):
    """Exportable MoE model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: MoEModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MoEModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.experts = copy.deepcopy(model.experts)
        self.gate = copy.deepcopy(model.gate)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(x)
        gate_scores = F.softmax(self.gate(x), dim=-1)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        for _ in range(expert_outputs.dim() - gate_scores.dim()):
            gate_scores = gate_scores.unsqueeze(-1)
        out = torch.sum(expert_outputs * gate_scores, dim=1)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
