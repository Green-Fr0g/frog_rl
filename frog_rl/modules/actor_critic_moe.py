# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from frog_rl.networks import EmpiricalNormalization, MoE


class ActorCriticMoE(nn.Module):
    """Actor-critic policy with separate mixture-of-experts actor and critic networks."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        actor_num_experts: int = 4,
        critic_num_experts: int = 4,
        actor_gate_hidden_dims: tuple[int, ...] | list[int] = (),
        critic_gate_hidden_dims: tuple[int, ...] | list[int] = (),
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the MoE actor-critic policy.

        Args:
            obs: Observation dictionary.
            obs_groups: Mapping from observation sets to observation group names.
            num_actions: Number of actions produced by the actor.
            actor_obs_normalization: Whether to normalize actor observations.
            critic_obs_normalization: Whether to normalize critic observations.
            actor_hidden_dims: Hidden dimensions of every actor expert.
            critic_hidden_dims: Hidden dimensions of every critic expert.
            activation: Activation function used by the experts and gates.
            init_noise_std: Initial action standard deviation.
            noise_std_type: Action standard deviation parameterization, either ``"scalar"`` or ``"log"``.
            state_dependent_std: Whether the actor predicts its standard deviation from observations.
            actor_num_experts: Number of actor experts.
            critic_num_experts: Number of critic experts.
            actor_gate_hidden_dims: Hidden dimensions of the actor gate.
            critic_gate_hidden_dims: Hidden dimensions of the critic gate.
        """
        if kwargs:
            print(
                "ActorCriticMoE.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs))
            )
        super().__init__()

        self.obs_groups = obs_groups
        num_actor_obs = self._get_obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._get_obs_dim(obs, obs_groups["critic"])
        self.state_dependent_std = state_dependent_std

        # Actor and critic use independent expert ensembles.
        actor_output_dim: int | tuple[int, int] = (2, num_actions) if state_dependent_std else num_actions
        self.actor = MoE(
            num_actor_obs,
            actor_output_dim,
            actor_hidden_dims,
            activation,
            actor_num_experts,
            actor_gate_hidden_dims,
        )
        self.critic = MoE(
            num_critic_obs,
            1,
            critic_hidden_dims,
            activation,
            critic_num_experts,
            critic_gate_hidden_dims,
        )
        print(f"Actor MoE: {self.actor}")
        print(f"Critic MoE: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else torch.nn.Identity()
        )
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else torch.nn.Identity()
        )

        self.noise_std_type = noise_std_type
        if state_dependent_std:
            self._initialize_state_dependent_std(init_noise_std, num_actions)
        elif noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _get_obs_dim(obs: TensorDict, obs_groups: list[str]) -> int:
        """Compute the flattened dimension of a set of 1D observations."""
        obs_dim = 0
        for obs_group in obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"ActorCriticMoE only supports 1D observations, got shape {obs[obs_group].shape} "
                    f"for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return obs_dim

    def _initialize_state_dependent_std(self, init_noise_std: float, num_actions: int) -> None:
        """Initialize the standard-deviation channels in every actor expert."""
        for expert in self.actor.experts:
            linear_layers = [layer for layer in expert.modules() if isinstance(layer, nn.Linear)]
            output_layer = linear_layers[-1]
            nn.init.zeros_(output_layer.weight[num_actions:])
            if self.noise_std_type == "scalar":
                nn.init.constant_(output_layer.bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                nn.init.constant_(output_layer.bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
            mean, std_values = torch.unbind(mean_and_std, dim=-2)
            std = std_values if self.noise_std_type == "scalar" else torch.exp(std_values)
        else:
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        actor_output = self.actor(actor_obs)
        return actor_output[..., 0, :] if self.state_dependent_std else actor_output

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load policy parameters and report that training can be resumed."""
        super().load_state_dict(state_dict, strict=strict)
        return True
