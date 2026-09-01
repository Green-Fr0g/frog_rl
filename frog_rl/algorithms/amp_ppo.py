# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from frog_rl.algorithms.ppo import PPO
from frog_rl.modules import ActorCritic, ActorCriticMoE, ActorCriticRecurrent, Discriminator
from frog_rl.modules import resolve_rnd_config, resolve_symmetry_config
from frog_rl.networks import EmpiricalNormalization
from frog_rl.storage import AMPStorage
from frog_rl.utils import resolve_obs_groups, resolve_optimizer, string_to_callable


class AMPPPO(PPO):
    """PPO with adversarial motion priors."""

    policy: ActorCritic | ActorCriticMoE | ActorCriticRecurrent

    def __init__(self, policy, amp_cfg: dict[str, Any], device: str = "cpu", **ppo_kwargs) -> None:
        super().__init__(policy, device=device, **ppo_kwargs)

        self.amp_cfg = dict(amp_cfg)
        self.expert_state_key = self.amp_cfg["expert_state_key"]
        self.state_dim = int(self.amp_cfg["state_dim"])
        self._current_amp_state: torch.Tensor | None = None

        if "motion_loader_class_name" not in self.amp_cfg:
            raise ValueError("AMPPPO requires amp_cfg.motion_loader_class_name.")

        motion_loader_class = string_to_callable(self.amp_cfg["motion_loader_class_name"])
        motion_loader_kwargs = dict(self.amp_cfg.get("motion_loader_kwargs", {}))
        motion_loader_kwargs.setdefault("device", self.device)
        motion_loader_kwargs.setdefault("time_between_frames", self.amp_cfg.get("time_between_frames", 0.02))

        self.amp_data = motion_loader_class(**motion_loader_kwargs)
        expert_state_dim = getattr(self.amp_data, "state_dim", None)
        if expert_state_dim is not None and expert_state_dim != self.state_dim:
            raise ValueError(
                f"AMP expert state dimension mismatch: observation '{self.expert_state_key}' has "
                f"dimension {self.state_dim}, motion loader has {expert_state_dim}."
            )

        self.discriminator = Discriminator(
            self.amp_cfg["discriminator_input_dim"],
            self.amp_cfg["amp_reward_coef"],
            self.amp_cfg["amp_discr_hidden_dims"],
            self.device,
            self.amp_cfg.get("amp_task_reward_lerp", 0.0),
        ).to(self.device)

        self.amp_normalizer = EmpiricalNormalization(
            shape=self.state_dim,
            until=int(self.amp_cfg.get("normalization_until", 1.0e8)),
        ).to(self.device)

        self.amp_storage = AMPStorage(
            self.state_dim,
            int(self.amp_cfg.get("amp_replay_buffer_size", 1000000)),
            self.device,
        )

        optimizer_class = resolve_optimizer(self.amp_cfg.get("discriminator_optimizer", "adam"))
        self.discriminator_optimizer = optimizer_class(
            [
                {
                    "params": self.discriminator.trunk.parameters(),
                    "weight_decay": self.amp_cfg.get("amp_trunk_weight_decay", 1e-3),
                },
                {
                    "params": self.discriminator.amp_linear.parameters(),
                    "weight_decay": self.amp_cfg.get("amp_head_weight_decay", 1e-2),
                },
            ],
            lr=self.amp_cfg.get("discriminator_lr", 1e-3),
        )

        self.grad_pen_coef = self.amp_cfg.get("grad_pen_coef", 10.0)

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env, cfg: dict, device: str, multi_gpu_cfg: dict | None = None
    ) -> "AMPPPO":
        """Construct an AMP-aware PPO algorithm from the runner config."""
        alg_cfg = deepcopy(cfg["algorithm"])
        policy_cfg = deepcopy(cfg["policy"])
        obs_groups = deepcopy(cfg.get("obs_groups", {}))

        if cfg.get("empirical_normalization") is not None:
            if policy_cfg.get("actor_obs_normalization") is None:
                policy_cfg["actor_obs_normalization"] = cfg["empirical_normalization"]
            if policy_cfg.get("critic_obs_normalization") is None:
                policy_cfg["critic_obs_normalization"] = cfg["empirical_normalization"]

        default_sets = ["critic", "amp_state"]
        if alg_cfg.get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        obs_groups = resolve_obs_groups(obs, obs_groups, default_sets)

        alg_cfg = resolve_rnd_config(alg_cfg, obs, obs_groups, env)
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        amp_cfg = deepcopy(alg_cfg.pop("amp_cfg"))
        expert_state_key = amp_cfg["expert_state_key"]
        if expert_state_key not in obs:
            raise ValueError(
                f"AMPPPO requires the '{expert_state_key}' observation group, but it is not present. "
                f"Available observations: {list(obs.keys())}"
            )
        amp_cfg["expert_state_key"] = expert_state_key
        amp_cfg["state_dim"] = int(obs[expert_state_key].shape[-1])
        amp_cfg["discriminator_input_dim"] = 2 * amp_cfg["state_dim"]

        motion_loader_kwargs = dict(amp_cfg["motion_loader_kwargs"])
        amp_cfg["motion_loader_kwargs"] = motion_loader_kwargs

        policy_class = eval(policy_cfg.pop("class_name"))
        actor_critic = policy_class(obs, obs_groups, env.num_actions, **policy_cfg).to(device)

        alg_cfg.pop("class_name", None)
        alg = AMPPPO(actor_critic, amp_cfg, device=device, multi_gpu_cfg=multi_gpu_cfg, **alg_cfg)
        alg.init_storage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions])
        return alg

    def act(self, obs: TensorDict) -> torch.Tensor:
        self._current_amp_state = obs[self.expert_state_key].detach().clone()
        return super().act(obs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self._current_amp_state is None:
            raise RuntimeError("AMPPPO.process_env_step() must be called after act().")

        next_amp_state = obs[self.expert_state_key].clone()
        reset_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            next_amp_state[reset_env_ids] = self._current_amp_state[reset_env_ids]

        self.amp_storage.insert(self._current_amp_state, next_amp_state)
        amp_reward, _ = self.discriminator.predict_amp_reward(
            self._current_amp_state, next_amp_state, rewards, normalizer=self.amp_normalizer
        )
        rewards.copy_(amp_reward)

        super().process_env_step(obs, rewards, dones, extras)
        self._current_amp_state = next_amp_state.detach().clone()

    def update(self) -> dict[str, float]:  # noqa: C901
        loss_dict = super().update()

        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0

        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        num_updates = self.num_learning_epochs * self.num_mini_batches
        amp_policy_generator = self.amp_storage.feed_forward_generator(num_updates, mini_batch_size)
        amp_expert_generator = self.amp_data.feed_forward_generator(num_updates, mini_batch_size)

        for (policy_state, policy_next_state), (expert_state, expert_next_state) in zip(
            amp_policy_generator, amp_expert_generator
        ):
            policy_state_norm = self.amp_normalizer(policy_state)
            policy_next_state_norm = self.amp_normalizer(policy_next_state)
            expert_state_norm = self.amp_normalizer(expert_state)
            expert_next_state_norm = self.amp_normalizer(expert_next_state)

            policy_d = self.discriminator(torch.cat([policy_state_norm, policy_next_state_norm], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state_norm, expert_next_state_norm], dim=-1))
            expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = F.mse_loss(policy_d, -torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state_norm, expert_next_state_norm, lambda_=self.grad_pen_coef
            )
            loss = amp_loss + grad_pen_loss

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_discriminator_parameters()
            self.discriminator_optimizer.step()

            self.amp_normalizer.update(policy_state.detach())
            self.amp_normalizer.update(policy_next_state.detach())
            self.amp_normalizer.update(expert_state.detach())
            self.amp_normalizer.update(expert_next_state.detach())

            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        loss_dict["amp"] = mean_amp_loss / num_updates
        loss_dict["amp_grad_pen"] = mean_grad_pen_loss / num_updates
        loss_dict["amp_policy_pred"] = mean_policy_pred / num_updates
        loss_dict["amp_expert_pred"] = mean_expert_pred / num_updates
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()
        self.amp_normalizer.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()
        self.amp_normalizer.eval()

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        if not self.is_multi_gpu:
            return
        model_params = [self.discriminator.state_dict(), self.amp_normalizer.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.discriminator.load_state_dict(model_params[0])
        self.amp_normalizer.load_state_dict(model_params[1])

    def reduce_discriminator_parameters(self) -> None:
        params = [param for param in self.discriminator.parameters() if param.grad is not None]
        if not params:
            return
        flat_grads = torch.cat([param.grad.detach().reshape(-1) for param in params])
        torch.distributed.all_reduce(flat_grads, op=torch.distributed.ReduceOp.SUM)
        flat_grads /= self.gpu_world_size
        offset = 0
        for param in params:
            numel = param.numel()
            param.grad.data.copy_(flat_grads[offset : offset + numel].view_as(param.grad.data))
            offset += numel

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        saved_dict["discriminator_optimizer_state_dict"] = self.discriminator_optimizer.state_dict()
        saved_dict["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_optimizer: bool = True, strict: bool = True) -> bool:
        if "model_state_dict" in loaded_dict:
            resumed_training = super().load(loaded_dict, load_optimizer=load_optimizer, strict=strict)
        elif "actor_state_dict" in loaded_dict:
            from collections import OrderedDict

            merged = OrderedDict()
            for k, v in loaded_dict["actor_state_dict"].items():
                new_key = k.replace("mlp.", "actor.", 1) if k.startswith("mlp.") else k
                merged[new_key] = v
            if "critic_state_dict" in loaded_dict:
                for k, v in loaded_dict["critic_state_dict"].items():
                    new_key = k.replace("mlp.", "critic.", 1) if k.startswith("mlp.") else k
                    merged[new_key] = v
            if "distribution.std_param" in merged:
                merged["std"] = merged.pop("distribution.std_param")
            self.policy.load_state_dict(merged, strict=False)
            if load_optimizer and "optimizer_state_dict" in loaded_dict:
                self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            resumed_training = True
        else:
            raise KeyError(f"Checkpoint has no recognized model keys. Found: {list(loaded_dict.keys())}")

        discriminator_state = loaded_dict.get("discriminator_state_dict") or loaded_dict.get(
            "amp_discriminator_state_dict"
        )
        if discriminator_state is not None:
            self.discriminator.load_state_dict(discriminator_state, strict=strict)

        normalizer_state = loaded_dict.get("amp_normalizer_state_dict") or loaded_dict.get("amp_normalizer")
        if normalizer_state is not None:
            if hasattr(normalizer_state, "state_dict"):
                normalizer_state = normalizer_state.state_dict()
            self.amp_normalizer.load_state_dict(normalizer_state, strict=strict)

        if load_optimizer:
            optimizer_state = loaded_dict.get("discriminator_optimizer_state_dict") or loaded_dict.get(
                "amp_discriminator_optimizer_state_dict"
            )
            if optimizer_state is not None:
                self.discriminator_optimizer.load_state_dict(optimizer_state)

        return resumed_training
