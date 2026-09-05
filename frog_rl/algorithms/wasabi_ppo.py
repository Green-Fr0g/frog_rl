"""WASABI-style adversarial motion prior extension for PPO."""

from __future__ import annotations

from typing import Literal

import torch
import torch.distributed as dist
from tensordict import TensorDict

from frog_rl.algorithms.ppo import PPO
from frog_rl.env import VecEnv
from frog_rl.networks import EmpiricalNormalization
from frog_rl.storage.rollout_storage_wasabi import WasabiStorage
from frog_rl.utils import resolve_optimizer, string_to_callable


LossType = Literal["BCEWithLogitsLoss", "MSELoss", "WassersteinLoss"]
RewardType = Literal["log", "quad", "wasserstein"]


class WasabiPPO(PPO):
    """PPO with a configurable state-pair discriminator."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        wasabi_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, device=device, multi_gpu_cfg=multi_gpu_cfg, **kwargs)
        if wasabi_cfg is None:
            raise ValueError("WasabiPPO requires an algorithm.wasabi_cfg configuration.")

        self.wasabi_cfg = dict(wasabi_cfg)
        self.policy_state_key = self.wasabi_cfg["wasabi_policy_state_key"]
        self.reference_state_key = self.wasabi_cfg["wasabi_reference_state_key"]
        self.task_reward_weight = float(self.wasabi_cfg["wasabi_task_reward_weight"])
        self.reward_type: RewardType = self.wasabi_cfg["wasabi_reward_type"]
        self.reward_coef = float(self.wasabi_cfg["wasabi_reward_coef"])
        self.loss_type: LossType = self.wasabi_cfg["wasabi_loss_type"]
        self.loss_coef = float(self.wasabi_cfg["wasabi_loss_coef"])
        self.grad_pen_coef = float(self.wasabi_cfg["wasabi_grad_pen_coef"])
        self.grad_tolerance = float(self.wasabi_cfg["wasabi_grad_tolerance"])
        self.trunk_weight_decay = float(self.wasabi_cfg["wasabi_trunk_weight_decay"])
        self.head_weight_decay = float(self.wasabi_cfg["wasabi_head_weight_decay"])

        discriminator_class = string_to_callable(self.wasabi_cfg.get("wasabi_discriminator_class_name", "WasabiDiscriminator"))
        discriminator_kwargs = dict(self.wasabi_cfg.get("wasabi_discriminator_kwargs", {}))
        discriminator_kwargs.pop("state_dim", None)
        self.discriminator = discriminator_class(
            state_dim=self.wasabi_cfg["state_dim"],
            **{
                "hidden_dims": self.wasabi_cfg["wasabi_discr_hidden_dims"],
                "activation": self.wasabi_cfg["wasabi_discr_activation"],
                "normalize_input": self.wasabi_cfg["wasabi_normalize_input"],
                "normalization_until": self.wasabi_cfg["wasabi_normalization_until"],
                **discriminator_kwargs,
            },
        ).to(device)

        optimizer_class = resolve_optimizer(self.wasabi_cfg["wasabi_discriminator_optimizer"])
        discriminator_optimizer_kwargs = dict(self.wasabi_cfg.get("wasabi_discriminator_optimizer_kwargs", {}))
        discriminator_optimizer_kwargs.setdefault("lr", self.wasabi_cfg["wasabi_discriminator_lr"])
        self.discriminator_optimizer = optimizer_class(self.discriminator.parameters(), **discriminator_optimizer_kwargs)

        self.wasabi_storage = WasabiStorage(
            storage.num_transitions_per_env, storage.num_envs, self.wasabi_cfg["state_dim"], device
        )
        self.current_state: torch.Tensor | None = None
        self.reference_state: torch.Tensor | None = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Capture discriminator inputs from the pre-action observation."""
        self.current_state = obs[self.policy_state_key]
        self.reference_state = obs[self.reference_state_key]
        return super().act(obs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Store WASABI samples and add the discriminator reward to task reward."""
        if self.current_state is None or self.reference_state is None:
            raise RuntimeError("WasabiPPO.process_env_step() must be called after act().")

        self.wasabi_storage.add(self.current_state, self.reference_state, dones)
        imitation_reward, _ = self.discriminator.reward(self.current_state, self.reward_type, self.reward_coef)
        rewards = self.task_reward_weight * rewards + imitation_reward.reshape_as(rewards)
        super().process_env_step(obs, rewards, dones, extras)
        self.current_state = None
        self.reference_state = None

    def update(self) -> dict[str, float]:
        """Run PPO and then optimize the discriminator from the collected rollout."""
        loss_dict = super().update()

        if self.wasabi_storage.num_samples == 0:
            self.wasabi_storage.clear()
            return loss_dict

        self.update_normalizer()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        metrics = {
            "wasabi_discriminator_loss": 0.0,
            "wasabi_gradient_penalty": 0.0,
            "wasabi_weight_decay": 0.0,
            "wasabi_logit_weight_decay": 0.0,
            "wasabi_policy_logit": 0.0,
            "wasabi_reference_logit": 0.0,
        }

        for current_states, reference_states, dones in self.wasabi_storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            policy_logits = self.discriminator(current_states)
            reference_logits = self.discriminator(reference_states)
            if self.loss_type == "BCEWithLogitsLoss":
                policy_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    policy_logits, torch.zeros_like(policy_logits)
                )
                reference_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    reference_logits, torch.ones_like(reference_logits)
                )
            elif self.loss_type == "MSELoss":
                policy_loss = torch.nn.functional.mse_loss(policy_logits, -torch.ones_like(policy_logits))
                reference_loss = torch.nn.functional.mse_loss(reference_logits, torch.ones_like(reference_logits))
            elif self.loss_type == "WassersteinLoss":
                policy_loss = policy_logits.mean()
                reference_loss = -reference_logits.mean()
            else:
                raise ValueError(f"Unsupported WASABI discriminator loss: {self.loss_type}")
            discriminator_loss = 0.5 * (policy_loss + reference_loss)

            gradient_penalty = self.discriminator.compute_grad_pen(
                current_states,
                reference_states,
                lambda_=self.grad_pen_coef,
                gradient_tolerance=self.grad_tolerance,
            )
            weight_decay = sum(parameter.square().sum() for parameter in self.discriminator.parameters())
            if self.head_weight_decay <= 0.0:
                logit_weight_decay = torch.zeros((), device=self.device)
            logit_weight_decay = self.discriminator.logit.weight.square().sum()
            loss = (
                self.loss_coef * discriminator_loss
                + gradient_penalty
                + self.trunk_weight_decay * weight_decay
                + self.head_weight_decay * logit_weight_decay
            )

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_discriminator_parameters()
            self.discriminator_optimizer.step()

            metrics["wasabi_discriminator_loss"] += discriminator_loss.item() / num_updates
            metrics["wasabi_gradient_penalty"] += gradient_penalty.item() / num_updates
            metrics["wasabi_weight_decay"] += weight_decay.item() / num_updates
            metrics["wasabi_logit_weight_decay"] += logit_weight_decay.item() / num_updates
            metrics["wasabi_policy_logit"] += policy_logits.mean().item() / num_updates
            metrics["wasabi_reference_logit"] += reference_logits.mean().item() / num_updates

        loss_dict.update(metrics)
        self.wasabi_storage.clear()
        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for the policy and discriminator."""
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for the policy and discriminator."""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """Return a dict of all models and states for saving."""
        saved_dict = super().save()
        saved_dict["wasabi_discriminator_state_dict"] = self.discriminator.state_dict()
        saved_dict["wasabi_discriminator_optimizer_state_dict"] = self.discriminator_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load the discriminator and optimizer in addition to the policy models."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        discriminator_state = loaded_dict.get("wasabi_discriminator_state_dict") or loaded_dict.get("discriminator")
        if discriminator_state is not None:
            self.discriminator.load_state_dict(discriminator_state, strict=strict)
        optimizer_state = loaded_dict.get("wasabi_discriminator_optimizer_state_dict") or loaded_dict.get(
            "discriminator_optimizer"
        )
        if optimizer_state is not None:
            self.discriminator_optimizer.load_state_dict(optimizer_state)
        return load_iteration

    def broadcast_parameters(self) -> None:
        """Broadcast policy and discriminator parameters to all GPUs."""
        super().broadcast_parameters()
        state = [self.discriminator.state_dict()]
        torch.distributed.broadcast_object_list(state, src=0)
        self.discriminator.load_state_dict(state[0])

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str, multi_gpu_cfg: dict | None = None
    ) -> "WasabiPPO":
        """Resolve WASABI observation keys before constructing PPO components."""
        wasabi_cfg = cfg["algorithm"].get("wasabi_cfg")
        if wasabi_cfg is None:
            raise ValueError("WasabiPPO requires algorithm.wasabi_cfg.")

        wasabi_cfg = dict(wasabi_cfg)
        policy_key = wasabi_cfg["wasabi_policy_state_key"]
        reference_key = wasabi_cfg["wasabi_reference_state_key"]
        missing_keys = [key for key in (policy_key, reference_key) if key not in obs]
        if missing_keys:
            raise ValueError(f"WasabiPPO observations are missing {missing_keys}; available: {list(obs.keys())}")

        policy_state = obs[policy_key]
        reference_state = obs[reference_key]
        if not isinstance(policy_state, torch.Tensor) or policy_state.ndim != 2:
            raise ValueError(f"WASABI state '{policy_key}' must be a 2-D Tensor, got {getattr(policy_state, 'shape', None)}.")
        if not isinstance(reference_state, torch.Tensor) or reference_state.ndim != 2:
            raise ValueError(
                f"WASABI state '{reference_key}' must be a 2-D Tensor, got {getattr(reference_state, 'shape', None)}."
            )
        policy_dim = policy_state.shape[-1]
        reference_dim = reference_state.shape[-1]
        if policy_dim != reference_dim:
            raise ValueError(
                f"WASABI policy/reference state dimensions differ: {policy_dim} != {reference_dim}. "
                "Encode both through the same state specification."
            )

        wasabi_cfg["state_dim"] = policy_dim
        cfg["algorithm"]["wasabi_cfg"] = wasabi_cfg
        return PPO.construct_algorithm(obs, env, cfg, device, multi_gpu_cfg)

    def reduce_discriminator_parameters(self) -> None:
        """Average discriminator gradients across GPUs."""
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

    def update_normalizer(self) -> None:
        normalizer = getattr(self.discriminator, "normalizer", None)
        if not isinstance(normalizer, EmpiricalNormalization):
            return

        current_states, reference_states = self.wasabi_storage.states()
        states = torch.cat([current_states, reference_states], dim=0)
        if self.is_multi_gpu and dist.is_initialized():
            self.update_normalizer_distributed(normalizer, states)
        else:
            normalizer.update(states)

    @staticmethod
    def update_normalizer_distributed(normalizer: EmpiricalNormalization, states: torch.Tensor) -> None:
        """Merge raw moments across ranks before updating the discriminator normalizer."""
        if normalizer.until is not None and normalizer.count >= normalizer.until:
            return

        count = torch.tensor(float(states.shape[0]), device=states.device)
        summed = states.sum(dim=0, keepdim=True)
        squared_sum = states.square().sum(dim=0, keepdim=True)
        dist.all_reduce(count)
        dist.all_reduce(summed)
        dist.all_reduce(squared_sum)

        batch_mean = summed / count
        batch_var = (squared_sum / count - batch_mean.square()).clamp_min(0.0)
        previous_count = normalizer.count.to(dtype=states.dtype)
        total_count = previous_count + count
        delta = batch_mean - normalizer._mean
        normalizer._mean += delta * (count / total_count)
        normalizer._var = (
            normalizer._var * previous_count + batch_var * count + delta.square() * previous_count * count / total_count
        ) / total_count
        normalizer._std = torch.sqrt(normalizer._var)
        normalizer.count += count.to(dtype=normalizer.count.dtype)
