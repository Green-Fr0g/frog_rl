"""WASABI-style adversarial motion prior extension for PPO."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tensordict import TensorDict

from frog_rl.algorithms.ppo import PPO
from frog_rl.env import VecEnv
from frog_rl.modules import EmpiricalNormalization
from frog_rl.storage.wasabi_storage import WasabiStorage
from frog_rl.utils import resolve_callable, resolve_optimizer


LossType = Literal["BCEWithLogitsLoss", "MSELoss", "WassersteinLoss"]
RewardType = Literal["log", "quad", "wasserstein"]
_WASABI_TERM_ORDER = ("projected_gravity", "joint_pos_rel", "joint_vel", "base_lin_vel", "base_ang_vel")


def _flatten_wasabi_state(state, state_key: str) -> torch.Tensor:
    """Flatten an InstinctLab-style non-concatenated state group."""
    if isinstance(state, torch.Tensor):
        if state.ndim != 2:
            raise ValueError(f"WASABI state '{state_key}' must be 2-D, got {tuple(state.shape)}.")
        return state
    if isinstance(state, Mapping) or (hasattr(state, "keys") and hasattr(state, "__getitem__")):
        missing = [term for term in _WASABI_TERM_ORDER if term not in state]
        if missing:
            raise ValueError(f"WASABI state '{state_key}' is missing terms: {missing}.")
        values = []
        for term in _WASABI_TERM_ORDER:
            value = state[term]
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                shape = getattr(value, "shape", None)
                raise ValueError(f"WASABI term '{state_key}.{term}' must be a 2-D Tensor, got {shape}.")
            values.append(value)
        return torch.cat(values, dim=-1)
    raise TypeError(f"WASABI state '{state_key}' must be a Tensor or mapping, got {type(state).__name__}.")


def _wasabi_state_dim(state, state_key: str) -> int:
    return _flatten_wasabi_state(state, state_key).shape[-1]


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
        self.policy_state_key = self.wasabi_cfg.setdefault("policy_state_key", "wasabi_policy")
        self.reference_state_key = self.wasabi_cfg.setdefault("reference_state_key", "wasabi_reference")
        self.task_reward_weight = float(self.wasabi_cfg.get("task_reward_weight", 1.0))
        self.reward_type: RewardType = self.wasabi_cfg.get("reward_type", "log")
        self.reward_coef = float(self.wasabi_cfg.get("reward_coef", 1.0))
        self.loss_type: LossType = self.wasabi_cfg.get("loss_type", "BCEWithLogitsLoss")
        self.loss_coef = float(self.wasabi_cfg.get("loss_coef", 1.0))
        self.gradient_penalty_coef = float(self.wasabi_cfg.get("gradient_penalty_coef", 10.0))
        self.gradient_tolerance = float(self.wasabi_cfg.get("gradient_tolerance", 0.0))
        self.weight_decay_coef = float(self.wasabi_cfg.get("weight_decay_coef", 0.0))
        self.logit_weight_decay_coef = float(self.wasabi_cfg.get("logit_weight_decay_coef", 0.0))
        self.discriminator_backbone_gradient_only = bool(
            self.wasabi_cfg.get("discriminator_backbone_gradient_only", False)
        )

        discriminator_class = resolve_callable(self.wasabi_cfg.get("discriminator_class_name", "WasabiDiscriminator"))
        discriminator_kwargs = dict(self.wasabi_cfg.get("discriminator_kwargs", {}))
        discriminator_kwargs.pop("state_dim", None)
        self.discriminator = discriminator_class(
            state_dim=self.wasabi_cfg["state_dim"],
            **{
                "hidden_dims": self.wasabi_cfg.get("hidden_dims", (512, 256)),
                "activation": self.wasabi_cfg.get("activation", "elu"),
                "normalize_input": self.wasabi_cfg.get("normalize_input", True),
                "normalization_until": self.wasabi_cfg.get("normalization_until", int(1e8)),
                **discriminator_kwargs,
            },
        ).to(device)

        optimizer_class = resolve_optimizer(self.wasabi_cfg.get("discriminator_optimizer", "adamw"))
        discriminator_optimizer_kwargs = dict(self.wasabi_cfg.get("discriminator_optimizer_kwargs", {}))
        discriminator_optimizer_kwargs.setdefault("lr", self.wasabi_cfg.get("learning_rate", self.learning_rate))
        self.discriminator_optimizer = optimizer_class(self.discriminator.parameters(), **discriminator_optimizer_kwargs)

        self.wasabi_storage = WasabiStorage(
            storage.num_transitions_per_env, storage.num_envs, self.wasabi_cfg["state_dim"], device
        )
        self._policy_state: torch.Tensor | None = None
        self._reference_state: torch.Tensor | None = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Capture discriminator inputs from the pre-action observation."""
        self._policy_state = _flatten_wasabi_state(obs[self.policy_state_key], self.policy_state_key)
        self._reference_state = _flatten_wasabi_state(obs[self.reference_state_key], self.reference_state_key)
        return super().act(obs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Store WASABI samples and add the discriminator reward to task reward."""
        if self._policy_state is None or self._reference_state is None:
            raise RuntimeError("WasabiPPO.process_env_step() must be called after act().")

        self.wasabi_storage.add(self._policy_state, self._reference_state, dones)
        imitation_reward, _ = self.discriminator.reward(self._policy_state, self.reward_type, self.reward_coef)
        rewards = self.task_reward_weight * rewards + imitation_reward.reshape_as(rewards)
        super().process_env_step(obs, rewards, dones, extras)
        self._policy_state = None
        self._reference_state = None

    def update(self) -> dict[str, float]:
        """Run PPO, then optimize the discriminator from the collected rollout."""
        loss_dict = super().update()
        if self.wasabi_storage.num_samples:
            self._update_discriminator(loss_dict)
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
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "WasabiPPO":
        """Resolve WASABI observation keys before constructing PPO components."""
        wasabi_cfg = cfg["algorithm"].get("wasabi_cfg")
        if wasabi_cfg is None:
            raise ValueError("WasabiPPO requires algorithm.wasabi_cfg.")

        wasabi_cfg = dict(wasabi_cfg)
        wasabi_cfg.setdefault("policy_state_key", "wasabi_policy")
        wasabi_cfg.setdefault("reference_state_key", "wasabi_reference")
        policy_key = wasabi_cfg["policy_state_key"]
        reference_key = wasabi_cfg["reference_state_key"]
        missing_keys = [key for key in (policy_key, reference_key) if key not in obs]
        if missing_keys:
            raise ValueError(f"WasabiPPO observations are missing {missing_keys}; available: {list(obs.keys())}")

        policy_dim = _wasabi_state_dim(obs[policy_key], policy_key)
        reference_dim = _wasabi_state_dim(obs[reference_key], reference_key)
        if policy_dim != reference_dim:
            raise ValueError(
                f"WASABI policy/reference state dimensions differ: {policy_dim} != {reference_dim}. "
                "Encode both through the same state specification."
            )

        wasabi_cfg["state_dim"] = policy_dim
        cfg["algorithm"]["wasabi_cfg"] = wasabi_cfg
        return PPO.construct_algorithm(obs, env, cfg, device)

    def _update_discriminator(self, loss_dict: dict[str, float]) -> None:
        """Optimize discriminator objectives migrated from the WASABI implementation."""
        self._update_normalizer()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        metrics = {
            "wasabi_discriminator_loss": 0.0,
            "wasabi_gradient_penalty": 0.0,
            "wasabi_weight_decay": 0.0,
            "wasabi_logit_weight_decay": 0.0,
            "wasabi_policy_logit": 0.0,
            "wasabi_reference_logit": 0.0,
        }

        for batch in self.wasabi_storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            policy_logits = self.discriminator(batch.policy_states)
            reference_logits = self.discriminator(batch.reference_states)
            discriminator_loss = self._classification_loss(policy_logits, reference_logits)
            gradient_penalty = self._gradient_penalty(batch.policy_states, batch.reference_states)
            weight_decay = sum(parameter.square().sum() for parameter in self.discriminator.parameters())
            logit_weight_decay = self._logit_weight_decay()
            loss = (
                self.loss_coef * discriminator_loss
                + gradient_penalty
                + self.weight_decay_coef * weight_decay
                + self.logit_weight_decay_coef * logit_weight_decay
            )

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self._reduce_discriminator_gradients()
            self.discriminator_optimizer.step()

            metrics["wasabi_discriminator_loss"] += discriminator_loss.item() / num_updates
            metrics["wasabi_gradient_penalty"] += gradient_penalty.item() / num_updates
            metrics["wasabi_weight_decay"] += weight_decay.item() / num_updates
            metrics["wasabi_logit_weight_decay"] += logit_weight_decay.item() / num_updates
            metrics["wasabi_policy_logit"] += policy_logits.mean().item() / num_updates
            metrics["wasabi_reference_logit"] += reference_logits.mean().item() / num_updates

        loss_dict.update(metrics)

    def _reduce_discriminator_gradients(self) -> None:
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

    def _classification_loss(self, policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "BCEWithLogitsLoss":
            policy_loss = F.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
            reference_loss = F.binary_cross_entropy_with_logits(reference_logits, torch.ones_like(reference_logits))
        elif self.loss_type == "MSELoss":
            policy_loss = F.mse_loss(policy_logits, -torch.ones_like(policy_logits))
            reference_loss = F.mse_loss(reference_logits, torch.ones_like(reference_logits))
        elif self.loss_type == "WassersteinLoss":
            policy_loss = policy_logits.mean()
            reference_loss = -reference_logits.mean()
        else:
            raise ValueError(f"Unsupported WASABI discriminator loss: {self.loss_type}")
        return 0.5 * (policy_loss + reference_loss)

    def _gradient_penalty(self, policy_states: torch.Tensor, reference_states: torch.Tensor) -> torch.Tensor:
        if self.gradient_penalty_coef <= 0.0:
            return torch.zeros((), device=self.device)

        states = torch.cat([policy_states, reference_states], dim=0).detach().requires_grad_(True)
        if (
            self.discriminator_backbone_gradient_only
            and getattr(self.discriminator, "encoders", None) is not None
            and hasattr(self.discriminator, "backbone_run")
        ):
            latent = self.discriminator.encoders(states).detach()
            latent.requires_grad = True
            logits = self.discriminator.backbone_run(latent)
            gradients = torch.autograd.grad(
                outputs=logits,
                inputs=latent,
                grad_outputs=torch.ones_like(logits),
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            logits = self.discriminator(states)
            gradients = torch.autograd.grad(
                outputs=logits,
                inputs=states,
                grad_outputs=torch.ones_like(logits),
                create_graph=True,
                retain_graph=True,
            )[0]

        excess_norm = torch.clamp(gradients.norm(2, dim=-1) - self.gradient_tolerance, min=0.0)
        return self.gradient_penalty_coef * excess_norm.square().mean()

    def _logit_weight_decay(self) -> torch.Tensor:
        if self.logit_weight_decay_coef <= 0.0:
            return torch.zeros((), device=self.device)
        if hasattr(self.discriminator, "logit"):
            return self.discriminator.logit.weight.square().sum()
        if hasattr(self.discriminator, "amp_linear"):
            return self.discriminator.amp_linear.weight.square().sum()
        return torch.zeros((), device=self.device)

    def _update_normalizer(self) -> None:
        normalizer = getattr(self.discriminator, "normalizer", None)
        if not isinstance(normalizer, EmpiricalNormalization):
            return

        policy_states, reference_states = self.wasabi_storage.states()
        states = torch.cat([policy_states, reference_states], dim=0)
        if self.is_multi_gpu and dist.is_initialized():
            self._update_normalizer_distributed(normalizer, states)
        else:
            normalizer.update(states)

    @staticmethod
    def _update_normalizer_distributed(normalizer: EmpiricalNormalization, states: torch.Tensor) -> None:
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
