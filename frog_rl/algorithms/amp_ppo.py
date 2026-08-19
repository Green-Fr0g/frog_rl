"""AMP (Adversarial Motion Priors) extension for PPO."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from frog_rl.algorithms.amp_discriminator import AMPDiscriminator
from frog_rl.algorithms.ppo import PPO
from frog_rl.modules.normalization import EmpiricalNormalization
from frog_rl.storage import AMPStorage
from frog_rl.utils import resolve_callable, resolve_optimizer


def resolve_amp_config(alg_cfg: dict, obs, env) -> dict:
    """Resolve the AMP configuration."""
    amp_cfg = alg_cfg.get("amp_cfg")
    if amp_cfg is None:
        alg_cfg["amp_cfg"] = None
        return alg_cfg

    amp_cfg = dict(amp_cfg)
    state_key = amp_cfg.get("expert_state_key", "amp_state")
    if not isinstance(state_key, str) or not state_key:
        raise ValueError("AMPPPO requires a non-empty amp_cfg.expert_state_key.")
    if state_key not in obs:
        raise ValueError(
            f"AMPPPO requires the '{state_key}' observation group, "
            f"but it is not present. Available observations: {list(obs.keys())}"
        )
    state = obs[state_key]
    if not isinstance(state, torch.Tensor) or state.ndim != 2:
        raise ValueError(
            f"AMPPPO observation '{state_key}' must be a 2-D Tensor, "
            f"got {type(state).__name__} with shape {getattr(state, 'shape', None)}."
        )
    amp_cfg["expert_state_key"] = state_key
    amp_cfg["state_dim"] = state.shape[-1]
    amp_cfg["discriminator_input_dim"] = 2 * amp_cfg["state_dim"]
    amp_cfg.setdefault("time_between_frames", env.unwrapped.step_dt)
    alg_cfg["amp_cfg"] = amp_cfg
    return alg_cfg


class AMPPPO(PPO):
    """PPO with a transition-based AMP discriminator."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        amp_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, device=device, multi_gpu_cfg=multi_gpu_cfg, **kwargs)

        if amp_cfg is None:
            raise ValueError("The AMP configuration 'amp_cfg' is required for AMPPPO.")

        self.amp_cfg = amp_cfg
        self.expert_state_key = amp_cfg.get("expert_state_key", "amp_state")
        self.state_dim = amp_cfg["state_dim"]
        self.discriminator = AMPDiscriminator(
            input_dim=amp_cfg["discriminator_input_dim"],
            amp_reward_coef=amp_cfg.get("amp_reward_coef", 1.0),
            hidden_layer_sizes=amp_cfg.get("amp_discr_hidden_dims", [1024, 512]),
            device=device,
            activation=amp_cfg.get("amp_discr_activation", "relu"),
            task_reward_lerp=amp_cfg.get("amp_task_reward_lerp", 0.0),
        )
        self.amp_storage = AMPStorage(self.state_dim, amp_cfg.get("amp_replay_buffer_size", 1000000), device)
        self.amp_normalizer = EmpiricalNormalization(shape=self.state_dim, until=int(1.0e8)).to(device)
        motion_loader_class_name = amp_cfg.get("motion_loader_class_name")
        if not motion_loader_class_name:
            raise ValueError("AMPPPO requires amp_cfg.motion_loader_class_name.")
        motion_loader_class = resolve_callable(motion_loader_class_name)
        motion_loader_kwargs = dict(amp_cfg.get("motion_loader_kwargs", {}))
        if "motion_files" not in motion_loader_kwargs and "amp_motion_files" in amp_cfg:
            motion_loader_kwargs["motion_files"] = amp_cfg["amp_motion_files"]
        self.amp_data = motion_loader_class(
            device=device,
            time_between_frames=amp_cfg.get("time_between_frames", 0.02),
            **motion_loader_kwargs,
        )
        expert_state_dim = getattr(self.amp_data, "state_dim", None)
        if expert_state_dim is not None and expert_state_dim != self.state_dim:
            raise ValueError(
                f"AMP expert state dimension mismatch: observation '{self.expert_state_key}' has "
                f"dimension {self.state_dim}, motion loader has {expert_state_dim}."
            )

        disc_opt_class = resolve_optimizer(amp_cfg.get("discriminator_optimizer", "adam"))
        self.discriminator_optimizer = disc_opt_class(
            [
                {
                    "params": self.discriminator.trunk.parameters(),
                    "weight_decay": amp_cfg.get("amp_trunk_weight_decay", 1e-3),
                },
                {
                    "params": self.discriminator.amp_linear.parameters(),
                    "weight_decay": amp_cfg.get("amp_head_weight_decay", 1e-2),
                },
            ],
            lr=amp_cfg.get("discriminator_lr", 1e-3),
        )

        self.grad_pen_coef = amp_cfg.get("grad_pen_coef", 10.0)
        self._current_amp_state = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Cache the current AMP state before sampling actions."""
        self._current_amp_state = obs[self.expert_state_key]
        return super().act(obs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Store the AMP transition and replace the task reward with AMP reward."""
        if self._current_amp_state is None:
            raise RuntimeError("AMPPPO.process_env_step() must be called after act().")

        next_amp_state = obs[self.expert_state_key]
        terminal_key = f"terminal_{self.expert_state_key}s"
        if terminal_key in extras:
            reset_env_ids = (dones > 0).flatten().nonzero(as_tuple=False).flatten()
            next_amp_state = next_amp_state.clone()
            next_amp_state[reset_env_ids] = extras[terminal_key][reset_env_ids]

        self.amp_storage.insert(self._current_amp_state, next_amp_state)
        amp_reward, _ = self.discriminator.reward(self._current_amp_state, next_amp_state, rewards, self.amp_normalizer)
        super().process_env_step(obs, amp_reward.reshape(-1), dones, extras)
        self._current_amp_state = next_amp_state

    def update(self) -> dict[str, float]:
        """Run PPO updates followed by discriminator training."""
        loss_dict = super().update()
        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        if self.amp_storage.num_samples >= mini_batch_size:
            self._train_discriminator(mini_batch_size, loss_dict)
        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for policy, discriminator, and AMP normalizer."""
        super().train_mode()
        self.discriminator.train()
        self.amp_normalizer.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for policy and discriminator."""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """Save policy, discriminator, optimizer, and AMP normalizer states."""
        saved_dict = super().save()
        saved_dict["amp_discriminator_state_dict"] = self.discriminator.state_dict()
        saved_dict["amp_discriminator_optimizer_state_dict"] = self.discriminator_optimizer.state_dict()
        saved_dict["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load the discriminator and AMP normalizer in addition to the policy."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)

        discriminator_state = loaded_dict.get("amp_discriminator_state_dict") or loaded_dict.get(
            "discriminator_state_dict"
        )
        if discriminator_state is not None:
            self.discriminator.load_state_dict(discriminator_state, strict=strict)

        optimizer_state = loaded_dict.get("amp_discriminator_optimizer_state_dict") or loaded_dict.get(
            "discriminator_optimizer_state_dict"
        )
        if optimizer_state is not None:
            self.discriminator_optimizer.load_state_dict(optimizer_state)

        normalizer_state = loaded_dict.get("amp_normalizer_state_dict") or loaded_dict.get("amp_normalizer")
        if normalizer_state is not None:
            self.amp_normalizer.load_state_dict(normalizer_state)

        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> "AMPPPO":
        """Construct the AMP-PPO algorithm."""
        if "policy" in cfg:
            policy_cfg = dict(cfg.pop("policy"))
            distribution_class = (
                "frog_rl.modules.distribution:HeteroscedasticGaussianDistribution"
                if policy_cfg.get("state_dependent_std", False)
                else "frog_rl.modules.distribution:GaussianDistribution"
            )

            cfg["actor"] = {
                "class_name": "frog_rl.models.mlp_model:MLPModel",
                "hidden_dims": policy_cfg["actor_hidden_dims"],
                "activation": policy_cfg["activation"],
                "obs_normalization": policy_cfg["actor_obs_normalization"],
                "distribution_cfg": {
                    "class_name": distribution_class,
                    "init_std": policy_cfg["init_noise_std"],
                    "std_type": policy_cfg.get("noise_std_type", "scalar"),
                },
            }
            cfg["critic"] = {
                "class_name": "frog_rl.models.mlp_model:MLPModel",
                "hidden_dims": policy_cfg["critic_hidden_dims"],
                "activation": policy_cfg["activation"],
                "obs_normalization": policy_cfg["critic_obs_normalization"],
            }

        amp_cfg = cfg["algorithm"].get("amp_cfg")
        if amp_cfg is not None:
            state_key = amp_cfg.get("expert_state_key", "amp_state")
            if state_key not in cfg["obs_groups"]:
                cfg["obs_groups"][state_key] = [state_key]
        cfg["algorithm"] = resolve_amp_config(cfg["algorithm"], obs, env)
        return PPO.construct_algorithm(obs, env, cfg, device)

    def broadcast_parameters(self) -> None:
        """Broadcast actor, critic, discriminator, and AMP normalizer parameters."""
        super().broadcast_parameters()
        state_dicts = [self.discriminator.state_dict(), self.amp_normalizer.state_dict()]
        torch.distributed.broadcast_object_list(state_dicts, src=0)
        self.discriminator.load_state_dict(state_dicts[0])
        self.amp_normalizer.load_state_dict(state_dicts[1])

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

    def _train_discriminator(self, mini_batch_size: int, loss_dict: dict) -> None:
        """Train the discriminator on policy and expert transitions."""
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0

        self.amp_normalizer.train()
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
            grad_pen_loss = self.discriminator.compute_gradient_penalty(
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

        loss_dict["amp_loss"] = mean_amp_loss / num_updates
        loss_dict["amp_grad_pen"] = mean_grad_pen_loss / num_updates
        loss_dict["amp_policy_pred"] = mean_policy_pred / num_updates
        loss_dict["amp_expert_pred"] = mean_expert_pred / num_updates
