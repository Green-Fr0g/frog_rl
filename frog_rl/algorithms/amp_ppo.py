"""AMP (Adversarial Motion Priors) extension for PPO."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from frog_rl.algorithms.ppo import PPO
from frog_rl.models.amp_discriminator import AMPDiscriminator
from frog_rl.modules.normalization import EmpiricalNormalization
from frog_rl.storage import AMPStorage, RolloutStorage
from frog_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


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
        self.amp_state_key = amp_cfg["amp_state_key"]
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
        self.amp_normalizer = EmpiricalNormalization(
            shape=self.state_dim, until=int(amp_cfg.get("normalization_until", 1.0e8))
        ).to(device)
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
                f"AMP expert state dimension mismatch: observation '{self.amp_state_key}' has "
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
        self._current_amp_state = obs[self.amp_state_key].detach().clone()
        return super().act(obs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Store the AMP transition and replace the task reward with AMP reward."""
        if self._current_amp_state is None:
            raise RuntimeError("AMPPPO.process_env_step() must be called after act().")

        next_amp_state = obs[self.amp_state_key].clone()
        reset_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            next_amp_state[reset_env_ids] = self._current_amp_state[reset_env_ids]

        self.amp_storage.add(self._current_amp_state, next_amp_state)
        amp_reward, _ = self.discriminator.reward(self._current_amp_state, next_amp_state, rewards, self.amp_normalizer)
        super().process_env_step(obs, amp_reward.reshape(-1), dones, extras)
        self._current_amp_state = next_amp_state.detach().clone()

    def update(self) -> dict[str, float]:
        """Run PPO updates followed by discriminator training."""
        loss_dict = super().update()
        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        # Train the discriminator as soon as the buffer holds any samples. During the cold-start
        # the buffer may hold fewer than one mini-batch, so cap the per-update batch size to the
        # samples currently available instead of skipping discriminator training entirely.
        num_samples = self.amp_storage.num_samples
        if num_samples > 0:
            num_updates = self.num_learning_epochs * self.num_mini_batches
            amp_batch_size = min(mini_batch_size, num_samples)
            mean_amp_loss = 0.0
            mean_grad_pen_loss = 0.0
            mean_policy_pred = 0.0
            mean_expert_pred = 0.0

            self.amp_normalizer.train()
            policy_generator = self.amp_storage.mini_batch_generator(num_updates, amp_batch_size)
            expert_generator = getattr(self.amp_data, "mini_batch_generator", None)
            if expert_generator is None:
                expert_generator = self.amp_data.mini_batch_generator
            expert_generator = expert_generator(num_updates, amp_batch_size)

            for (policy_state, policy_next_state), (expert_state, expert_next_state) in zip(
                policy_generator, expert_generator
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
        """Construct AMP-PPO with all AMP settings resolved in one pass."""
        config = deepcopy(cfg)
        algorithm_cfg = config["algorithm"]
        amp_cfg = deepcopy(algorithm_cfg.pop("amp_cfg"))
        if amp_cfg is None:
            raise ValueError("AMPPPO requires algorithm.amp_cfg.")

        state_key = amp_cfg["amp_state_key"]
        if state_key not in obs:
            raise ValueError(f"AMP observation '{state_key}' is missing. Available: {list(obs.keys())}")
        state = obs[state_key]
        if not isinstance(state, torch.Tensor) or state.ndim != 2:
            raise ValueError(f"AMP observation '{state_key}' must be a 2-D tensor.")
        amp_cfg["state_dim"] = int(state.shape[-1])
        amp_cfg["discriminator_input_dim"] = 2 * amp_cfg["state_dim"]
        amp_cfg.setdefault("time_between_frames", env.unwrapped.step_dt)

        policy_cfg = config.get("policy")
        if policy_cfg is not None:
            policy_cfg = dict(policy_cfg)
            distribution_class = (
                "frog_rl.modules.distribution:HeteroscedasticGaussianDistribution"
                if policy_cfg.get("state_dependent_std", False)
                else "frog_rl.modules.distribution:GaussianDistribution"
            )
            config["actor"] = {
                "class_name": "frog_rl.models.mlp_model:MLPModel",
                "hidden_dims": policy_cfg["actor_hidden_dims"],
                "activation": policy_cfg["activation"],
                "obs_normalization": policy_cfg["actor_obs_normalization"],
                "distribution_cfg": {"class_name": distribution_class, "init_std": policy_cfg["init_noise_std"], "std_type": policy_cfg.get("noise_std_type", "scalar")},
            }
            config["critic"] = {
                "class_name": "frog_rl.models.mlp_model:MLPModel",
                "hidden_dims": policy_cfg["critic_hidden_dims"],
                "activation": policy_cfg["activation"],
                "obs_normalization": policy_cfg["critic_obs_normalization"],
            }

        obs_groups = resolve_obs_groups(obs, deepcopy(config.get("obs_groups", {})), ["actor", "critic", state_key])
        actor_class = resolve_callable(config["actor"].pop("class_name"))
        critic_class = resolve_callable(config["critic"].pop("class_name"))
        actor = actor_class(obs, obs_groups, "actor", env.num_actions, **config["actor"]).to(device)
        critic = critic_class(obs, obs_groups, "critic", 1, **config["critic"]).to(device)
        storage = RolloutStorage("rl", env.num_envs, config["num_steps_per_env"], obs, [env.num_actions], device)
        algorithm_cfg.pop("class_name", None)
        algorithm = AMPPPO(
            actor,
            critic,
            storage,
            device=device,
            amp_cfg=amp_cfg,
            multi_gpu_cfg=config.get("multi_gpu"),
            **algorithm_cfg,
        )
        algorithm.compile(config.get("torch_compile_mode"))
        return algorithm

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
