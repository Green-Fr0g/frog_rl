# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from tensordict import TensorDict

from frog_rl.algorithms.ppo import PPO
from frog_rl.env import VecEnv
from frog_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from frog_rl.models import MoEModel
from frog_rl.storage import RolloutStorage
from frog_rl.utils import resolve_obs_groups


class MoEPPO(PPO):
    """PPO with Mixture-of-Experts (MoE) models for the actor and the critic.

    This variant of :class:`~frog_rl.algorithms.ppo.PPO` replaces the plain MLP head of both models with a
    mixture-of-experts head (see :class:`~frog_rl.models.moe_model.MoEModel`). Since ``MoEModel`` exposes the same
    interface as :class:`~frog_rl.models.mlp_model.MLPModel`, the whole PPO training loop (``act`` /
    ``process_env_step`` / ``update`` / ``save`` / ``load``) is inherited unchanged; only the model class used to build
    the actor and critic differs.
    """
    actor: MoEModel
    """The actor model."""

    critic: MoEModel
    """The critic model."""

    def __init__(
        self,
        actor: MoEModel,
        critic: MoEModel,
        storage: RolloutStorage,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize MoE-PPO with MoE models.

        Args:
            actor: MoE actor model.
            critic: MoE critic model.
            storage: Rollout storage.
            device: Device to run the models on.
            multi_gpu_cfg: Multi-GPU configuration. Defaults to ``None`` (single GPU).
            **kwargs: Additional PPO hyper-parameters forwarded to :class:`~frog_rl.algorithms.ppo.PPO`.
        """
        super().__init__(actor, critic, storage, device=device, multi_gpu_cfg=multi_gpu_cfg, **kwargs)

    def get_policy(self) -> MoEModel:
        """Get the policy model."""
        return self._raw_actor

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> MoEPPO:
        """Construct the MoE-PPO algorithm with :class:`MoEModel` for both the actor and the critic."""
        # The model class is fixed to MoEModel for MoE-PPO, so drop any class_name from the model configs.
        cfg["algorithm"].pop("class_name", None)
        actor_class: type[MoEModel] = MoEModel
        critic_class: type[MoEModel] = MoEModel
        cfg["actor"].pop("class_name", None)
        cfg["critic"].pop("class_name", None)

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor: MoEModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        critic: MoEModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: MoEPPO = MoEPPO(
            actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg