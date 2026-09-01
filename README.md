## Setup

The package can be installed via PyPI with:

```bash
pip install frog-rl
```

or by cloning this repository and installing it with:

```bash
git clone https://github.com/Green-Fr0g/frog_rl
cd frog_rl
pip install -e .
```

The package supports the following logging frameworks which can be configured through `logger`:

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/

## Example Runner Configurations

The examples below show complete Isaac Lab style runner configuration files for the most common `frog_rl` training setups.

### PPO

```python
"""Example PPO runner configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ExamplePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "example_ppo"
    run_name = ""
    logger = "tensorboard"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    empirical_normalization = None
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,
        symmetry_cfg=None,
    )
```

### PPO MoE

Use `ActorCriticMoE` when the actor and critic should use mixture-of-experts networks.

```python
"""Example PPO MoE runner configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from frog_lab.rl_cfg.moe_cfg import RslRlMoEActorCriticCfg


@configclass
class ExampleMoEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "example_ppo_moe"
    run_name = ""
    logger = "tensorboard"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    empirical_normalization = None

    policy = RslRlMoEActorCriticCfg(
        class_name="ActorCriticMoE",
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_num_experts=4,
        critic_num_experts=4,
        actor_gate_hidden_dims=[64],
        critic_gate_hidden_dims=[64],
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,
        symmetry_cfg=None,
    )
```

### AMP PPO

Use `AMPPPO` when training with adversarial motion priors. The environment must provide the observation group named by `expert_state_key`; in this example that group is `amp_state`.

```python
"""Example AMP PPO runner configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from frog_lab.rl_cfg.amp_cfg import AmpCfg, RslRlAmpAlgorithmCfg, RslRlAmpRunnerCfg


@configclass
class ExampleAmpPPORunnerCfg(RslRlAmpRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "example_amp_ppo"
    run_name = ""
    logger = "tensorboard"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    empirical_normalization = None
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlAmpAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,
        symmetry_cfg=None,
        amp_cfg=AmpCfg(
            amp_reward_coef=0.1,
            amp_replay_buffer_size=2000000,
            amp_discr_hidden_dims=[1024, 512],
            amp_discr_activation="relu",
            discriminator_lr=1.0e-3,
            discriminator_optimizer="adam",
            amp_trunk_weight_decay=1.0e-3,
            amp_head_weight_decay=1.0e-2,
            grad_pen_coef=10.0,
            amp_task_reward_lerp=0.75,
            expert_state_key="amp_state",
            motion_loader_class_name="frog_lab.tasks.amp.utils.motion_loader:AMPBodyStateMotionLoader",
            motion_loader_kwargs={
                "motion_files": "/path/to/motion/files",
                "body_names": ("pelvis", "left_hip_pitch_link", "right_hip_pitch_link"),
                "anchor_name": "pelvis",
                "all_body_names": ("pelvis", "left_hip_pitch_link", "right_hip_pitch_link"),
                "quat_order": "wxyz",
            },
        ),
    )
```
