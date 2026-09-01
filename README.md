# Frog-RL

A fast and simple implementation of learning algorithms for robotics. For an overview of the library please have a look at https://arxiv.org/pdf/2509.10771.

Environment repositories using the framework:

* **`Isaac Lab`** (built on top of NVIDIA Isaac Sim): https://github.com/isaac-sim/IsaacLab
* **`Legged Gym`** (built on top of NVIDIA Isaac Gym): https://leggedrobotics.github.io/legged_gym/
* **`MuJoCo Playground`** (built on top of MuJoCo MJX and Warp): https://github.com/google-deepmind/mujoco_playground/

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

## MoE PPO

To use the mixture-of-experts actor-critic with PPO, set the policy class to `ActorCriticMoE`:

```yaml
policy:
  class_name: ActorCriticMoE
  actor_hidden_dims: [256, 256, 256]
  critic_hidden_dims: [256, 256, 256]
  actor_num_experts: 4
  critic_num_experts: 4
  actor_gate_hidden_dims: [64]
  critic_gate_hidden_dims: [64]
  activation: elu
  init_noise_std: 1.0
  noise_std_type: scalar
```
