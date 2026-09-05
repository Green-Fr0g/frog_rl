# Frog RL 使用说明

本文介绍当前仓库中 PPO、AMP、WASABI，以及 VAE、Transformer 的主要配置和使用方式。

## 通用配置结构

训练配置通常分为算法、策略模型、观测组和 rollout 参数：

```python
cfg = {
    "num_steps_per_env": 24,
    "obs_groups": {
        "actor": ["policy"],
        "critic": ["critic"],
    },
    "policy": {
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
        "activation": "elu",
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "init_noise_std": 1.0,
    },
    "algorithm": {
        "class_name": "frog_rl.algorithms:PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1e-3,
    },
}
```

`obs` 是环境返回的 `TensorDict`。每个被策略使用的 observation group 应该是二维 Tensor，形状为 `(num_envs, feature_dim)`。

## PPO

PPO 是基础算法。策略和价值网络分别读取 `actor`、`critic` observation groups：

```python
cfg["algorithm"] = {
    "class_name": "frog_rl.algorithms:PPO",
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "clip_param": 0.2,
    "gamma": 0.99,
    "lam": 0.95,
    "value_loss_coef": 1.0,
    "entropy_coef": 0.01,
    "learning_rate": 1e-3,
    "max_grad_norm": 1.0,
    "optimizer": "adam",
}
```

训练循环由 `OnPolicyRunner` 调用：

```python
actions = algorithm.act(obs)
obs, rewards, dones, extras = env.step(actions)
algorithm.process_env_step(obs, rewards, dones, extras)
losses = algorithm.update()
```

## AMP

AMP 使用策略状态和参考动作数据训练 transition discriminator。`amp_state_key` 必须对应环境 observation 中的二维 Tensor。

```python
cfg["algorithm"] = {
    "class_name": "frog_rl.algorithms:AMPPPO",
    "amp_cfg": {
        "amp_state_key": "amp_state",
        "amp_motion_files": ["path/to/motion_file.npz"],
        "motion_loader_class_name": "your_package:MotionLoader",
        "motion_loader_kwargs": {},
        "amp_reward_coef": 1.0,
        "amp_discr_hidden_dims": [1024, 512],
        "amp_discr_activation": "relu",
        "amp_replay_buffer_size": 1000000,
        "discriminator_optimizer": "adam",
        "discriminator_lr": 1e-3,
        "amp_trunk_weight_decay": 1e-3,
        "amp_head_weight_decay": 1e-2,
        "grad_pen_coef": 10.0,
        "normalization_until": 100000000,
    },
}
```

构建时会自动推导：

```text
state_dim = obs[amp_state_key].shape[-1]
discriminator_input_dim = 2 * state_dim
```

AMP 的 transition 是 `(current_state, next_state)`。环境返回 `done=True` 时，当前实现将 `next_state` 设置为对应的 `current_state`，避免把 reset 后新 episode 的初始状态拼接到上一 episode 的末状态上。

## WASABI

WASABI 使用两个 observation group：策略状态和参考状态。两者必须是二维 Tensor，且最后一维相同。

```python
cfg["algorithm"] = {
    "class_name": "frog_rl.algorithms:WasabiPPO",
    "wasabi_cfg": {
        "wasabi_policy_state_key": "wasabi_policy",
        "wasabi_reference_state_key": "wasabi_reference",
        "wasabi_task_reward_weight": 1.0,
        "wasabi_reward_type": "log",
        "wasabi_reward_coef": 1.0,
        "wasabi_loss_type": "BCEWithLogitsLoss",
        "wasabi_loss_coef": 1.0,
        "wasabi_grad_pen_coef": 10.0,
        "wasabi_grad_tolerance": 0.0,
        "wasabi_trunk_weight_decay": 0.0,
        "wasabi_head_weight_decay": 0.0,
        "wasabi_discr_hidden_dims": [512, 256],
        "wasabi_discr_activation": "elu",
        "wasabi_normalize_input": True,
        "wasabi_normalization_until": 100000000,
        "wasabi_discriminator_optimizer": "adamw",
        "wasabi_discriminator_lr": 1e-3,
    },
}
```

支持的 reward 类型：

- `log`: `softplus(logit)`
- `quad`: `max(0, 1 - 0.25 * (logit - 1)^2)`
- `wasserstein`: 直接使用 logit

支持的 discriminator loss 类型：`BCEWithLogitsLoss`、`MSELoss` 和 `WassersteinLoss`。

## MoE-PPO

`MoEPPO` 使用 `MoEModel` 替代普通 `MLPModel` 作为 actor 和 critic。PPO 的 rollout、更新、保存和加载流程不变，只有策略网络头部变成多个 expert 与一个 gate 的软混合。

```python
cfg = {
    "num_steps_per_env": 24,
    "obs_groups": {
        "actor": ["policy"],
        "critic": ["critic"],
    },
    "actor": {
        "hidden_dims": [256, 256],
        "activation": "elu",
        "obs_normalization": False,
        "num_experts": 8,
        "gate_hidden_dims": [64],
        "distribution_cfg": {
            "class_name": "frog_rl.modules.distribution:GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    },
    "critic": {
        "hidden_dims": [256, 256],
        "activation": "elu",
        "obs_normalization": False,
        "num_experts": 8,
        "gate_hidden_dims": [64],
    },
    "algorithm": {
        "class_name": "frog_rl.algorithms:MoEPPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1e-3,
    },
}
```

MoE 相关参数：

- `num_experts`：expert 网络数量，必须是正整数；
- `gate_hidden_dims`：gate 网络的隐藏层；设为空列表时使用线性 gate；
- `hidden_dims`：每个 expert 的隐藏层；
- `distribution_cfg`：actor 的动作分布配置，critic 通常不需要配置。

通过配置构建后，算法会固定使用 `MoEModel` 创建 actor 和 critic：

```python
algorithm = MoEPPO.construct_algorithm(obs, env, cfg, device="cuda:0")
```

`MoEModel` 计算所有 expert，并使用 gate 的 softmax 权重进行加权，因此这是 dense/soft MoE，不是只激活 top-k expert 的 sparse MoE。模型也支持与 `MLPModel` 相同的 `as_jit()` 和 `as_onnx(verbose)` 导出接口。

## VAE

VAE 的 encoder 输出 `mean` 和 `var`，训练时使用重参数化采样：

```python
vae = VAE(
    input_dim=128,
    latent_dim=32,
    encoder_hidden_dims=(256, 256),
    decoder_hidden_dims=(256, 256),
    activation="elu",
)

reconstruction, mean, var = vae(observations, sample=True)
losses = VAE.loss_function(reconstruction, observations, mean, var, beta=1.0)
losses["loss"].backward()
```

推理时可以关闭随机采样：

```python
reconstruction, mean, var = vae(observations, sample=False)
```

导出接口：

```python
torch.jit.script(vae.as_jit())
torch.onnx.export(
    vae.as_onnx(),
    vae.as_onnx().get_dummy_inputs(),
    "vae.onnx",
    input_names=vae.as_onnx().input_names,
    output_names=vae.as_onnx().output_names,
)
```

导出模型固定使用 posterior mean，不执行随机 latent sampling。

## Transformer

Transformer 输入单个序列时形状为 `(batch, time, feature)`：

```python
transformer = Transformer(
    input_dim=64,
    hidden_dim=256,
    num_layers=2,
    num_heads=4,
    max_seq_len=128,
    pooling="cls",
    use_cls_token=True,
)

embedding = transformer(observations)
```

支持 `cls`、`mean`、`max` 三种 pooling，也支持多个输入片段拼接：

```python
transformer = Transformer(input_dims=(32, 16), hidden_dim=128, num_heads=4)
embedding = transformer((state_part, command_part))
```

Cross-attention 使用独立的 query、key、value：

```python
transformer = Transformer(
    q_dim=32,
    k_dim=64,
    v_dim=64,
    hidden_dim=128,
    num_heads=4,
)

embedding = transformer(q=query, k=key, v=value)
```

导出接口分为 self-attention 和 cross-attention：

```python
self_jit = transformer.as_jit()
self_onnx = transformer.as_onnx()
cross_jit = transformer.as_cross_attention_jit()
cross_onnx = transformer.as_cross_attention_onnx()
```

cross-attention 导出模型的输入顺序是 `query, key, value`。导出 wrapper 提供 `get_dummy_inputs()`、`input_names` 和 `output_names`，可直接用于 ONNX 导出。
