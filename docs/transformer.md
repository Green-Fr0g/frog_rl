# Transformer 模块

`frog_rl.modules.Transformer` 是一个独立的序列编码器。它只负责把输入 token 编成特征，不包含 Actor、Critic、动作分布或循环隐藏状态。

这个版本支持单张量输入，也支持多个输入片段拼接后再编码，更适合复杂观测组合。

## 导入

```python
from frog_rl.modules import Transformer
```

## 基本用法

```python
import torch
from frog_rl.modules import Transformer

encoder = Transformer(
    input_dim=48,
    hidden_dim=256,
    num_layers=2,
    num_heads=4,
    output_dim=128,
    max_seq_len=32,
)

x = torch.randn(64, 16, 48)
features = encoder(x)
```

## 多输入片段

如果一条 token 的特征由多个片段组成，可以用 `input_dims` 和 `forward()` 里的输入列表：

```python
encoder = Transformer(input_dims=[16, 32], hidden_dim=256)

part_a = torch.randn(64, 20, 16)
part_b = torch.randn(64, 20, 32)
features = encoder([part_a, part_b])
```

模块会先把这些片段按最后一维拼起来，再送入 Transformer。

## 独立 Q/K/V 输入

如果 Q、K、V 来自不同输入，可以设置它们各自的原始特征维度，并使用关键字参数调用：

```python
cross_attention = Transformer(
    q_dim=32,
    k_dim=48,
    v_dim=64,
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    use_cls_token=True,
    pooling="cls",
)

q = torch.randn(64, 4, 32)   # (batch, query_time, q_dim)
k = torch.randn(64, 16, 48)  # (batch, key_time, k_dim)
v = torch.randn(64, 16, 64)  # (batch, key_time, v_dim)

features = cross_attention(q=q, k=k, v=v)
# features: (64, 256)
```

该模式中，Q、K、V 会分别投影到 `hidden_dim`，然后执行 cross-attention：

```text
Q: (B, Tq, q_dim) -> (B, Tq, hidden_dim)
K: (B, Tk, k_dim) -> (B, Tk, hidden_dim)
V: (B, Tk, v_dim) -> (B, Tk, hidden_dim)
```

如果有 padding，可以传入：

```python
features = cross_attention(
    q=q,
    k=k,
    v=v,
    padding_mask=q_padding_mask,
    key_padding_mask=kv_padding_mask,
)
```

其中 `padding_mask` 对应 Q 序列，`key_padding_mask` 对应 K/V 序列，`True` 表示 padding。

## 构造函数

```python
Transformer(
    input_dim=None,
    *,
    input_dims=None,
    q_dim=None,
    k_dim=None,
    v_dim=None,
    hidden_dim=256,
    num_layers=2,
    num_heads=4,
    ff_hidden_dim=None,
    output_dim=None,
    input_hidden_sizes=None,
    output_hidden_sizes=None,
    activation="gelu",
    dropout=0.0,
    normalize_input=False,
    normalization_until=int(1e8),
    max_seq_len=128,
    use_cls_token=True,
    pooling="cls",
    batch_first=True,
    layer_norm_eps=1e-5,
    norm_first=True,
)
```

### 主要参数

- `input_dim`：单输入时的 token 特征维度。
- `input_dims`：多输入时每个片段的特征维度列表。
- `input_hidden_sizes`：输入投影前的 MLP 隐藏层。
- `output_hidden_sizes`：池化后的输出 MLP 隐藏层。
- `pooling`：支持 `cls`、`mean`、`max` 三种池化。
- `layer_norm_eps` 和 `norm_first`：暴露 Transformer 的核心稳定性参数。

## 输出

- `return_sequence=False` 时，返回每条序列一个向量，形状为 `(batch, output_dim)`。
- `return_sequence=True` 时，返回完整 token 序列，形状为 `(batch, time, output_dim)`；启用 CLS token 时，会多出第 0 个位置。

## Mask 与归一化

`padding_mask` 形状为 `(batch, time)`，`True` 表示 padding。

如果启用了 `normalize_input=True`，训练时可调用：

```python
encoder.update_normalization(x)
```

然后在 `eval()` 模式下冻结统计量。

## 特点

- 支持多输入片段拼接，比只收单一张量更灵活。
- 支持 CLS、均值、最大池化，便于不同任务选择输出方式。
- 支持输入和输出两侧的 MLP，适合做真正的 backbone。
- 保留了显式 `padding_mask`，比把 mask 藏在输入维度里更直观。
