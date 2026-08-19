# 方案:PPO 在 MLP 与 MoE 基础模型之间切换

## Context(背景与目标)

`frog_rl`(`source/frog_rl/`)是 rsl_rl v3 的 fork,已新增 `MoEModel`(`models/moe_model.py`,已完成 JIT/ONNX 适配)。目标是让 PPO 训练时能在 **MLP 与 MoE 基础模型之间通过配置切换**,并让 **actor 与 critic 都用 MoE**。

核心结论:**切换机制在 frog_rl 侧已经完整存在**,缺的只是配置层里能表达 MoE 的模型配置类。本次改动就是把这块配置补上(按用户要求放在 `frog_rl` 包内,不碰外部 IsaacLab 仓库)。

### 已有机制(无需改动,作为方案基础)

`source/frog_rl/algorithms/ppo.py` 的 `PPO.construct_algorithm`:

```python
# 417-418 行:读取并 pop 每个模型的 class_name,用 resolve_callable 解析
actor_class  = resolve_callable(cfg["actor"].pop("class_name"))
critic_class = resolve_callable(cfg["critic"].pop("class_name"))
# 433 / 437 行:把 actor/critic 字典里剩下的所有键作为 **kwargs 传给模型构造器
actor  = actor_class(obs, cfg["obs_groups"], "actor",  env.num_actions, **cfg["actor"])
critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"])
```

- `resolve_callable`(`utils/utils.py:97`)对裸名如 `"MoEModel"` 会扫描 `frog_rl` 顶层子模块并命中 `frog_rl.models.MoEModel`(已在 `models/__init__.py` 导出)。
- `MoEModel.__init__` 签名在 `MLPModel` 基础上多了 `num_experts: int = 8` 和 `gate_hidden_dims: tuple|list = ()`。
- 因此:切换 = 改 `class_name` + 增加 `num_experts` / `gate_hidden_dims` 两个键。**关键在于 `class_name="MLPModel"` 时绝不能把 `num_experts` 传进去**(`MLPModel.__init__` 不接收该参数,会 TypeError)——这也是用"两个配置类"而非"一个类塞满所有字段"的原因。

## 改动内容

### 1. 新建 `source/frog_rl/config.py`(核心交付)

用**普通 `@dataclass`**(不依赖 isaaclab,与 `mjlab/src/mjlab/rl/config.py` 的 `RslRlModelCfg` 同风格)。isaaclab 的 `class_to_dict`(`IsaacLab/source/isaaclab/isaaclab/utils/dict.py:24-72`)通过 `obj.__dict__` 递归序列化,普通 dataclass 会被正确转成嵌套 dict,因此能被 agent 配置的 `.to_dict()` 正常处理。

定义三个类(字段与模型构造器签名严格对齐):

```python
@dataclass
class GaussianDistributionCfg:
    class_name: str = "GaussianDistribution"
    init_std: float = 1.0
    std_type: str = "scalar"          # "scalar" | "log"
    std_range: tuple[float, float] = (1e-6, 1e6)
    learn_std: bool = True

@dataclass
class RslRlMLPModelCfg:
    class_name: str = "MLPModel"
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: GaussianDistributionCfg | None = None  # None = 确定性输出(critic 用)

@dataclass
class RslRlMoEModelCfg(RslRlMLPModelCfg):
    class_name: str = "MoEModel"
    num_experts: int = 8
    gate_hidden_dims: list[int] = field(default_factory=list)  # 空 = 线性门控
```

序列化后得到的 `cfg["actor"]` 字典即为:

```python
{
    "class_name": "MoEModel",
    "hidden_dims": [...],
    "activation": "elu",
    "obs_normalization": False,
    "distribution_cfg": {"class_name": "GaussianDistribution", ...},
    "num_experts": 8,
    "gate_hidden_dims": [],
}
```

——与 `MoEModel.__init__` 参数一一对应。

### 2. `source/frog_rl/__init__.py` 导出

在现有 docstring 之后加 `from .config import GaussianDistributionCfg, RslRlMLPModelCfg, RslRlMoEModelCfg`,方便 `from frog_rl import RslRlMoEModelCfg`。

### 3. 修复并扩展 AMP agent 配置(作为 MLP/MoE 示例)

文件:`source/frog_lab/frog_lab/tasks/amp/config/g1/agents/rsl_rl_ppo_cfg.py`

- 现状:`from isaaclab_rl.rsl_rl import RslRlMLPModelCfg`(第 6 行)——该符号在**检出版 isaaclab_rl(IsaacLab v2.3.2)中不存在**,这个文件当前是 broken 的。改为 `from frog_rl.config import RslRlMLPModelCfg`(`RslRlBaseRunnerCfg`、`RslRlPpoAlgorithmCfg` 仍从 isaaclab_rl 导入,它们存在)。
- `_make_actor_cfg` 里 `RslRlMLPModelCfg.GaussianDistributionCfg(...)` 改为顶层 `GaussianDistributionCfg(...)`(同步 import)。
- 新增一个 **MoE 变体** runner 配置类(actor + critic 都用 MoE),示范切换:

```python
def _make_actor_moe_cfg() -> RslRlMoEModelCfg:
    return RslRlMoEModelCfg(
        hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False,
        distribution_cfg=GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        num_experts=8, gate_hidden_dims=[64],
    )
def _make_critic_moe_cfg() -> RslRlMoEModelCfg:
    return RslRlMoEModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
# G1_29DOFAmpMoERunnerCfg(G1_29DOFAmpRunnerCfg): actor/critic 换成 _make_*_moe_cfg()
```

> 说明:AMP 配置是 frog_rl 新 schema 的唯一现成示例,借此同时完成"修复 broken import"和"给出 MLP/MoE 对照"。locomotion/mimic 的 agent 配置仍走旧 `policy` schema + 旧 `rsl_rl` runner,不在本次范围;如需,后续按同模式迁移即可。

## 验证(无需 IsaacSim)

在 `z_lab` 环境、`PYTHONPATH=source` 下跑一段冒烟脚本:

1. **序列化**:实例化 `RslRlMoEModelCfg`(含 `distribution_cfg=GaussianDistributionCfg(...)`),用 `isaaclab.utils.dict.class_to_dict` 或包一层 `@configclass` 后 `.to_dict()` 序列化,断言:
   - `class_name == "MoEModel"`、含 `num_experts`/`gate_hidden_dims`;
   - 嵌套 `distribution_cfg` 展开为含 `class_name == "GaussianDistribution"` 的 dict;
   - `RslRlMLPModelCfg` 序列化后**不含** `num_experts`(证明不会污染 MLP)。
2. **类解析**:`resolve_callable("MoEModel") is frog_rl.models.MoEModel`,`resolve_callable("MLPModel") is MLPModel`。
3. **构造**:用假 obs `TensorDict`(如 `{"policy": torch.zeros(1, 47)}`,`obs_groups={"actor": ["policy"], "critic": ["policy"]}`)分别 `MoEModel(obs, ..., "actor", 12, **moe_cfg_dict)` 与 `MLPModel(obs, ..., "critic", 1, **mlp_cfg_dict)`,断言 MoEModel 存在 `experts`/`gate` 且无 `mlp` 属性,两者 forward 输出形状正确。
4. **端到端回归**:`RslRlMLPModelCfg` 走一遍 `construct_algorithm` 等价的构造路径不报 TypeError,`RslRlMoEModelCfg` 同样。

## 关键文件清单

- 新增:`source/frog_rl/config.py`
- 修改:`source/frog_rl/__init__.py`
- 修改(示例 + 修复):`source/frog_lab/frog_lab/tasks/amp/config/g1/agents/rsl_rl_ppo_cfg.py`
- 只读参考:`source/frog_rl/algorithms/ppo.py`(已支持,无需改)、`source/frog_rl/models/moe_model.py`、`mjlab/src/mjlab/rl/config.py`(同风格参考)
