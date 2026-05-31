# models/

`models/` 是当前仓库最宽的模块。真正有训练入口意义的，主要是 `BaseModel`、`MSRModel`、`MODEL_REGISTRY`、`DomainTowerHead` 和一批 backbone。

## 1. 基础模型

### `BaseModel`

源码在 [`src/betterbole/models/base.py`](../../src/betterbole/models/base.py)。

```python
class BaseModel(nn.Module):
    def __init__(self, manager: SchemaManager):
        self.manager = manager
        self.omni_embedding = OmniEmbLayer(manager=manager)
```

它负责自动挂载统一 embedding 层，其他逻辑由子类自己实现。

### `ModelOutput`

`ModelOutput` 在源码里存在，但当前主流 MSR 模型大多直接返回 `Tensor`，没有强依赖这个 dataclass。

## 2. `MSRModel`

源码在 [`src/betterbole/models/msr/base.py`](../../src/betterbole/models/msr/base.py)。

```python
class MSRModel(BaseModel):
    def __init__(self, manager, num_domains, **kwargs):
        ...

    @classmethod
    def from_manager(cls, manager, num_domains, **kwargs):
        ...
```

### 核心约定

绝大多数 MSR 实现都会在 `__init__` 里设置：

- `self.DOMAIN = self.manager.domain_field`
- `self.LABEL = self.manager.label_field`
- `self.input_view = self.omni_embedding.whole`

### `from_manager()`

如果 `__init__` 允许 `**kwargs`，就直接透传。
如果不允许，它会过滤掉不支持的参数，并发出 warning。

## 3. `DomainTowerHead`

源码在 [`src/betterbole/models/msr/components/heads.py`](../../src/betterbole/models/msr/components/heads.py)。

```python
head = DomainTowerHead(
    num_domains=3,
    input_dim=128,
    hidden_dims=(64,),
)
```

它内部其实是 `MultiScenarioContainer(num_domains, lambda: MLP(..., 1))`。

## 4. Backbone 总览

### 共享底座

源码在 [`src/betterbole/models/backbone`](../../src/betterbole/models/backbone)。

当前可解析的 backbone 名称有：

- `ple`
- `small_ple`
- `pleversion1`
- `mmoe`
- `small_mmoe`
- `singlelayermmoe`
- `single_layer_mmoe`
- `sharedbottomless`
- `shared_bottom_less`
- `sharedbottomplus`
- `shared_bottom_plus`
- `star`
- `starple`
- `star_ple`

### `SharedBottomLess` / `SharedBottomPlus`

最基础的共享底座。

### `SingleLayerMMoE` / `SingleLayerMTLMMoE`

一个 domain/task 对应一个 gate，专家共享。

### `PLE` / `PLEVersion1`

多层 shared/specific experts。当前实现是轻量版，不是旧文档里写的那种通用大框架。

### `STAR` / `StarPle`

`STAR` 通过参数融合得到各个 domain 的特化专家；`StarPle` 在这个基础上叠了一层 PLE 风格组合。

## 5. `MODEL_REGISTRY`

源码在 [`src/betterbole/models/msr/__init__.py`](../../src/betterbole/models/msr/__init__.py)。

当前注册名包括：

```text
sharedbottom
mmoe
ple
ple_v1
ple_version1
star
m3oe
m3oe_v1
m3oe_v2
m2m
ppnet
epnet
pepnet
feature_gate
crocodile
crocodile_v1
pareto
hierrec
automtl
riple
hamur
```

### `build_model()`

```python
model = build_model(manager, num_domains=5, model_cls="mmoe")
```

也可以直接传类：

```python
model = build_model(manager, num_domains=5, model_cls=MMoEModel)
```

## 6. 主要模型族

### `sharedbottom`

最简单的多场景基线，shared backbone + per-domain tower。

### `mmoe`

门控混合专家，每个 domain 有独立 gate。

### `ple`

分层 shared/specific experts。

### `star`

参数融合式场景建模。

### `m3oe`

合并门控混合专家，有 `m3oe_v1` / `m3oe_v2` 变体。

### `m2m`

Transformer + meta gate 风格。

### `ppnet` / `epnet` / `pepnet`

PEPNet 家族的三个入口名。实现统一在 `src/betterbole/models/msr/pepnet/` 下。

### `feature_gate`

基于特征门控的多场景模型。

### `crocodile` / `crocodile_v1`

也是多场景模型族的一部分，已注册在 `MODEL_REGISTRY` 中。

### `pareto`

与 Pareto / 多目标优化相关的模型入口。

### `hierrec`

层次化场景建模。

### `automtl`

自动化多任务/多场景架构搜索。

### `riple`

共享/专属专家加辅助约束的多场景模型。

### `hamur`

超网络适配器风格的多场景建模。

## 7. 模型工具

源码在 [`src/betterbole/models/utils`](../../src/betterbole/models/utils)。

### `MultiScenarioContainer`

给每个 domain 一份独立子网络，前向时按 `domain_ids` 选择对应输出。

### `MultiTaskContainer`

给每个 task 一份独立子网络，返回字典输出。

### `MLP`

当前推荐使用的通用 MLP 实现。

### `DNN`

偏 DeepCTR 风格的兼容版本。

### `FeatureBifurcator` / `BifurcatedMLP`

用于把输出拆成“bias + fluctuation”两路。

## 8. 使用层面的真实建议

- 新 MSR 模型优先继承 `MSRModel`。
- `predict()` 和 `calculate_loss()` 要和 `BaseTrainer` 的默认评估逻辑对齐。
- 大多数模型默认输出 raw logits，而不是 sigmoid 后概率。
- 如果模型使用序列特征，通常需要显式从 `omni_embedding.seq_groups[...]` 取，不要期待 `whole` 自动帮你展开序列语义。
