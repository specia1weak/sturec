# 模型模块 (`models/`)

> **层级**: L4 (模型层)
>
> 依赖 L2 (`OmniEmbLayer`、`SchemaManager`) + L3 (`Evaluator`)。为 L5 (训练层) 提供 `BaseModel`。
>
> 所有模型共享 `OmniEmbLayer` 注入，仅通过交换 Backbone 实现架构升级。

> 统一的多场景推荐模型体系。所有模型共享相同的特征注入层 (`OmniEmbLayer`)，仅通过交换 Backbone 与 Head 实现架构升级。

## 快速导航

- [模型体系总览](#模型体系总览)
- [BaseModel（基类）](#basemodel基类)
- [Backbone（骨干网络）](#backbone骨干网络)
  - [SharedBottom（共享底部）](#sharedbottom共享底部)
  - [MMoE（多门控混合专家）](#mmoe多门控混合专家)
  - [PLE（渐进式分层提取）](#ple渐进式分层提取)
  - [STAR / StarPle](#star--starple)
  - [M3oE（合并门控混合专家）](#m3oe合并门控混合专家)
  - [M2M（元到元）](#m2m元到元)
  - [PEPNet（PPNet/EPNet 统一包）](#pepnetppnetepnet-统一包)
- [多场景模型 (MSR)](#多场景模型-msr)
  - [MSRModel（基类）](#msrmodel基类)
  - [模型列表](#模型列表)
  - [MODEL_REGISTRY 与 build_model](#model_registry-与-build_model)
- [AutoMTL（自动化多任务学习）](#automtl自动化多任务学习)
- [HAMUR（超网络适配多场景）](#hamur超网络适配多场景)
- [HierRec（层次化场景建模）](#hierrec层次化场景建模)
- [模型容器与工具](#模型容器与工具)
  - [MultiScenarioContainer](#multiscenariocontainer)
  - [DomainTowerHead](#domaintowerhead)
  - [MLP 与 DNN](#mlp-与-dnn)

---

## 模型体系总览

```
Interaction (Tensor dict)
        │
        ▼
  OmniEmbLayer.whole       ← 统一的 Embedding 注入
        │
        ▼
   Backbone (SharedBottom / MMoE / PLE / STAR / M3oE / M2M / PPNet / EPNet)
        │
        ▼
   DomainTowerHead         ← 每个 domain 私有的 Tower
        │
        ▼
    Logits (B,)            ← 输出点击率预测
```

- 所有模型继承自 [`BaseModel`](src/betterbole/models/base.py:12)
- 所有多场景模型 (MSR) 继承自 [`MSRModel`](src/betterbole/models/msr/base.py:9)
- 所有 Backbone 继承自 [`MSRBackbone`](src/betterbole/models/msr/backbone/base.py)
- 标准接口: `predict(interaction) → Tensor` + `calculate_loss(interaction) → Tensor`

---

## BaseModel（基类）

[`BaseModel`](src/betterbole/models/base.py:12) 是所有模型的基类，继承自 `nn.Module`。

```python
class BaseModel(nn.Module):
    def __init__(self, manager: SchemaManager):
```

- 接收一个 [`SchemaManager`](src/betterbole/emb/manager.py:27) 实例
- 通过 `manager.omni_embedding` 自动挂载 [`OmniEmbLayer`](src/betterbole/emb/emblayer.py:397)
- 提供 `omni_embedding` 属性访问嵌入层
- 子类需实现：`predict()` 和 `calculate_loss()`

---

## Backbone（骨干网络）

所有 Backbone 位于 [`models/backbone/`](src/betterbole/models/backbone/__init__.py)，接收 `(x, domain_ids)` 返回特征张量。

### SharedBottom（共享底部）

**文件**: [`models/backbone/shabtm.py`](src/betterbole/models/backbone/shabtm.py)

最简单的多场景基线——所有场景共享同一 MLP，忽略 `domain_ids`。

```
输入 → MLP(shared) → 输出
```

**变体**:
- `SharedBottomLess` — 浅层共享 MLP
- `SharedBottomPlus` — 更深层的共享 MLP

### MMoE（多门控混合专家）

**文件**: [`models/backbone/mmoe.py`](src/betterbole/models/backbone/mmoe.py)

MMoE 为每个 domain 分配一个独立的门控网络，加权聚合多个专家网络的输出。

```
              ┌── Expert_1 ──┐
输入 ── Gate_0 ── Expert_2 ──┼── 加权求和 → 输出 (domain 特化)
              └── Expert_3 ──┘
```

**变体**:
- [`SingleLayerMMoE`](src/betterbole/models/backbone/mmoe.py:8) — domain-based 门控（每 domain 一个 gate）
- [`SingleLayerMTLMMoE`](src/betterbole/models/backbone/mmoe.py:36) — task-based 门控（每 task 一个 gate，忽略 domain_ids）

**MSR Backbone**: [`MMoEBackbone`](src/betterbole/models/msr/backbone/mmoe.py:10) 位于 `models/msr/backbone/`

- `num_experts`: 专家数量，默认 `num_domains + 1`
- 每个 expert 是一个 MLP
- 每个 gate 是 `Linear → Softmax`

### PLE（渐进式分层提取）

**文件**: [`models/backbone/ple.py`](src/betterbole/models/backbone/ple.py)

PLE 通过多层专家网络实现 "shared-specific" 分离。

```
第一层: [Shared Expert] + [Specific Expert_0 ... Expert_N]
                │                    │
                ▼                    ▼
第二层: [Shared Expert] + [Specific Expert_0 ... Expert_N]
                │                    │
                ▼                    ▼
             Tower → 输出
```

**变体**:
| 版本 | 特点 |
|------|------|
| [`PLE`](src/betterbole/models/backbone/ple.py:9) | 2 层专家，每层含 shared + specific |
| `PLEVersion1` | 1 层专家 |
| `PLEVersion2` | 2 层但不带 shared expert |
| `PLEVersion3` | 3 层专家 |
| `PLEVersion4` | 4 层专家 |

**MSR Backbone**: [`PLEBackbone`](src/betterbole/models/msr/backbone/ple.py) 位于 `models/msr/backbone/`

- `num_levels`: PLE 层次数
- `num_specific_experts`: 每场景专用专家数
- `num_shared_experts`: 共享专家数

### STAR / StarPle

**文件**: [`models/backbone/star.py`](src/betterbole/models/backbone/star.py)

STAR (Situation-Aware Representation) 为每个 domain 维护独立的 MLP 参数，通过加权融合实现场景特化。

```
StarExpert: 每个 domain 独立 MLP → merge_with() 聚合
STAR: 输入 → StarExpert(domain_0) + StarExpert(domain_1) → 加权融合 → 输出
StarPle: 结合 STAR 的门控融合 + SharedBottom 结构
```

**核心**:
- [`StarExpert`](src/betterbole/models/backbone/star.py:10): 单个 MLP，支持 `merge_with()` 将多个 expert 的参数加权融合
- [`STAR`](src/betterbole/models/backbone/star.py:56): 标准 STAR 架构
- [`StarPle`](src/betterbole/models/backbone/star.py:81): STAR + PLE 混合

**MSR Backbone**: [`STARBackbone`](src/betterbole/models/msr/backbone/star.py) 位于 `models/msr/backbone/`

### M3oE（合并门控混合专家）

合并门控混合专家 (Merged MoE)，结合 STAR 的加权门控机制。

**MSR Backbone**: [`M3oEBackbone`](src/betterbole/models/msr/backbone/m3oe.py)

| 版本 | 特点 |
|------|------|
| `M3oEBackbone` | 标准版本 |
| `M3oEVersion1Backbone` | 变体 1 |
| `M3oEVersion2Backbone` | 变体 2 |

参数:
- `star_dims`: STAR 网络的维度
- `expert_dims`: 专家网络维度
- `num_shared_experts`: 共享专家数
- `factor_mode`: 因子化模式
- `shared_gate_detach`: 门控梯度分离

### M2M（元到元）

**文件**: [`models/msr/backbone/m2m.py`](src/betterbole/models/msr/backbone/m2m.py)

M2M (Meta-to-Meta) 使用 Transformer 编码器和领域嵌入，通过元网络生成门控权重。

```
输入 → Linear Project → Transformer Encoder → 多专家网络
                                          ↓
领域嵌入 → Domain Proj → Meta Gate (拼接池化特征+领域状态) → 加权融合
                                                          ↓
                                             Meta Tower → 残差输出
```

**核心机制**:
- `domain_embedding`: 可学习的领域嵌入
- `transformer`: `nn.TransformerEncoder` 对输入进行序列编码
- `meta_gate`: 基于 `[pooled_hint, domain_state]` 生成专家权重
- `meta_tower`: 元网络残差模块

### PEPNet（PPNet/EPNet 统一包）

**文件**: [`models/msr/pepnet/model.py`](src/betterbole/models/msr/pepnet/model.py)

PPNet 和 EPNet 现在统一收敛到 `pepnet` 包里，对外暴露三个模型名：

- `ppnet` → `PPNetModel`
- `epnet` → `EPNetModel`
- `pepnet` → `PEPNetModel`

包内实现拆分为：

- [`pepnet/blocks.py`](src/betterbole/models/msr/pepnet/blocks.py)
- [`pepnet/backbones.py`](src/betterbole/models/msr/pepnet/backbones.py)
- [`pepnet/model.py`](src/betterbole/models/msr/pepnet/model.py)

`PPNetModel` 通过场景嵌入生成个性化参数，对共享网络进行条件变换。

- `scenario_dim`: 场景嵌入维度
- `hidden_dims`: 共享网络隐层维度

`EPNetModel` 是 PPNet 的高效变体，在参数效率和模型容量间取得平衡。

---

## 多场景模型 (MSR)

### MSRModel（基类）

**文件**: [`models/msr/base.py`](src/betterbole/models/msr/base.py)

所有多场景模型的基类，继承自 [`BaseModel`](src/betterbole/models/base.py:12)。

```python
class MSRModel(BaseModel):
    def __init__(self, manager: SchemaManager, num_domains: int, **kwargs):
```

**核心特性**:
- `num_domains`: 场景数量
- `from_manager()`: **类方法工厂**，自动适配构造参数——只传入 `__init__` 接受的 kwargs，忽略不支持的参数并发出警告
- 所有子类约定三件套：
  - `self.DOMAIN = self.manager.domain_field` — 场景标识字段名
  - `self.LABEL = self.manager.label_field` — 标签字段名
  - `self.input_view = self.omni_embedding.whole` — 输入视图

### 模型列表

所有 MSR 模型注册在 [`MODEL_REGISTRY`](src/betterbole/models/msr/__init__.py:18) 中：

| 注册名 | 模型类 | 源文件 | Backbone | 核心思想 |
|--------|--------|--------|----------|----------|
| `sharedbottom` | [`SharedBottomModel`](src/betterbole/models/msr/sharedbottom.py:15) | SharedBottomBackbone | 所有场景共享同一 MLP |
| `mmoe` | [`MMoEModel`](src/betterbole/models/msr/mmoe.py:12) | MMoEBackbone | 每场景独立门控 + 共享专家 |
| `ple` | [`PLEModel`](src/betterbole/models/msr/ple.py:12) | PLEBackbone | 多层 shared/specific 专家 |
| `star` | [`STARModel`](src/betterbole/models/msr/star.py:12) | STARBackbone | 每场景独立 MLP + 加权融合 |
| `m3oe` | [`M3oEModel`](src/betterbole/models/msr/m3oe.py:66) | M3oEBackbone | 合并门控混合专家 |
| `m3oe_v1` | `M3oEVersion1Model` | M3oEVersion1Backbone | M3oE 变体 1 |
| `m3oe_v2` | `M3oEVersion2Model` | M3oEVersion2Backbone | M3oE 变体 2 |
| `m2m` | [`M2MModel`](src/betterbole/models/msr/m2m.py:12) | M2MBackbone | Transformer + Meta Gate |
| `ppnet` | [`PPNetModel`](src/betterbole/models/msr/pepnet/model.py:1) | `pepnet/backbones.py::PPNetBackbone` | 个性化参数网络 |
| `epnet` | [`EPNetModel`](src/betterbole/models/msr/pepnet/model.py:1) | `pepnet/backbones.py::EPNetBackbone` | 高效参数网络 |
| `pepnet` | [`PEPNetModel`](src/betterbole/models/msr/pepnet/model.py:1) | `pepnet/backbones.py::EPNetBackbone` | 统一入口 |
| `hierrec` | [`HierRec`](src/betterbole/models/msr/hierrec.py:110) | (自定义) | 层次化场景建模 |
| `automtl` | [`AutoMTLModel`](src/betterbole/models/msr/automtl/model.py:14) | AutoMTLSuperNet | 神经架构搜索 + 多场景 |
| `riple` | [`RIPLEModel`](src/betterbole/models/msr/riple/model.py:152) | SharedBottom + RIPLEEncoder | shared/specific expert + 解耦约束 |
| `hamur` | [`HAMURModel`](src/betterbole/models/msr/hamur/model.py:31) | (自定义) | 超网络适配器 + 多场景 |

### 通用模型结构

几乎所有标准 MSR 模型遵循相同模板（以 `SharedBottomModel` 为例）：

```python
class SharedBottomModel(MSRModel):
    def __init__(self, manager, num_domains, hidden_dims=(256, 128), ...):
        super().__init__(manager, num_domains)
        self.input_view = self.omni_embedding.whole  # 统一特征注入
        self.backbone = SharedBottomBackbone(...)
        self.head = DomainTowerHead(...)  # 每 domain 独立 Tower

    def encode_features(self, interaction):
        x = self.input_view(interaction)  # [B, total_emb_dim]
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(self, x, domain_ids):
        return self.head(self.backbone(x, domain_ids), domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return F.binary_cross_entropy_with_logits(logits, labels)
```

### MODEL_REGISTRY 与 build_model

**文件**: [`models/msr/__init__.py`](src/betterbole/models/msr/__init__.py)

```python
MODEL_REGISTRY: Dict[str, Type[MSRModel]] = { ... }

def build_model(schema_manager, num_domains, model_cls, **model_kwargs):
    if isinstance(model_cls, str):
        model_cls = MODEL_REGISTRY[model_cls.lower()]
    return model_cls.from_manager(schema_manager, num_domains, **model_kwargs)
```

**使用方式**:
```python
# 按字符串名构建
model = build_model(manager, num_domains=3, model_cls="mmoe")

# 直接传入类
model = build_model(manager, num_domains=3, model_cls=MMoEModel)
```

---

## RIPLEModel（正则化个性化学习）

**文件**: [`models/msr/riple/model.py`](src/betterbole/models/msr/riple/model.py)

RIPLE 现在拆成独立包，并补成了更接近 PLE 的 shared/specific expert 结构。它保留一个 SharedBottom 共享主干，同时增加：

- 每个 domain 的 specific experts
- 所有 domain 共享的 shared experts
- domain-wise task gate
- 一个 shared gate 聚合所有 specific/shared expert

整体上可以理解为“SharedBottom 主干 + RIPLE 专家编码器 + 表示解耦约束”：

```
输入 x
  ├─ SharedBottom Backbone ───────────────→ shared_state
  └─ RIPLEEncoder
      ├─ specific experts + task gate ───→ specific_state
      └─ shared experts + shared gate ───→ shared_context

fused = shared_state + specific_state + shared_context
                               │
                             Tower → 输出
```

**损失** = `BCE(y_pred, y_true) + aux_loss_weight * clamp(|cos_sim(shared_state, specific_state)| - margin, 0)` 。

这里的辅助项仍然是表示解耦约束，用余弦相似度惩罚共享表示和场景专属表示过度重合。

关键参数：

- `hidden_dims`: SharedBottom 主干维度
- `expert_dims`: RIPLE expert 维度
- `num_levels`: RIPLE 层数
- `num_specific_experts`: 每个 domain 的专属专家数
- `num_shared_experts`: 共享专家数
- `margin`: 余弦间隔
- `aux_loss_weight`: 辅助损失权重

---

## AutoMTL（自动化多任务学习）

**文件**: [`models/msr/automtl/model.py`](src/betterbole/models/msr/automtl/model.py)

AutoMTL 将 **神经架构搜索 (NAS)** 引入多场景推荐，在训练过程中自动搜索最优的专家网络结构。

### 搜索-训练流程

```
Warmup (warmup_epochs) → Search (search_epochs) → Finetune
    │                          │                       │
    ▼                          ▼                       ▼
所有参数训练          架构参数 (alpha/beta) +    冻结架构参数
                      网络权重同时训练            微调网络权重
```

### 架构搜索

- 使用 [`AutoMTLSuperNet`](src/betterbole/models/msr/automtl/supernet.py) 超级网络
- `expert_candidate_ops`: 候选操作列表（如 `MLP-16`, `MLP-64`, `MLP-256` 等）
- `discretize_ops`: 每次 discretize 的操作数
- 搜索结束后调用 `export_architecture()` 导出最优结构

### 自定义训练步骤

```python
# AutoMTLModel 实现 custom_train_step 接口
def custom_train_step(self, batch_interaction, ctx: TrainContext):
    # 1. 计算任务损失
    loss = self.calculate_loss(batch_interaction)
    loss.backward()
    # 2. 搜索阶段同时更新架构参数
    if self.stage == "search" and not self.exported:
        arch_optimizer.step()
    self._clear_arch_grads()
    ctx.optimizer.step()
```

### Trainer Hooks

```python
# 自动注册到 Trainer 的 hook 系统
def on_train_epoch_start(self, ctx):
    self.stage = self._resolve_stage(ctx.epoch)   # warmup/search/finetune
    if self.stage != "warmup":
        self.backbone.set_chosen_op_active()

def on_train_epoch_end(self, ctx):
    if self.stage == "search" and not self.exported:
        self.backbone.discretize_one_op()  # 逐步 discretize
```

### 限制

- 不支持序列特征 (sequence field)
- 所有 sparse 特征必须使用相同的 `embedding_dim`

---

## HAMUR（超网络适配多场景）

**文件**: [`models/msr/hamur/model.py`](src/betterbole/models/msr/hamur/model.py)

HAMUR (Hypernetwork Adapter for Multi-scene Rec) 使用超网络生成适配器参数，实现轻量级场景特化。

### 架构模式

| 模式 | backbone 参数 | 说明 |
|------|--------------|------|
| **MLP** | `backbone="mlp"` | 标准 MLP + 超网络适配器 |
| **DCN** | `backbone="dcn"` | Deep & Cross Network + 适配器 |
| **WideDeep** | `backbone="widedeep"` | Wide & Deep + 适配器 |

### 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| [`HAMURAdapterCell`](src/betterbole/models/msr/hamur/common.py:161) | 超网络适配器 Cell | 双线性降维 → Sigmoid → 升维 → 残差 |
| [`build_hyper_network`](src/betterbole/models/msr/hamur/common.py:202) | 超网络生成器 | 输入场景嵌入，输出适配器权重 |
| [`build_mlp_net`](src/betterbole/models/msr/hamur/adapter.py) | MLP 网络构建 | 含适配器的 MLP |
| [`build_dcn_net`](src/betterbole/models/msr/hamur/adapter_dcn.py) | DCN 网络构建 | 含适配器的 Cross Network |
| [`build_widedeep_net`](src/betterbole/models/msr/hamur/adapter_wd.py) | WideDeep 构建 | Wide + Deep + 适配器 |

### 特征视图

```python
# 自动构建特征视图 —— 灵活选择字段
input_view = build_feature_view(
    omni_embedding,
    include_fields=["user_id", "item_id", ...],  # None 表示使用默认目标源
    exclude_fields=("domain_id",),
)
```

- WideDeep 模式下可自动推断 `wide_fields` (dense 特征) 和 `deep_fields` (sparse 特征)

### HAMURBaseModel

**文件**: [`models/msr/hamur/common.py:215`](src/betterbole/models/msr/hamur/common.py:215)

共享基类，提供 `_flatten_embedding()`、`_clamp_probability()` 工具方法。

---

## HierRec（层次化场景建模）

**文件**: [`models/msr/hierrec.py`](src/betterbole/models/msr/hierrec.py)

HierRec 通过 **显式生成器 (Explicit Generator)** + **隐式生成器 (Implicit Generator)** 联合学习场景特有的参数变换。

### 架构

```
输入字段 Embedding (B, F, D)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
DNN1    Domain Embedding
   │         │
   │    Explicit Generator ──→ W1, b1, W2, b2, α
   │         │
   └── MatMul(W1) + b1 ──→ MatMul(W2) + b2 ──→┐
        │                                      │
   Implicit Generator ←─────────────────────────┘
        │
   MatMul + Reshape → OutTrans → DNN3 → Logits
```

### 约束

- `hidden_dims` 必须恰好 4 个元素: `(dnn1_dim, shared_bottleneck_dim, shared_output_dim, head_hidden_dim)`
- 所有非 domain 特征的 `embedding_dim` 必须相等
- `domain_id` 的 embedding_dim 也必须等于该值

### 论文引用

```bibtex
@inproceedings{gao2024hierrec,
  title={HierRec: Scenario-Aware Hierarchical Modeling for Multi-scenario Recommendations},
  author={Gao, Jingtong and Chen, Bo and Zhu, Menghui and Zhao, Xiangyu and Li, Xiaopeng and Wang, Yuhao and Wang, Yichao and Guo, Huifeng and Tang, Ruiming},
  booktitle={CIKM},
  pages={653--662},
  year={2024}
}
```

---

## 模型容器与工具

### MultiScenarioContainer

**文件**: [`models/utils/container.py:50`](src/betterbole/models/utils/container.py:50)

多场景网络容器 —— 为每个 domain 复制一份独立网络，按 `domain_ids` 选取对应输出。

```python
tower = MultiScenarioContainer(
    num_domains=3,
    network_factory=lambda: MLP(128, 64, 1)
)
logits = tower(x, domain_ids)  # 每个 sample 自动路由到对应 domain 的 Tower
```

### MultiTaskContainer

**文件**: [`models/utils/container.py:74`](src/betterbole/models/utils/container.py:74)

多任务网络容器 —— 为每个 task 维护独立网络，全量分发：

```python
tasks = MultiTaskContainer(
    task_names=["ctr", "cvr"],
    network_factory=lambda: MLP(128, 64, 1)
)
outputs = tasks(x)  # {"ctr": Tensor, "cvr": Tensor}
```

### DomainTowerHead

**文件**: [`models/msr/components/heads.py:11`](src/betterbole/models/msr/components/heads.py:11)

标准的多场景输出头 —— 每个 domain 一个独立的 MLP Tower，输出标量 logit。

```python
head = DomainTowerHead(num_domains=3, input_dim=128, hidden_dims=(64,))
logits = head(x, domain_ids)  # → (B,), 每个 sample 取其 domain 的 Tower 输出
```

### MLP 与 DNN

**文件**: [`models/utils/general.py`](src/betterbole/models/utils/general.py)

| 类 | 说明 |
|------|------|
| [`MLP`](src/betterbole/models/utils/general.py:44) | 标准 MLP：Linear → BN → Activation → Dropout（最后一层无 BN/Activation/Dropout） |
| [`DNN`](src/betterbole/models/utils/general.py:7) | DeepCTR 兼容 MLP，支持 Dice 激活函数 |
| [`FeatureBifurcator`](src/betterbole/models/utils/general.py:101) | 特征分化：将输出分离为"静态基准(Bias)" + "零均值动态波动(Fluctuation)" |
| [`BifurcatedMLP`](src/betterbole/models/utils/general.py:173) | 分叉 MLP：最后一层输出 (B, 2, D) = [Bias, Fluctuation] |
| [`ModuleFactory`](src/betterbole/models/utils/general.py:213) | 专家/门控/Tower 工厂函数 |

---

**下一步**: 继续阅读 [`UTILS.md`](docs/betterbole/UTILS.md) 了解工具模块
