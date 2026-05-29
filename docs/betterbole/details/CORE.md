# ⚙️ 核心模块 (core/)

> **层级**: L1 (基础层) + L5 (训练层)
>
> L1 部分: 枚举类型 + `Interaction` 数据容器 → 自包含，无下层依赖
>
> L5 部分: `BaseTrainer` + `TrainContext` + `Hooks` + `EarlyStopper` → 依赖 L1~L4

---

## 组织说明

```
L1 ──────────────────────────────────────────────────
core/enum_type.py     → 6 个核心枚举（FeatureSource 等）
core/interaction.py   → Tensor 字典容器
─────────────────────────────────────────────────────
L5 ──────────────────────────────────────────────────
core/train/trainer.py       → BaseTrainer 训练循环
core/train/context.py       → 训练上下文
core/train/hooks.py         → Hook 扩展协议
core/train/early_stepper.py → 早停机制
─────────────────────────────────────────────────────
```

## 枚举类型 — [`core/enum_type.py`](../../src/betterbole/core/enum_type.py)

### `ModelType`

| 值 | 说明 |
|----|------|
| `GENERAL` | 通用推荐 |
| `SEQUENTIAL` | 序列推荐 |
| `CONTEXT` | 上下文感知推荐 |
| `KNOWLEDGE` | 基于知识的推荐 |

### `EvaluatorType`

| 值 | 说明 |
|----|------|
| `RANKING` | 排序指标 (NDCG/Recall/HR) |
| `VALUE` | 数值指标 (AUC/LogLoss) |

### `InputType`

| 值 | 说明 |
|----|------|
| `POINTWISE` | 单点输入 `(uid, iid, label)` |
| `PAIRWISE` | 成对输入 `(uid, pos_iid, neg_iid)` |

### `FeatureType` — 特征值类型

| 值 | 说明 |
|----|------|
| `TOKEN` | 离散 Token (user_id, item_id) |
| `FLOAT` | 浮点数 (rating, timestamp) |
| `TOKEN_SEQ` | 离散序列 (review) |
| `FLOAT_SEQ` | 浮点序列 (预训练向量) |

### `FeatureSource` — 特征来源

| 值 | 说明 |
|----|------|
| `USER_ID` | 用户主键 |
| `ITEM_ID` | 物品主键 |
| `USER` | 用户侧属性 |
| `ITEM` | 物品侧属性 |
| `INTERACTION` | 交互侧属性 |
| `SEQ` | 序列特征 |
| `SEQ_GROUP` | 序列组 |
| `KG` | 知识图谱 |
| `NET` | 社交网络 |

---

## Interaction — [`core/interaction.py`](../../src/betterbole/core/interaction.py)

推荐系统中**一个 Batch 的交互记录**的 Tensor 字典容器。

### 核心特性

```
Interaction 内部结构：{ "user_id": Tensor[B], "item_id": Tensor[B], "label": Tensor[B] }
```

### 常用方法

| 方法 | 说明 |
|------|------|
| `to(device)` | 将 Tensor 转移到指定设备 |
| `cpu()` / `numpy()` | 转移至 CPU 或转为 NumPy |
| `repeat(n)` / `repeat_interleave(n)` | 沿 Batch 维重复 |
| `shuffle()` | 原地打乱 |
| `sort(by, ascending)` | 按指定字段排序 |
| `update(new_inter)` | 合并另一个 Interaction |
| `drop(column)` | 删除列 |
| `add_prefix(prefix)` | 列名前缀 |

### `cat_interactions(interactions)`

拼接多个 Interaction → 一个 Interaction。

---

## 训练框架 — [`core/train/`](../../src/betterbole/core/train/)

### `BaseTrainer` — [`trainer.py`](../../src/betterbole/core/train/trainer.py)

统一的训练基座：

```
初始化：model + optimizer + manager + loaders + components + cfg
    │
    ▼
run():
    for epoch in max_epochs:
        train_epoch()          ← model.calculate_loss() + optimizer.step()
        evaluate_epoch()       ← model.predict() + evaluator.collect() + summary()
        early_stepper.step()   ← 早停 / 保存最佳 Checkpoint
```

### `TrainContext` — [`context.py`](../../src/betterbole/core/train/context.py)

训练时的上下文容器，包含：

| 字段 | 说明 |
|------|------|
| `epoch` | 当前 epoch |
| `global_step` | 全局 step |
| `batch_idx` | 当前 batch |
| `optimizer` | 优化器引用 |
| `manager` | SchemaManager |
| `cfg` | 配置 |
| `timer` | 计时器 |
| `recorder` | 特征记录器 |

### `TrainerDataLoaders` — [`context.py`](../../src/betterbole/core/train/context.py)

```python
@dataclass
class TrainerDataLoaders:
    train: Iterable[Interaction]   # 训练 DataLoader
    valid: Iterable[Interaction]   # 验证 DataLoader
    test: Optional[Iterable[Interaction]]  # 可选测试集
```

### `TrainerComponents` — [`context.py`](../../src/betterbole/core/train/context.py)

```python
@dataclass
class TrainerComponents:
    evaluator_manager: EvaluatorManager  # 评估管理器
    recorder: ExplicitFeatureRecorder    # 特征记录器
    timer: CudaNamedTimer                # CUDA 计时器
    early_stepper: Optional[EarlyStopper] # 早停器
```

### Hooks 协议 — [`hooks.py`](../../src/betterbole/core/train/hooks.py)

模型可以通过实现这些 Protocol 接入训练流程：

```python
class CustomTrainStepProtocol(Protocol):
    def custom_train_step(self, batch_interaction, ctx: TrainContext): ...

class TrainerHooksProtocol(Protocol):
    def on_train_epoch_start(self, ctx: TrainContext): ...
    def on_train_epoch_end(self, ctx: TrainContext): ...
    def on_eval_epoch_end(self, metrics, ctx: TrainContext): ...
```

### `EarlyStopper` — [`early_stepper.py`](../../src/betterbole/core/train/early_stepper.py)

自动早停 + 最优模型保存：

```python
EarlyStopper(patience=5, min_delta=0.0)
    .step(summary_dict, epoch=epoch)
    → (is_best, should_stop)
```

- 自动识别指标模式：`loss/logloss/mae/mse/rmse` → min 模式，其余 → max 模式
- 指标优先级：`auc > gauc > ndcg@10 > hr@10 > recall@10 > logloss > loss`
