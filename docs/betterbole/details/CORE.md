# core/

`core/` 提供的是最底层但最常用的两个东西：

1. 推荐训练里通用的数据容器 `Interaction`
2. 训练循环相关的 `BaseTrainer` / `TrainContext` / `EarlyStopper`

## 1. 枚举

源码在 [`src/betterbole/core/enum_type.py`](../../src/betterbole/core/enum_type.py)。

### `ModelType`

- `GENERAL`
- `SEQUENTIAL`
- `CONTEXT`
- `KNOWLEDGE`
- `TRADITIONAL`
- `DECISIONTREE`

### `KGDataLoaderState`

- `RSKG`
- `RS`
- `KG`

### `EvaluatorType`

- `RANKING`
- `VALUE`

### `InputType`

- `POINTWISE`
- `PAIRWISE`
- `LISTWISE`

### `FeatureType`

- `TOKEN`
- `FLOAT`
- `TOKEN_SEQ`
- `FLOAT_SEQ`

### `FeatureSource`

- `UNKNOWN`
- `INTERACTION`
- `USER`
- `ITEM`
- `USER_ID`
- `ITEM_ID`
- `SEQ`
- `SEQ_GROUP`
- `KG`
- `NET`

## 2. `Interaction`

源码在 [`src/betterbole/core/interaction.py`](../../src/betterbole/core/interaction.py)。

`Interaction` 是一个 batch 级别的 tensor 容器，本质上可以把它理解成“能搬到 GPU 上的字典”。

```python
from betterbole.core.interaction import Interaction

batch = Interaction({
    "user_id": torch.tensor([1, 2]),
    "item_id": torch.tensor([10, 11]),
    "label": torch.tensor([1, 0]),
})
```

### 常用方法

- `to(device, selected_field=None)`
- `cpu()`
- `numpy()`
- `repeat(sizes)`
- `repeat_interleave(repeats, dim=0)`
- `update(new_inter)`
- `drop(column)`
- `shuffle()`
- `sort(by, ascending=True)`
- `add_prefix(prefix)`

### 行为特点

- `Interaction["field"]` 返回单列 tensor。
- `Interaction[index]` 返回裁切后的新 `Interaction`。
- `Interaction.to(...)` 不会原地修改，而是返回新对象。
- `shuffle()` / `sort()` 会原地改内部顺序。

### `cat_interactions(interactions)`

把多个 `Interaction` 按 batch 维拼起来。前提是字段集合完全一致。

## 3. 训练上下文

源码在 [`src/betterbole/core/train/context.py`](../../src/betterbole/core/train/context.py)。

### `TrainContext`

```python
@dataclass
class TrainContext:
    epoch: int
    global_step: int
    batch_idx: int
    optimizer: torch.optim.Optimizer
    manager: Any
    cfg: Any
    timer: Any
    recorder: Any = None
    kwargs: Dict[str, Any] = None
```

它主要用于把训练状态和实验上下文传给自定义 step / hook。

### `TrainerDataLoaders`

```python
@dataclass
class TrainerDataLoaders:
    train: Iterable[Interaction]
    valid: Iterable[Interaction]
    test: Optional[Iterable[Interaction]] = None
```

### `TrainerComponents`

```python
@dataclass
class TrainerComponents:
    evaluator_manager: EvaluatorManager
    recorder: ExplicitFeatureRecorder = ...
    timer: CudaNamedTimer = ...
    early_stepper: Optional[EarlyStopper] = None
```

## 4. Hook 协议

源码在 [`src/betterbole/core/train/hooks.py`](../../src/betterbole/core/train/hooks.py)。

### `CustomTrainStepProtocol`

实现这个协议后，`BaseTrainer.train_epoch()` 会优先调用你的 `custom_train_step(batch, ctx)`。

### `TrainerHooksProtocol`

支持三个 hook：

- `on_train_epoch_start(ctx)`
- `on_train_epoch_end(ctx)`
- `on_eval_epoch_end(metrics, ctx)`

## 5. `EarlyStopper`

源码在 [`src/betterbole/core/train/early_stepper.py`](../../src/betterbole/core/train/early_stepper.py)。

```python
EarlyStopper(patience=5, min_delta=0.0)
```

### 选择指标的规则

`step(summary_dict, epoch=None)` 会从 `summary_dict` 里挑一个指标做 early-stop 依据：

1. 优先看 `overall` / `all` 这类评估器名。
2. 再按指标名优先级搜索：
   - `auc`
   - `gauc`
   - `ndcg@10`
   - `ndcg@20`
   - `ndcg`
   - `hr@10`
   - `hr@20`
   - `hr`
   - `recall@10`
   - `recall@20`
   - `recall`
   - `logloss`
   - `loss`
3. 找不到时退化为该评估器里的第一个指标。

注意：这里的 `recall@k` 只是 `EarlyStopper` 的候选指标名之一，并不代表当前 `evaluate.metrics` 已经实现了 `recall`。

### 模式

- `loss` / `logloss` / `mae` / `mse` / `rmse` 使用 `min` 模式。
- 其他指标使用 `max` 模式。

### 返回值

`step(...) -> (is_best, should_stop)`

## 6. `BaseTrainer`

源码在 [`src/betterbole/core/train/trainer.py`](../../src/betterbole/core/train/trainer.py)。

### 初始化

```python
BaseTrainer(
    model,
    optimizer,
    manager,
    loaders,
    components,
    cfg,
)
```

### 默认流程

```text
run()
  -> train_epoch()
  -> evaluate_epoch()
  -> EarlyStopper.step(...)
  -> save_checkpoint(tag="best", metrics=...)
```

### 默认训练步

如果模型没有实现 `custom_train_step`，就会走：

```python
loss = model.calculate_loss(batch)
loss.backward()
clip_grad_norm_(...)
optimizer.step()
```

### 默认评估步

`evaluate_epoch()` 会：

1. 把 batch 移到 `cfg.device`
2. 取 `uids = batch[self.manager.uid_field]`
3. 取 `labels = batch[self.manager.label_field]`
4. 调 `scores = self.model.predict(batch)`
5. 调 `self.evaluator.collect(uids, labels, batch, batch_preds_1d=scores)`

这也是为什么 `predict()` 的语义必须和 evaluator 匹配。

### checkpoint

`save_checkpoint()` 只有在 `cfg.ckpt_dir` 非空时才会写文件。

保存字段包括：

- `epoch`
- `global_step`
- `model_state_dict`
- `optimizer_state_dict`
- `metrics`
- `experiment_name`
- `dataset_name`

## 7. 实战注意事项

- `BaseTrainer` 不是 `TrainingTracker` 的封装层，两个系统彼此独立。
- 如果你想在验证时输出 `logloss`，不要让 `predict()` 返回未 sigmoid 的 logits，除非你自己接受这个误差。
- 默认 `train_epoch()` 会对 batch 做 `batch.to(cfg.device)`，所以 `Interaction` 必须是可搬运 tensor 容器。
