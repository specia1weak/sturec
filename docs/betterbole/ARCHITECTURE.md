# 架构设计

当前仓库可以按 6 个层次理解，但更准确的说法不是“严格框架分层”，而是“按训练链路分组的模块集”。

## 分层视图

```text
L6  experiment/     参数解析、脚本调度、实验辅助
L5  core/train/     训练循环、hook、早停
L4  models/         基础模型、多场景模型、backbone、head
L3  evaluate/       指标计算、多评估器编排
L2  emb/ + data/    特征规则、编码、embedding、流式数据集
L1  core/ + utils/ + datasets/  基础枚举、Interaction、工具函数、数据路径封装
```

## 一条真实的执行链

```text
raw parquet / lazyframe
    -> SchemaManager.split_dataset(...)
    -> SchemaManager.fit(train_raw)
    -> SchemaManager.transform(...)
    -> save_as_dataset(...)
    -> ParquetStreamDataset / RawParquetStreamDataset
    -> Interaction
    -> OmniEmbLayer
    -> MSR model / custom BaseModel
    -> EvaluatorManager
    -> BaseTrainer
```

## 每层负责什么

### L1: 基础抽象

- [`core/enum_type.py`](../../src/betterbole/core/enum_type.py)
  - `FeatureSource`、`InputType`、`EvaluatorType` 等基础枚举。
- [`core/interaction.py`](../../src/betterbole/core/interaction.py)
  - batch 级别的 tensor 容器，训练和评估都围绕它传递。
- [`utils/`](../../src/betterbole/utils)
  - 负采样、时序序列构造、优化器分组、时间桶、可视化等杂项工具。
- [`datasets/`](../../src/betterbole/datasets)
  - 若干本地数据集路径适配器，不负责自动下载。

### L2: 特征与数据流

- [`emb/schema/`](../../src/betterbole/emb/schema)
  - 每个字段如何拟合、如何转换、如何转 tensor、如何前向取值，都由 `EmbSetting` 子类定义。
- [`emb/manager.py`](../../src/betterbole/emb/manager.py)
  - 负责 `fit / transform / split / save / load` 的总调度。
- [`emb/emblayer.py`](../../src/betterbole/emb/emblayer.py)
  - `OmniEmbLayer` 把当前 batch 的 `Interaction` 转成模型可用的 embedding。
- [`data/dataset.py`](../../src/betterbole/data/dataset.py)
  - 流式扫描 parquet，按 worker 分片、shuffle、format，最终产出 `Interaction`。

### L3: 评估

- [`evaluate/evaluator.py`](../../src/betterbole/evaluate/evaluator.py)
  - 单个 evaluator，内部组合 point-wise / top-k evaluator。
- [`evaluate/manager.py`](../../src/betterbole/evaluate/manager.py)
  - 多 evaluator 注册表，支持按 domain 过滤后分别汇总。

### L4: 模型

- [`models/base.py`](../../src/betterbole/models/base.py)
  - `BaseModel` 自动挂上 `OmniEmbLayer`。
- [`models/msr/`](../../src/betterbole/models/msr)
  - 所有多场景模型的实现和注册表。
- [`models/backbone/`](../../src/betterbole/models/backbone)
  - 共享底座、MMoE、PLE、STAR 等骨干实现。
- [`models/msr/components/heads.py`](../../src/betterbole/models/msr/components/heads.py)
  - `DomainTowerHead` 把共享表征映射到 domain-specific logit。

### L5: 训练

- [`core/train/trainer.py`](../../src/betterbole/core/train/trainer.py)
  - `BaseTrainer` 定义默认训练/验证循环。
- [`core/train/hooks.py`](../../src/betterbole/core/train/hooks.py)
  - `custom_train_step` 和 epoch hook 的协议。
- [`core/train/early_stepper.py`](../../src/betterbole/core/train/early_stepper.py)
  - 自动挑指标、维护最佳值和 patience。

### L6: 参数和实验

- [`experiment/param.py`](../../src/betterbole/experiment/param.py)
  - `ConfigBase`、`ParamManager`、`seed_everything`。
- [`experiment/engine.py`](../../src/betterbole/experiment/engine.py)
  - `GridSearchEngine` 用 subprocess 启多个实验脚本。
- [`experiment/tracker.py`](../../src/betterbole/experiment/tracker.py)
  - 一个独立的 checkpoint / vector 跟踪小工具，但当前 `BaseTrainer` 默认不会自动接入它。

## 三个关键契约

### 1. `SchemaManager` 产出的字段集合决定后续 batch 结构

`manager.fields()` 会把：

- 所有 `setting.get_output_field_names()`
- `time_field`
- `label_fields`
- `domain_fields`

统一加入输出列集合。后面的 dataset formatter 和 trainer 都默认按这个结构工作。

### 2. 模型前向依赖 `Interaction`

大多数 MSR 模型的最小路径是：

```python
x = self.omni_embedding.whole(interaction)
domain_ids = interaction[self.DOMAIN].long()
logits = self.backbone(x, domain_ids)
```

这也是为什么 domain 列即使不做 embedding，也必须保留在 batch 里。

### 3. 评估默认走 `model.predict()`

`BaseTrainer` 不区分“训练 logits”和“评估概率”，它只会调用 `predict()`。因此你需要自己保证：

- 评估指标和 `predict()` 输出语义匹配。
- 如果模型需要特殊评估逻辑，可以覆写 `predict_step()` 或整个 `evaluate_epoch()`。

## 哪些东西不要误以为是“框架约定”

- `TrainingTracker` 不是 trainer 默认依赖。
- `prepare_data()` 虽然存在，但源码注释已经明确“不建议使用”；推荐显式走 `split -> fit -> transform`。
- `@examples/ml-1m` 中的很多实验文件是历史草稿，不代表当前公共 API。
