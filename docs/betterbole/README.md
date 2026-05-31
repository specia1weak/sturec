# betterbole 文档总览

`betterbole` 是一个围绕多场景推荐训练流程搭建的 PyTorch 工程，当前源码核心位于 [`src/betterbole`](../../src/betterbole)。它覆盖了四件事：

1. 用 `SchemaManager` 把原始宽表转成稳定可复用的编码规则。
2. 用 `ParquetStreamDataset` / `RawParquetStreamDataset` 把 Parquet 流式喂给模型。
3. 用 `OmniEmbLayer`、`MSRModel` 和一组 backbone 训练多场景模型。
4. 用 `EvaluatorManager`、`BaseTrainer`、`EarlyStopper` 组织评估与训练。

这套文档按当前仓库源码重写，重点修正了旧文档里的三类问题：

- 类名、路径、注册名和实际源码不一致。
- 示例沿用了历史 API，例如 `SparseSetEmbSetting`、`preset_workdir`，但这些并不是当前实现。
- 训练/评估细节被写得过于理想化，没有反映实际约束和坑点。

## 先看什么

- 新用户先看 [`QUICKSTART.md`](QUICKSTART.md)
- 想把握分层结构先看 [`ARCHITECTURE.md`](ARCHITECTURE.md)
- 需要查类和方法入口看 [`API_REFERENCE.md`](API_REFERENCE.md)
- 要接入新模型看 [`ADD_MSR_MODEL_AND_VERIFY.md`](ADD_MSR_MODEL_AND_VERIFY.md)

## 当前推荐认知模型

源码可以按“数据处理 -> 模型 -> 训练”理解，不必先记住所有目录：

| 模块 | 目录 | 先关心什么 |
|---|---|---|
| 特征与编码 | `emb/` | `EmbSetting`、`SchemaManager`、`OmniEmbLayer` |
| 数据流 | `data/` | `ParquetStreamDataset`、`RawParquetStreamDataset` |
| 训练 | `core/train/` | `BaseTrainer`、`EarlyStopper` |
| 评估 | `evaluate/` | `Evaluator`、`EvaluatorManager` |
| 多场景模型 | `models/msr/` | `build_model()`、`MODEL_REGISTRY` |
| 参数与实验 | `experiment/` | `ConfigBase`、`ParamManager`、`GridSearchEngine` |

## 当前版本里最重要的事实

- 真正稳定的特征定义类是 `SparseEmbSetting`、`MultiSparseSetting`、`QuantileEmbSetting`、`MinMaxDenseSetting`、`VectorDenseSetting`、`SequenceSetting`。
- `SchemaManager.fit()` 会把 schema 持久化到 `work_dir/feature_meta.json`。同一个 `work_dir` 下如果元数据已存在，再次 `fit()` 会直接加载旧 schema，而不是强制重算。
- `ParquetStreamDataset` 和 `RawParquetStreamDataset` 都是 `IterableDataset`。如果你自己再包 `DataLoader`，必须使用 `batch_size=None`。
- `BaseTrainer.evaluate_epoch()` 会把 `model.predict()` 的输出直接传给 point-wise evaluator。
  如果你想算 `logloss`，`predict()` 最好返回概率值；只算 AUC 时返回 raw logits 也能工作。
- `betterbole.utils` 顶层包只重导出了时间桶相关工具。大多数工具函数需要从子模块导入，例如 `betterbole.utils.optimize`。
- `@examples/ml-1m` 下有不少历史实验稿，含旧类名，不应当作为当前 API 依据。当前更接近源码状态的完整实验入口是 [`@examples/kuairand-1k/kuairan1k.py`](../../@examples/kuairand-1k/kuairan1k.py)。

## 文档地图

- 总览
  - [`ARCHITECTURE.md`](ARCHITECTURE.md)
  - [`API_REFERENCE.md`](API_REFERENCE.md)
  - [`QUICKSTART.md`](QUICKSTART.md)
- 详细模块
  - [`details/CORE.md`](details/CORE.md)
  - [`details/DATA.md`](details/DATA.md)
  - [`details/EMB.md`](details/EMB.md)
  - [`details/EVALUATE.md`](details/EVALUATE.md)
  - [`details/MODELS.md`](details/MODELS.md)
  - [`details/EXPERIMENT.md`](details/EXPERIMENT.md)
  - [`details/UTILS.md`](details/UTILS.md)
  - [`details/DATASETS.md`](details/DATASETS.md)

## 一条推荐的学习路径

1. 读 [`QUICKSTART.md`](QUICKSTART.md)，先跑通 `split -> fit -> transform -> dataset -> trainer`。
2. 遇到特征问题时回到 [`details/EMB.md`](details/EMB.md) 和 [`details/DATA.md`](details/DATA.md)。
3. 想换模型时看 [`details/MODELS.md`](details/MODELS.md) 里的注册表和构造方式。
4. 想查某个类的真实入口，看 [`API_REFERENCE.md`](API_REFERENCE.md) 里的路径，再跳到源码。
