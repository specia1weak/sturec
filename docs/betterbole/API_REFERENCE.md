# API 参考

这份索引只列当前源码里确实存在、并且在训练主链路里有意义的入口。更细的行为解释请跳转到 `details/` 文档或直接看源码。

## 先记住这些入口

| 主题 | 入口 |
|---|---|
| 特征编排 | `betterbole.emb.SchemaManager` |
| 特征定义 | `betterbole.emb.schema.*Setting` |
| 流式数据集 | `betterbole.data.dataset.ParquetStreamDataset` |
| 统一 embedding | `betterbole.emb.emblayer.OmniEmbLayer` |
| 多场景模型构建 | `betterbole.models.msr.build_model` |
| 训练 | `betterbole.core.train.BaseTrainer` |
| 评估 | `betterbole.evaluate.evaluator.Evaluator` |
| 多评估器编排 | `betterbole.evaluate.manager.EvaluatorManager` |
| 参数解析 | `betterbole.experiment.param.ParamManager` |

## Core

### 枚举与交互

- [`core/enum_type.py`](../../src/betterbole/core/enum_type.py)
  - `ModelType`
  - `KGDataLoaderState`
  - `EvaluatorType`
  - `InputType`
  - `FeatureType`
  - `FeatureSource`
- [`core/interaction.py`](../../src/betterbole/core/interaction.py)
  - `Interaction`
  - `cat_interactions(interactions)`

### 训练

- [`core/train/context.py`](../../src/betterbole/core/train/context.py)
  - `TrainContext`
  - `TrainerDataLoaders`
  - `TrainerComponents`
- [`core/train/hooks.py`](../../src/betterbole/core/train/hooks.py)
  - `CustomTrainStepProtocol`
  - `TrainerHooksProtocol`
- [`core/train/early_stepper.py`](../../src/betterbole/core/train/early_stepper.py)
  - `EarlyStopper`
  - `EarlyStepper`，兼容别名
- [`core/train/trainer.py`](../../src/betterbole/core/train/trainer.py)
  - `BaseTrainer`

## Data

- [`data/dataset.py`](../../src/betterbole/data/dataset.py)
  - `DataScanner`
  - `DataTransformer`
  - `ShuffleBuffer`
  - `PipelineStreamDataset`
  - `ParquetStreamDataset`
  - `RawParquetStreamDataset`
- [`data/padding.py`](../../src/betterbole/data/padding.py)
  - `ColumnFormatter`
  - `DenseFormatter`
  - `VectorDenseFormatter`
  - `IntFormatter`
  - `FallbackFormatter`
  - `PaddedIntSequenceFormatter`
  - `PaddedFloatSequenceFormatter`
  - `PaddedNestedSequenceFormatter`
  - `FeatureContext`
  - `TensorFormatter`

## Embedding 与特征

### 特征 schema

- [`emb/schema/base.py`](../../src/betterbole/emb/schema/base.py)
  - `EmbType`
  - `SeqGroupConfig`
  - `EmbSetting`
- [`emb/schema/categorical.py`](../../src/betterbole/emb/schema/categorical.py)
  - `SparseEmbSetting`
  - `MultiSparseSetting`
  - `QuantileEmbSetting`
- [`emb/schema/numerical.py`](../../src/betterbole/emb/schema/numerical.py)
  - `MinMaxDenseSetting`
  - `VectorDenseSetting`
- [`emb/schema/sequence.py`](../../src/betterbole/emb/schema/sequence.py)
  - `SequenceSetting`

### 编排与 embedding

- [`emb/manager.py`](../../src/betterbole/emb/manager.py)
  - `SchemaManager`
- [`emb/emblayer.py`](../../src/betterbole/emb/emblayer.py)
  - `RecEmbedding`
  - `BoleEmbLayer`
  - `SideEmb`
  - `EmbView`
  - `SeqGroupView`
  - `OmniEmbLayer`
- [`emb/split.py`](../../src/betterbole/emb/split.py)
  - `LooConfig`
  - `SequentialRatioConfig`
  - `TimeSplitConfig`
  - `RandomRatioConfig`
  - `build_split_config()`
  - `create_split_strategy()`

## Evaluate

- [`evaluate/metrics.py`](../../src/betterbole/evaluate/metrics.py)
  - `AUC`
  - `LogLoss`
  - `GAUC`
  - `HR`
  - `NDCG`
  - `METRIC_REGISTRY`
- [`evaluate/evaluator.py`](../../src/betterbole/evaluate/evaluator.py)
  - `PointWiseEvaluator`
  - `TopKEvaluator`
  - `Evaluator`
  - `EvaluatorDecorator`
  - `LogDecorator`
- [`evaluate/manager.py`](../../src/betterbole/evaluate/manager.py)
  - `EvaluatorFilter`
  - `AllowAllFilter`
  - `DomainFilter`
  - `EvaluatorManager`

## Models

### 基础模型层

- [`models/base.py`](../../src/betterbole/models/base.py)
  - `ModelOutput`
  - `BaseModel`
- [`models/msr/base.py`](../../src/betterbole/models/msr/base.py)
  - `MSRModel`
- [`models/msr/components/heads.py`](../../src/betterbole/models/msr/components/heads.py)
  - `DomainTowerHead`

### 构建入口

- [`models/msr/__init__.py`](../../src/betterbole/models/msr/__init__.py)
  - `MODEL_REGISTRY`
  - `build_model(schema_manager, num_domains, model_cls, **model_kwargs)`

当前注册表包含这些名字：

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

### backbone 与模型工具

- [`models/backbone/__init__.py`](../../src/betterbole/models/backbone/__init__.py)
  - `BACKBONE_REGISTRY`
  - `resolve_backbone()`
  - `build()`
  - `SmallBackboneFactory`
- [`models/utils/container.py`](../../src/betterbole/models/utils/container.py)
  - `domain_select`
  - `MultiScenarioCloneBase`
  - `MultiScenarioContainer`
  - `MultiTaskContainer`
- [`models/utils/general.py`](../../src/betterbole/models/utils/general.py)
  - `DNN`
  - `MLP`
  - `FeatureBifurcator`
  - `BifurcatedLinear`
  - `BifurcatedMLP`
  - `ModuleFactory`

## Experiment

- [`experiment/__init__.py`](../../src/betterbole/experiment/__init__.py)
  - `ROOT_DIR`
  - `WORKSPACE`
  - `ignore_future_warning()`
  - `change_root_workdir()`
  - `set_all()`
- [`experiment/param.py`](../../src/betterbole/experiment/param.py)
  - `ConfigBase`
  - `seed_everything()`
  - `ParamManager`
- [`experiment/engine.py`](../../src/betterbole/experiment/engine.py)
  - `GridSearchEngine`
- [`experiment/tracker.py`](../../src/betterbole/experiment/tracker.py)
  - `TrainingTracker`

## Datasets

- [`datasets/base.py`](../../src/betterbole/datasets/base.py)
  - `DatasetBase`
- [`datasets/movielens.py`](../../src/betterbole/datasets/movielens.py)
  - `MovieLensDataset`
- [`datasets/kuairand.py`](../../src/betterbole/datasets/kuairand.py)
  - `KuaiRandDataset`
- [`datasets/aliccp.py`](../../src/betterbole/datasets/aliccp.py)
  - `AliCCPDataset`
- [`datasets/douban.py`](../../src/betterbole/datasets/douban.py)
  - `DoubanDataset`
- [`datasets/taac2026.py`](../../src/betterbole/datasets/taac2026.py)
  - `TAAC2026Dataset`
- [`datasets/overview.py`](../../src/betterbole/datasets/overview.py)
  - `get_general_info()`
  - `get_group_stats()`
  - `get_head_info()`

## Utils

注意：`betterbole.utils` 顶层包只重导出了时间桶相关工具。其他工具需要从子模块导入。

- [`utils/optimize.py`](../../src/betterbole/utils/optimize.py)
  - `split_params_by_decay()`
  - `create_complex_optimizer_groups()`
- [`utils/sequential.py`](../../src/betterbole/utils/sequential.py)
  - `extract_history_sequences()`
  - `extract_seq_len()`
  - `extract_history_items()`
- [`utils/sample.py`](../../src/betterbole/utils/sample.py)
  - `AbstractSampler`
  - `PolarsUISampler`
  - `extract_history_dict()`
- [`utils/time.py`](../../src/betterbole/utils/time.py)
  - `timer`
  - `NamedTimer`
  - `CudaNamedTimer`
- [`utils/time_bucket.py`](../../src/betterbole/utils/time_bucket.py)
  - `TIME_BUCKET_BOUNDARIES`
  - `get_num_time_buckets()`
  - `build_padding_mask()`
  - `bucketize_relative_time()`
  - `RelativeTimeEmbedding`
- [`utils/recorder.py`](../../src/betterbole/utils/recorder.py)
  - `IndividualReLURecorder`
  - `ExplicitFeatureRecorder`
- [`utils/visualize.py`](../../src/betterbole/utils/visualize.py)
  - `plot_bias_distributions()`
  - `plot_sparsity_distributions()`
  - `plot_sparsity_ecdf()`
  - `plot_power2_sparsity()`
- [`utils/process.py`](../../src/betterbole/utils/process.py)
  - `set_priority()`
  - `get_cpu_load_rank()`
  - `set_affinity()`
  - `get_affinity()`
  - `get_idle_cpus()`
- [`utils/task_chain.py`](../../src/betterbole/utils/task_chain.py)
  - `auto_queue()`

## 当前不建议照抄的历史名字

如果你在旧笔记、旧实验脚本里看到以下符号，请优先回到 `src/betterbole` 确认：

- `SparseSetEmbSetting`
- `SparseSeqEmbSetting`
- `SharedVocabSeqSetting`
- `SeqGroupEmbSetting`
- `preset_workdir`

这些名字至少不是当前文档所依据的公共主路径。
