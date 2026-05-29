# API 参考 (`API_REFERENCE.md`)

> betterbole 完整类/函数索引。按**层级** (L1~L6) 组织，每个条目链接到对应详细文档。

---

## 总览：各层级包含的模块

| 层级 | 模块 | 文档 |
|------|------|------|
| **L6** | [`experiment/`](docs/betterbole/EXPERIMENT.md) | 实验管理 |
| **L5** | [`core/train/`](docs/betterbole/CORE.md) | 训练循环 |
| **L4** | [`models/`](docs/betterbole/MODELS.md) | 模型体系 |
| **L3** | [`evaluate/`](docs/betterbole/EVALUATE.md) | 评估体系 |
| **L2** | [`emb/`](docs/betterbole/EMB.md) + [`data/`](docs/betterbole/DATA.md) | 数据与嵌入 |
| **L1** | [`core/`](docs/betterbole/CORE.md) + [`utils/`](docs/betterbole/UTILS.md) + [`datasets/`](docs/betterbole/DATASETS.md) | 基础层 |

---

### L1 - 基础层

> betterbole 完整类/函数索引。按模块组织，每个条目链接到对应详细文档。

## 核心模块 (`core/`)

### 枚举类型 (`core/enum_type.py`)

| 枚举 | 值 |
|------|----|
| `ModelType` | GENERAL, KG, GENERATIVE |
| `FeatureType` | ID, USER, ITEM, INTERACTION, SEQ |
| `FeatureSource` | USER_ID, ITEM_ID, USER, ITEM, INTERACTION, SEQ, DOMAIN, LABEL |
| `EvaluatorType` | POINTWISE, TOPK, PAIRWISE, SIDES |
| `InputType` | POINTWISE, PAIRWISE, LISTWISE |
| `KGDataLoaderState` | WORK, TEST, TRAIN, VALID |

**文档**: [`CORE.md`](docs/betterbole/CORE.md#枚举类型-enum_typepy)

### 交互 (`core/interaction.py`)

| 类/方法 | 说明 |
|---------|------|
| `Interaction` | Tensor 字典容器，类似 dict[str, Tensor] |
| `Interaction.to(device)` | 移至指定设备 |
| `Interaction.cpu()` | 移至 CPU |
| `Interaction.numpy()` | 转为 numpy（返回 dict） |
| `Interaction.repeat(sizes)` | 沿 batch 维重复 |
| `Interaction.repeat_interleave(repeats)` | 沿 batch 维插值重复 |
| `Interaction.update(new_inter)` | 合并另一个 Interaction |
| `Interaction.drop(column)` | 删除字段 |
| `Interaction.shuffle()` | 随机打乱 batch 顺序 |
| `Interaction.sort(by, ascending)` | 按指定字段排序 |
| `cat_interactions(interactions)` | 合并多个 Interaction |

**文档**: [`CORE.md`](docs/betterbole/CORE.md#interaction-交互数据容器)

### 训练器 (`core/train/trainer.py`)

| 类/方法 | 说明 |
|---------|------|
| `BaseTrainer` | 通用训练器 |
| `BaseTrainer.run()` | 完整训练循环（train_epoch → evaluate_epoch → save_checkpoint） |
| `BaseTrainer.train_epoch()` | 单 epoch 训练，支持 `custom_train_step` |
| `BaseTrainer.evaluate_epoch()` | @torch.no_grad() 评估 |
| `BaseTrainer.save_checkpoint(tag, metrics)` | 保存 checkpoint |
| `TrainContext` | 训练上下文 dataclass（model, optimizer, epoch, step 等） |
| `TrainerDataLoaders` | 数据加载器 dataclass（train, valid, test） |
| `TrainerComponents` | 组件 dataclass（model, optimizer, evaluator_manager, early_stopper, hooks） |

**文档**: [`CORE.md`](docs/betterbole/CORE.md#训练器-basetrainer)

### Hooks (`core/train/hooks.py`)

| Protocol | 方法 |
|----------|------|
| `CustomTrainStepProtocol` | `custom_train_step(batch, ctx)` |
| `TrainerHooksProtocol` | `on_train_epoch_start(ctx)`, `on_train_epoch_end(ctx)`, `on_eval_epoch_start(ctx)`, `on_eval_epoch_end(metrics, ctx)` |

### Early Stopper (`core/train/early_stepper.py`)

| 类 | 说明 |
|----|------|
| `EarlyStopper(patience=5)` | 基于验证集指标的早停 |
| `EarlyStopper.step(summary_dict)` | 每次验证后调用，返回 `(is_best, should_stop)` |

**文档**: [`CORE.md`](docs/betterbole/CORE.md#earlystopper-早停)

---

## 数据模块 (`data/`)

### 数据集 (`data/dataset.py`)

| 类 | 说明 |
|----|------|
| `DataScanner` | 从 LazyFrame 或 Parquet 源读取批数据 |
| `DataTransformer` | 对 PyArrow Table 应用 Polars transform |
| `ShuffleBuffer` | 流式 shuffle 缓冲区（on-the-fly） |
| `PipelineStreamDataset` | PyTorch IterableDataset 流水线基类 |
| `ParquetStreamDataset(PipelineStreamDataset)` | Parquet 流式数据集 |
| `RawParquetStreamDataset(PipelineStreamDataset)` | 原始 Parquet 流式数据集（无 transform） |

**文档**: [`DATA.md`](docs/betterbole/DATA.md#数据集-datasetpy)

### 填充 (`data/padding.py`)

| 类 | 说明 |
|----|------|
| `ColumnFormatter` | 列格式化基类 |
| `IntFormatter` | 标量整数格式化 |
| `DenseFormatter` | 稠密标量格式化 |
| `VectorDenseFormatter(dim)` | 定长向量格式化 |
| `PaddedIntSequenceFormatter(max_len)` | 填充整数序列 |
| `PaddedFloatSequenceFormatter(max_len)` | 填充浮点序列 |
| `PaddedNestedSequenceFormatter(max_seq_len, max_inner_len)` | 填充嵌套序列 |
| `FeatureContext` | 特征表结构描述（列名 → 类型） |
| `TensorFormatter` | Tensor 格式化器（编译 Formatter → format_tensors） |

**文档**: [`DATA.md`](docs/betterbole/DATA.md#填充-paddingpy)

---

## 嵌入模块 (`emb/`)

### Schema 定义 (`emb/schema/`)

| 类 | `emb_type` | 说明 |
|----|-----------|------|
| `EmbSetting` (ABC) | UNKNOWN | 嵌入配置基类 |
| `SparseEmbSetting` | SPARSE | 单值离散特征（词表 + Embedding） |
| `MultiSparseSetting` | MULTI_SPARSE | 多值离散特征（Set, e.g. 标签列表） |
| `QuantileEmbSetting` | QUANTILE | 分位数离散化（连续值 → 桶 → Embedding） |
| `MinMaxDenseSetting` | DENSE | Min-Max 归一化标量 |
| `VectorDenseSetting` | VECTOR_DENSE | 定长向量输入 |
| `SequenceSetting` | SEQUENCE | 历史行为序列（复用子设置的词表） |
| `EmbType` | - | 嵌入类型枚举 |
| `SeqGroupConfig` | - | 序列分组配置 dataclass |

**文档**: [`EMB.md`](docs/betterbole/EMB.md#schema-定义-embschema)

### 管理器 (`emb/manager.py`)

| 方法 | 说明 |
|------|------|
| `SchemaManager(settings_list, work_dir)` | 初始化（注册 EmbSetting、设置工作目录） |
| `.fit(train_raw_lf)` | 在训练集上拟合（构建词表、统计 min/max、计算分位数） |
| `.transform(raw_lf)` | 使用拟合结果转换数据（OOV 替换、归一化、序列截断） |
| `.prepare_data(lazy_df)` | fit + transform 一步完成 |
| `.split_dataset(lf, strategy_name, **kwargs)` | 按策略切分数据集 |
| `.save_as_dataset(train_lf, valid_lf, test_lf)` | 保存为 Parquet 数据集 + feature_meta.json |
| `.save_schema()` | 保存 schema 到 `feature_meta.json` |
| `.load_schema()` | 从 `feature_meta.json` 加载 schema |
| `.generate_profiles(lazy_df)` | 生成 user/item profile |
| `.fields()` | 返回所有字段名列表 |
| `.get_setting(field_name)` | 按字段名获取 EmbSetting |
| `.source2emb_dim(*sources)` | 计算指定 FeatureSource 总嵌入维度 |

**文档**: [`EMB.md`](docs/betterbole/EMB.md#schemamanager-核心管理器)

### 嵌入层 (`emb/emblayer.py`)

| 类 | 说明 |
|----|------|
| `RecEmbedding` | 推荐系统专用 Embedding（支持 reinitialize, padding_idx） |
| `BoleEmbLayer` | 基础嵌入层（EmbSetting → nn.Embedding + formatter） |
| `OmniEmbLayer` | **统一嵌入层** — 注册所有 EmbSetting，提供多种视图 |
| `EmbView` | 嵌入视图（按 `target_sources` / `include_fields` / `exclude_fields` 路由） |
| `SeqGroupView` | 序列组视图（按 group_name 聚合序列字段） |
| `SideEmb` | 侧信息嵌入层 |
| `ProfileEncoder` | Profile 编码器（从预计算 Profile 查询嵌入） |
| `SeqEmbedder` | 序列嵌入器 |

**预定义视图** (`OmniEmbLayer`):
| 视图 | 内容 |
|------|------|
| `.whole` | 所有特征 |
| `.whole_without_domain` | 除 domain 外的所有特征 |
| `.user_all` | 所有 USER 源特征 |
| `.item_all` | 所有 ITEM 源特征 |
| `.inter_all` | 所有 INTERACTION 源特征 |
| `.user_id` | 仅 USER_ID |
| `.item_id` | 仅 ITEM_ID |
| `.domain_id` | 仅 DOMAIN |

**文档**: [`EMB.md`](docs/betterbole/EMB.md#omniemblayer-统一嵌入层)

### 数据切分 (`emb/split.py`)

| 策略 | 配置类 | 说明 |
|------|--------|------|
| `LooSplitStrategy` | `LooConfig` | Leave-One-Out：每个用户最后一条为测试 |
| `TimeSplitStrategy` | `TimeSplitConfig` | 按时间阈值切分 |
| `SequentialRatioStrategy` | `SequentialRatioConfig` | 按顺序比例切分 |
| `RandomRatioStrategy` | `RandomRatioConfig` | 随机比例切分 |

**文档**: [`EMB.md`](docs/betterbole/EMB.md#数据切分-emb_splitpy)

---

## 评估模块 (`evaluate/`)

### 评估器 (`evaluate/evaluator.py`)

| 类 | 说明 |
|----|------|
| `Evaluator` | 统一评估器（内置 PointWise + TopK 双模式） |
| `PointWiseEvaluator` | 点估评估器（AUC, LogLoss, GAUC） |
| `TopKEvaluator` | TopK 排序评估器（HR@K, NDCG@K） |
| `LogDecorator` | 评估日志装饰器（自动记录到文件） |
| `EvaluatorDecorator` | 装饰器基类 |

### 管理器 (`evaluate/manager.py`)

| 类 | 说明 |
|----|------|
| `EvaluatorManager` | 多评估器管理器（注册、收集、汇总） |
| `DomainFilter` | 场景过滤器（按 domain 分流评估） |

### 指标 (`evaluate/metrics.py`)

| 类 | 说明 |
|----|------|
| `AUC.calculate(y_true, y_pred)` | AUC 指标 |
| `LogLoss.calculate(y_true, y_pred)` | LogLoss 指标 |
| `GAUC.calculate(y_true, y_pred, users)` | 分组 AUC（按用户） |
| `HR.calculate(hits_k, k)` | 命中率 HR@K |
| `NDCG.calculate(hits_k, k)` | 归一化折损累积增益 NDCG@K |
| `METRIC_REGISTRY` | 指标注册表 dict |

**文档**: [`EVALUATE.md`](docs/betterbole/EVALUATE.md)

---

## 实验模块 (`experiment/`)

| 类 | 说明 |
|----|------|
| `ConfigBase` | 配置基类 dataclass（带 extras、类型注解校验） |
| `ParamManager(config_class)` | 参数管理器（代码参数 > CLI > 默认值） |
| `GridSearchEngine(script_path)` | 网格搜索引擎（多 GPU 并行） |
| `TrainingTracker(workdir)` | 训练追踪器（checkpoint、metrics 历史、向量导出） |
| `seed_everything(seed)` | 全局随机种子设置 |

**文档**: [`EXPERIMENT.md`](docs/betterbole/EXPERIMENT.md)

---

## 模型模块 (`models/`)

### 基类

| 类 | 说明 |
|----|------|
| `BaseModel(manager)` | 模型基类，自动挂载 OmniEmbLayer |
| `MSRModel(manager, num_domains)` | 多场景模型基类，`from_manager()` 工厂方法 |

### Backbone (`models/backbone/`)

| 类 | 说明 |
|----|------|
| `SharedBottomLess/Plus` | 共享底部 MLP |
| `SingleLayerMMoE` | 单层多门控混合专家（domain-based） |
| `SingleLayerMTLMMoE` | 多任务 MMoE（task-based） |
| `PLE` | 渐进式分层提取（2 层） |
| `PLEVersion1~4` | PLE 变体（1~4 层） |
| `StarExpert` | STAR 专家 MLP（支持 merge_with） |
| `STAR` | 场景感知表示 |
| `StarPle` | STAR + PLE 混合 |

### MSR Backbone (`models/msr/backbone/`)

| 类 | 说明 |
|----|------|
| `MSRBackbone` | MSR Backbone 基类 |
| `SharedBottomBackbone` | 共享底部 Backbone |
| `MMoEBackbone` | MMoE Backbone（gate per domain） |
| `PLEBackbone` | PLE Backbone |
| `STARBackbone` | STAR Backbone |
| `M3oEBackbone/V1/V2` | 合并门控混合专家（3 个版本） |
| `M2MBackbone` | Meta-to-Meta（Transformer + Meta Gate） |
| `PPNetBackbone` | `pepnet` 包内的个性化参数网络实现 |
| `EPNetBackbone` | `pepnet` 包内的高效参数网络实现 |

### MSR 模型 (`models/msr/`)

所有模型通过 [`build_model()`](src/betterbole/models/msr/__init__.py:36) 构建。

| 注册名 | 类 | Backbone |
|--------|-----|----------|
| `sharedbottom` | `SharedBottomModel` | SharedBottom |
| `mmoe` | `MMoEModel` | MMoE |
| `ple` | `PLEModel` | PLE |
| `star` | `STARModel` | STAR |
| `m3oe` | `M3oEModel` | M3oE |
| `m3oe_v1` | `M3oEVersion1Model` | M3oE V1 |
| `m3oe_v2` | `M3oEVersion2Model` | M3oE V2 |
| `m2m` | `M2MModel` | M2M |
| `ppnet` | `PPNetModel` | PPNet |
| `epnet` | `EPNetModel` | EPNet |
| `pepnet` | `PEPNetModel` | PEPNet unified entry |
| `hierrec` | `HierRec` | (自定义) |
| `automtl` | `AutoMTLModel` | AutoMTLSuperNet |
| `riple` | `RIPLEModel` | SharedBottom + shared/specific experts |
| `hamur` | `HAMURModel` | (自定义) |

### 模型容器 (`models/utils/container.py`)

| 类 | 说明 |
|----|------|
| `MultiScenarioContainer(num_domains, factory)` | 多场景容器：每 domain 独立副本，按 domain_ids 选取 |
| `MultiTaskContainer(task_names, factory)` | 多任务容器：每 task 独立网络，全量分发 |
| `MultiScenarioCloneBase` | 多场景分身抽象基类 |

### 模型工具 (`models/utils/`)

| 类/函数 | 说明 |
|---------|------|
| `MLP(*dims, ...)` | 标准 MLP（Linear → BN → Act → Dropout） |
| `DNN(inputs_dim, hidden_units, ...)` | DeepCTR 兼容 DNN |
| `FeatureBifurcator` | 特征分化组件（Bias + Fluctuation） |
| `BifurcatedMLP` | 分叉输出 MLP |
| `BifurcatedLinear` | 分叉线性层 |
| `ModuleFactory` | 专家/门控/Tower 工厂 |
| `build_mlp()` | MLP 构建辅助 |
| `to_dims()` | 维度标准化 |

**文档**: [`MODELS.md`](docs/betterbole/MODELS.md)

---

## 内置数据集 (`datasets/`)

| 数据集 | 文件 | 说明 |
|--------|------|------|
| MovieLens | `movielens.py` | 经典电影评分数据集 |
| Amazon | `amazon.py` | Amazon 商品评论数据集 |
| KuaiRand | `kuairand.py` | 快手短视频推荐数据集 |
| Alibaba CCP | `aliccp.py` | 阿里展示广告点击率数据集 |
| Douban | `douban.py` | 豆瓣评分数据集 |
| TAAC 2026 | `taac2026.py` | TAAC 竞赛数据集 |

### 概览工具 (`datasets/overview.py`)

| 函数 | 说明 |
|------|------|
| `get_general_info(df)` | 通用统计：行数、列数、缺失率、互动密度 |
| `get_group_stats(df, col_name)` | 分组统计：按列分组的用户数/物品数/交互数 |
| `get_head_info(df)` | 头分布：展示 Top-K 高频组合 |

**文档**: [`DATASETS.md`](docs/betterbole/DATASETS.md)

---

## 工具模块 (`utils/`)

| 模块 | 文件 | 核心功能 |
|------|------|----------|
| 优化器 | `optimize.py` | `split_params_by_decay()`、`create_complex_optimizer_groups()` |
| 负采样 | `sample.py` | `AbstractSampler`、`PolarsUISampler`、`extract_history_dict()` |
| 历史序列 | `sequential.py` | `extract_history_sequences()`（ASOF JOIN 反泄露） |
| 大文件排序 | `sort.py` | `duckdb_sort_parquet()`（DuckDB 外排序） |
| 性能计时 | `time.py` | `NamedTimer`、`CudaNamedTimer`、`timer()` |
| 时间编码 | `time_bucket.py` | `bucketize_relative_time()`、`RelativeTimeEmbedding` |
| 激活记录 | `recorder.py` | `IndividualReLURecorder`、`ExplicitFeatureRecorder` |
| 可视化 | `visualize.py` | `plot_bias_distributions`、`plot_sparsity_distributions`、`plot_sparsity_ecdf`、`plot_power2_sparsity` |
| 进程管理 | `process.py` | `set_priority()`、`get_idle_cpus()`、`get_cpu_load_rank()` |
| 任务编排 | `task_chain.py` | `auto_queue()`（PID 锁） |

**文档**: [`UTILS.md`](docs/betterbole/UTILS.md)

---

## 快速文件索引

| 文件路径 | 内容 |
|----------|------|
| [`core/enum_type.py`](src/betterbole/core/enum_type.py) | 所有枚举定义 |
| [`core/interaction.py`](src/betterbole/core/interaction.py) | Interaction 数据容器 |
| [`core/train/trainer.py`](src/betterbole/core/train/trainer.py) | BaseTrainer 训练器 |
| [`core/train/context.py`](src/betterbole/core/train/context.py) | 训练上下文 dataclass |
| [`core/train/hooks.py`](src/betterbole/core/train/hooks.py) | 训练 Hook Protocol |
| [`core/train/early_stepper.py`](src/betterbole/core/train/early_stepper.py) | 早停机制 |
| [`data/dataset.py`](src/betterbole/data/dataset.py) | 流式数据流水线 |
| [`data/padding.py`](src/betterbole/data/padding.py) | 列格式化和 Tensor 格式化 |
| [`emb/emblayer.py`](src/betterbole/emb/emblayer.py) | OmniEmbLayer 统一嵌入层 |
| [`emb/manager.py`](src/betterbole/emb/manager.py) | SchemaManager 核心管理器 |
| [`emb/split.py`](src/betterbole/emb/split.py) | 数据切分策略 |
| [`emb/schema/base.py`](src/betterbole/emb/schema/base.py) | EmbSetting 基类 |
| [`emb/schema/categorical.py`](src/betterbole/emb/schema/categorical.py) | 离散特征配置 |
| [`emb/schema/numerical.py`](src/betterbole/emb/schema/numerical.py) | 数值特征配置 |
| [`emb/schema/sequence.py`](src/betterbole/emb/schema/sequence.py) | 序列特征配置 |
| [`evaluate/evaluator.py`](src/betterbole/evaluate/evaluator.py) | 评估器 |
| [`evaluate/manager.py`](src/betterbole/evaluate/manager.py) | EvaluatorManager |
| [`evaluate/metrics.py`](src/betterbole/evaluate/metrics.py) | 评估指标 |
| [`experiment/param.py`](src/betterbole/experiment/param.py) | ConfigBase + ParamManager |
| [`experiment/engine.py`](src/betterbole/experiment/engine.py) | GridSearchEngine |
| [`experiment/tracker.py`](src/betterbole/experiment/tracker.py) | TrainingTracker |
| [`models/base.py`](src/betterbole/models/base.py) | BaseModel |
| [`models/msr/base.py`](src/betterbole/models/msr/base.py) | MSRModel |
| [`models/msr/__init__.py`](src/betterbole/models/msr/__init__.py) | MODEL_REGISTRY + build_model |
| [`models/backbone/shabtm.py`](src/betterbole/models/backbone/shabtm.py) | SharedBottom |
| [`models/backbone/mmoe.py`](src/betterbole/models/backbone/mmoe.py) | MMoE |
| [`models/backbone/ple.py`](src/betterbole/models/backbone/ple.py) | PLE |
| [`models/backbone/star.py`](src/betterbole/models/backbone/star.py) | STAR / StarPle |
| [`models/msr/m2m.py`](src/betterbole/models/msr/m2m.py) | M2M Model |
| [`models/msr/hierrec.py`](src/betterbole/models/msr/hierrec.py) | HierRec |
| [`models/msr/automtl/model.py`](src/betterbole/models/msr/automtl/model.py) | AutoMTL |
| [`models/msr/hamur/model.py`](src/betterbole/models/msr/hamur/model.py) | HAMUR |
| [`utils/optimize.py`](src/betterbole/utils/optimize.py) | 优化器参数分组 |
| [`utils/sample.py`](src/betterbole/utils/sample.py) | 负采样 |
| [`utils/sequential.py`](src/betterbole/utils/sequential.py) | 历史序列提取 |
| [`utils/sort.py`](src/betterbole/utils/sort.py) | Parquet 排序 |
| [`utils/time.py`](src/betterbole/utils/time.py) | 性能计时 |
| [`utils/time_bucket.py`](src/betterbole/utils/time_bucket.py) | 相对时间编码 |
| [`utils/visualize.py`](src/betterbole/utils/visualize.py) | 数据可视化 |
| [`utils/process.py`](src/betterbole/utils/process.py) | CPU 亲和性 |
| [`utils/task_chain.py`](src/betterbole/utils/task_chain.py) | 任务编排 |
| [`datasets/overview.py`](src/betterbole/datasets/overview.py) | 数据集概览工具 |
