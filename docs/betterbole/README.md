# betterbole — 多场景推荐系统架构

> 一个面向工业级**多场景/多任务推荐系统**的训练框架，支持从特征工程 → 数据加载 → 模型训练 → 评估的全链路。
>
> 基于 PyTorch + Polars 构建，原生支持大规模 Parquet 数据的流式处理。

---

## 📐 分层架构总览

betterbole 按**依赖方向**组织为 6 层，上层依赖下层，下层对上层无感知：

```
┌──────────────────────────────────────────────────────────────┐
│  L6: 实验层 (experiment/)                                     │
│  GridSearchEngine │ TrainingTracker │ ParamManager            │
│  依赖: L1~L5                                                  │
├──────────────────────────────────────────────────────────────┤
│  L5: 训练层 (core/train/)                                     │
│  BaseTrainer │ TrainContext │ Hooks │ EarlyStopper            │
│  依赖: L1~L4                                                  │
├──────────────────────────────────────────────────────────────┤
│  L4: 模型层 (models/)                                         │
│  BaseModel → Backbone(MMoE/PLE/STAR) → MSR(M2M/HAMUR/AutoMTL)│
│  依赖: L1~L3                                                  │
├──────────────────────────────────────────────────────────────┤
│  L3: 评估层 (evaluate/)                                       │
│  Evaluator │ PointWise/TopK │ EvaluatorManager │ Metrics      │
│  依赖: L1 (自包含)                                             │
├──────────────────────────────────────────────────────────────┤
│  L2: 嵌入层 (emb/) + 数据层 (data/)                           │
│  SchemaManager → OmniEmbLayer → DataScanner → Dataset        │
│  依赖: L1                                                    │
├──────────────────────────────────────────────────────────────┤
│  L1: 基础层 (core/ + utils/ + datasets/)                      │
│  Interaction │ FeatureSource │ 采样/排序/计时/可视化          │
│  自包含，无内部依赖                                             │
└──────────────────────────────────────────────────────────────┘
```

### 各层定位速查

| 层级 | 目录 | 核心职责 | 关键抽象 | 向下依赖 |
|------|------|----------|----------|----------|
| **L6** | [`experiment/`](docs/betterbole/EXPERIMENT.md) | 实验管理、超参搜索、Checkpoint | `ConfigBase` `ParamManager` `GridSearchEngine` | L1~L5 |
| **L5** | [`core/train/`](docs/betterbole/CORE.md) | 训练循环、早停、Hook 扩展 | `BaseTrainer` `TrainContext` `EarlyStopper` | L1~L4 |
| **L4** | [`models/`](docs/betterbole/MODELS.md) | 14 种 MSR 模型、可插拔 Backbone | `MSRModel` `Backbone` `DomainTowerHead` | L1~L3 |
| **L3** | [`evaluate/`](docs/betterbole/EVALUATE.md) | 评估指标注册、多评估器编排 | `EvaluatorManager` `DomainFilter` | L1 |
| **L2** | [`emb/`](docs/betterbole/EMB.md) + [`data/`](docs/betterbole/DATA.md) | 特征工程 → 流式加载 → Tensor | `SchemaManager` `OmniEmbLayer` `PipelineStreamDataset` | L1 |
| **L1** | [`core/`](docs/betterbole/CORE.md) + [`utils/`](docs/betterbole/UTILS.md) + [`datasets/`](docs/betterbole/DATASETS.md) | 数据容器、枚举、工具函数、内置数据集 | `Interaction` `FeatureSource` `NamedTimer` | 无 |

---

## 🚀 快速入口

| 文档 | 说明 | 适合 |
|------|------|------|
| [`QUICKSTART.md`](QUICKSTART.md) | 5 分钟跑通完整流程 | **新用户首选** |
| [`API_REFERENCE.md`](API_REFERENCE.md) | 全量类/函数索引 | 所有人快查 |
| [`details/ADD_MSR_MODEL_AND_VERIFY.md`](details/ADD_MSR_MODEL_AND_VERIFY.md) | 新增 MSR 模型并用 KuaiRand 脚本验证 | 模型开发者 |

---

## 📖 分层详解（从底向上）

### L1 - 基础层

> 所有模块共享的基础抽象。自包含，不依赖 betterbole 内部任何其他模块。

| 子模块 | 文档 | 核心内容 |
|--------|------|----------|
| `core/enum_type` | [`CORE.md → 枚举类型`](docs/betterbole/CORE.md#枚举类型-enum_typepy) | `FeatureSource`(USER_ID/ITEM_ID/USER/ITEM/...)、`ModelType`、`EvaluatorType` |
| `core/interaction` | [`CORE.md → Interaction`](docs/betterbole/CORE.md#interaction-交互数据容器) | Tensor 字典容器：`to()`/`cpu()`/`repeat()`/`shuffle()`/`sort()` |
| `datasets/` | [`DATASETS.md`](docs/betterbole/DATASETS.md) | 6 个内置数据集 (MovieLens/Amazon/KuaiRand/CCP/Douban/TAAC) |
| `utils/` | [`UTILS.md`](docs/betterbole/UTILS.md) | 优化器分组、负采样、序列提取、排序、计时、可视化、进程管理 |

---

### L2 - 数据与嵌入层

> 特征工程 + 流式数据加载 + 统一 Embedding 层。这是 betterbole 的核心竞争力所在。

```
原始 Parquet → SchemaManager(fit/transform) → DataScanner → ShuffleBuffer → TensorFormatter → Interaction
                     │
                OmniEmbLayer → EmbView(whole/user_all/item_all/...) → [B, total_dim]
```

| 子模块 | 文档 | 核心内容 |
|--------|------|----------|
| `emb/schema/` | [`EMB.md → Schema 定义`](docs/betterbole/EMB.md#schema-定义-embschema) | 6 种 `EmbSetting`：Sparse/MultiSparse/Quantile/Dense/VectorDense/Sequence |
| `emb/manager.py` | [`EMB.md → SchemaManager`](docs/betterbole/EMB.md#schemamanager-核心管理器) | `fit()`/`transform()`/`split_dataset()`/`save_as_dataset()` |
| `emb/emblayer.py` | [`EMB.md → OmniEmbLayer`](docs/betterbole/EMB.md#omniemblayer-统一嵌入层) | `EmbView` + `SeqGroupView` + `ProfileEncoder` |
| `emb/split.py` | [`EMB.md → 数据切分`](docs/betterbole/EMB.md#数据切分-emb_splitpy) | 4 种切分策略 (LOO/Time/SequentialRatio/RandomRatio) |
| `data/dataset.py` | [`DATA.md → 数据集`](docs/betterbole/DATA.md#数据集-datasetpy) | `DataScanner`/`DataTransformer`/`ShuffleBuffer`/`PipelineStreamDataset` |
| `data/padding.py` | [`DATA.md → 填充`](docs/betterbole/DATA.md#填充-paddingpy) | 6 种 `ColumnFormatter` + `TensorFormatter` |

---

### L3 - 评估层

> 自包含的评估模块。支持 PointWise 和 TopK 两种评估模式，可多评估器并行 + 按 Domain 分流。

| 子模块 | 文档 | 核心内容 |
|--------|------|----------|
| `evaluate/evaluator.py` | [`EVALUATE.md → 评估器`](docs/betterbole/EVALUATE.md#评估器-evaluatorpy) | `Evaluator`(双模式)、`PointWiseEvaluator`、`TopKEvaluator` |
| `evaluate/manager.py` | [`EVALUATE.md → EvaluatorManager`](docs/betterbole/EVALUATE.md#evaluatormanager) | 多评估器注册、`DomainFilter` 场景分流 |
| `evaluate/metrics.py` | [`EVALUATE.md → 指标`](docs/betterbole/EVALUATE.md#指标-metricspy) | AUC/LogLoss/GAUC/HR@K/NDCG@K |

---

### L4 - 模型层

> 统一的模型体系：所有模型共享 OmniEmbLayer 注入，仅通过交换 Backbone 实现架构升级。

```
OmniEmbLayer.whole → Backbone → DomainTowerHead → logits (B,)
                        │
     ┌──────────────────┼──────────────────┐
SharedBottom   MMoE/PLE/STAR    M2M/PPNet/EPNet
     │                                    │
RIPLE(残差)                     HierRec/HAMUR/AutoMTL
```

| 子模块 | 文档 | 核心内容 |
|--------|------|----------|
| `models/base.py` | [`MODELS.md → BaseModel`](docs/betterbole/MODELS.md#basemodel基类) | 模型基类，自动挂载 OmniEmbLayer |
| `models/msr/base.py` | [`MODELS.md → MSRModel`](docs/betterbole/MODELS.md#msrmodel基类) | 多场景基类，`from_manager()` 工厂 |
| `models/backbone/` | [`MODELS.md → Backbone`](docs/betterbole/MODELS.md#backbone骨干网络) | SharedBottom/MMoE/PLE/STAR/StarPle |
| `models/msr/backbone/` | [`MODELS.md → MSR Backbone`](docs/betterbole/MODELS.md#msr-backbone-modelmsrbackbone) | M2M/PPNet/EPNet/M3oE + MSR 版 MMoE/PLE/STAR |
| `models/msr/` | [`MODELS.md → 14 个 MSR 模型`](docs/betterbole/MODELS.md#msr-模型列表) | `MODEL_REGISTRY` + `build_model()` |
| `models/msr/automtl/` | [`MODELS.md → AutoMTL`](docs/betterbole/MODELS.md#automtl自动化多任务学习) | NAS 搜索最优专家结构 |
| `models/msr/hamur/` | [`MODELS.md → HAMUR`](docs/betterbole/MODELS.md#hamur超网络适配多场景) | 超网络适配器 (MLP/DCN/WideDeep) |
| `models/msr/hierrec.py` | [`MODELS.md → HierRec`](docs/betterbole/MODELS.md#hierrec层次化场景建模) | 显式+隐式生成器参数变换 |

---

### L5 - 训练层

> 通用训练循环，支持 Hook 扩展和 Early Stopping。与具体模型解耦。

| 子模块 | 文档 | 核心内容 |
|--------|------|----------|
| `core/train/trainer.py` | [`CORE.md → BaseTrainer`](docs/betterbole/CORE.md#训练器-basetrainer) | `run()` → `train_epoch()` → `evaluate_epoch()` → `save_checkpoint()` |
| `core/train/context.py` | [`CORE.md → 上下文`](docs/betterbole/CORE.md#训练上下文-traincontext) | `TrainContext`/`TrainerDataLoaders`/`TrainerComponents` |
| `core/train/hooks.py` | [`CORE.md → Hooks`](docs/betterbole/CORE.md#hooks-协议) | `CustomTrainStepProtocol` + `TrainerHooksProtocol` |
| `core/train/early_stepper.py` | [`CORE.md → EarlyStopper`](docs/betterbole/CORE.md#earlystopper-早停) | 自适应指标选择 + 早停 |

---

### L6 - 实验层

> 顶层入口。参数管理、网格搜索、Checkpoint 追踪。

| 子模块 | 文档 | 核心内容 |
|--------|------|----------|
| `experiment/param.py` | [`EXPERIMENT.md → 参数管理`](docs/betterbole/EXPERIMENT.md#参数管理-parammanager) | `ConfigBase` + `ParamManager` (code > CLI > default) |
| `experiment/engine.py` | [`EXPERIMENT.md → 网格搜索`](docs/betterbole/EXPERIMENT.md#网格搜索-gridsearchengine) | 多 GPU 并行 GridSearch + ProcessPoolExecutor |
| `experiment/tracker.py` | [`EXPERIMENT.md → 追踪器`](docs/betterbole/EXPERIMENT.md#训练追踪-trainingtracker) | Checkpoint 保存/加载、Vector 导出 |

---

## 🧩 核心设计理念

| 理念 | 说明 | 所在层 |
|------|------|--------|
| **多场景 (Multi-Scenario)** | 一个模型服务多个 Domain，每 Domain 独立 Tower/Expert | L4 |
| **多任务 (Multi-Task)** | MMoE/PLE 变体同时优化多个目标 | L4 |
| **声明式特征工程** | `EmbSetting` 定义规则，`SchemaManager` 编排执行 | L2 |
| **OOV 管理** | 训练集拟合词表 → 测试集映射，OOV + Padding | L2 |
| **流式数据管线** | Polars LazyFrame + PyArrow 零拷贝 | L2 |
| **可插拔 Backbone** | SharedBottom/MMoE/PLE/STAR 即插即用 | L4 |
| **Hook 扩展** | `custom_train_step` / `on_epoch_*` 回调 | L5 |

---

## 🔑 关键文件索引（按层级）

### L1 - 基础层
| 文件 | 作用 |
|------|------|
| [`core/enum_type.py`](../../src/betterbole/core/enum_type.py) | 6 个核心枚举 |
| [`core/interaction.py`](../../src/betterbole/core/interaction.py) | 推荐系统 Tensor 容器 |
| [`utils/sequential.py`](../../src/betterbole/utils/sequential.py) | 历史序列提取（ASOF JOIN 反泄露） |
| [`utils/sample.py`](../../src/betterbole/utils/sample.py) | 负采样器 |
| [`utils/time.py`](../../src/betterbole/utils/time.py) | 性能计时器 |
| [`utils/visualize.py`](../../src/betterbole/utils/visualize.py) | 稀疏度/偏置可视化 |

### L2 - 数据与嵌入层
| 文件 | 作用 |
|------|------|
| [`emb/schema/base.py`](../../src/betterbole/emb/schema/base.py) | `EmbSetting` 抽象基类 |
| [`emb/schema/categorical.py`](../../src/betterbole/emb/schema/categorical.py) | 离散/多值/分位数特征配置 |
| [`emb/schema/numerical.py`](../../src/betterbole/emb/schema/numerical.py) | 连续/向量特征配置 |
| [`emb/schema/sequence.py`](../../src/betterbole/emb/schema/sequence.py) | 序列特征配置 |
| [`emb/manager.py`](../../src/betterbole/emb/manager.py) | `SchemaManager` 特征工程编排引擎 |
| [`emb/emblayer.py`](../../src/betterbole/emb/emblayer.py) | `OmniEmbLayer` 统一 Embedding 层 |
| [`emb/split.py`](../../src/betterbole/emb/split.py) | 4 种数据切分策略 |
| [`data/dataset.py`](../../src/betterbole/data/dataset.py) | 流式 Dataset 管线 |
| [`data/padding.py`](../../src/betterbole/data/padding.py) | 列 → Tensor 格式化 |

### L3 - 评估层
| 文件 | 作用 |
|------|------|
| [`evaluate/evaluator.py`](../../src/betterbole/evaluate/evaluator.py) | PointWise + TopK 评估器 |
| [`evaluate/manager.py`](../../src/betterbole/evaluate/manager.py) | `EvaluatorManager` + `DomainFilter` |
| [`evaluate/metrics.py`](../../src/betterbole/evaluate/metrics.py) | AUC/LogLoss/GAUC/HR/NDCG |

### L4 - 模型层
| 文件 | 作用 |
|------|------|
| [`models/base.py`](../../src/betterbole/models/base.py) | `BaseModel` 基类 |
| [`models/msr/base.py`](../../src/betterbole/models/msr/base.py) | `MSRModel` + `from_manager()` |
| [`models/msr/__init__.py`](../../src/betterbole/models/msr/__init__.py) | `MODEL_REGISTRY` + `build_model()` |
| [`models/backbone/shabtm.py`](../../src/betterbole/models/backbone/shabtm.py) | SharedBottom |
| [`models/backbone/mmoe.py`](../../src/betterbole/models/backbone/mmoe.py) | MMoE |
| [`models/backbone/ple.py`](../../src/betterbole/models/backbone/ple.py) | PLE |
| [`models/backbone/star.py`](../../src/betterbole/models/backbone/star.py) | STAR / StarPle |
| [`models/msr/backbone/m2m.py`](../../src/betterbole/models/msr/backbone/m2m.py) | M2M Backbone |
| [`models/msr/automtl/model.py`](../../src/betterbole/models/msr/automtl/model.py) | AutoMTL 模型 |
| [`models/msr/hamur/model.py`](../../src/betterbole/models/msr/hamur/model.py) | HAMUR 模型 |

### L5 - 训练层
| 文件 | 作用 |
|------|------|
| [`core/train/trainer.py`](../../src/betterbole/core/train/trainer.py) | `BaseTrainer` 训练循环 |
| [`core/train/early_stepper.py`](../../src/betterbole/core/train/early_stepper.py) | 早停机制 |

### L6 - 实验层
| 文件 | 作用 |
|------|------|
| [`experiment/param.py`](../../src/betterbole/experiment/param.py) | `ParamManager` 参数管理 |
| [`experiment/engine.py`](../../src/betterbole/experiment/engine.py) | `GridSearchEngine` 网格搜索 |
| [`experiment/tracker.py`](../../src/betterbole/experiment/tracker.py) | `TrainingTracker` 实验追踪 |
