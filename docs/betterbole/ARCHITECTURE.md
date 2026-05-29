# 🏛️ 架构设计 (ARCHITECTURE)

> betterbole 按**分层架构**组织，共 6 层。下层提供基础抽象，上层编排调度。

## 分层架构图

```
┌─────────────────────────────────────────────────────────────┐
│  L6: 实验层 (experiment/)                                    │
│  GridSearchEngine │ TrainingTracker │ ParamManager           │
│  🎯 职责: 实验管理、超参搜索、Checkpoint 追踪                   │
├─────────────────────────────────────────────────────────────┤
│  L5: 训练层 (core/train/)                                    │
│  BaseTrainer │ TrainContext │ Hooks │ EarlyStopper           │
│  🎯 职责: 训练循环编排、Hook 扩展、早停                         │
├─────────────────────────────────────────────────────────────┤
│  L4: 模型层 (models/)                                        │
│  MSRModel → Backbone(MMoE/PLE/STAR/M2M/PPNet/EPNet)         │
│          → DomainTowerHead → AutoMTL/HAMUR/HierRec           │
│  🎯 职责: 14 种多场景推荐模型，可插拔 Backbone                  │
├─────────────────────────────────────────────────────────────┤
│  L3: 评估层 (evaluate/)                                      │
│  EvaluatorManager → (PointWise|TopK)Evaluator → Metrics      │
│  🎯 职责: 多评估器编排、Domain 分流、指标计算                    │
├─────────────────────────────────────────────────────────────┤
│  L2: 数据与嵌入层 (emb/ + data/)                              │
│  SchemaManager → OmniEmbLayer → DataScanner → TensorFormatter│
│  🎯 职责: 特征工程、流式加载、统一 Embedding                     │
├─────────────────────────────────────────────────────────────┤
│  L1: 基础层 (core/ + utils/ + datasets/)                     │
│  Interaction │ FeatureSource │ 采样/计时/排序/可视化/数据集     │
│  🎯 职责: 数据容器、枚举、工具函数、内置数据集                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 调用链路 (从入口到底层)

```
用户入口
    │
    ▼
L6 ParamManager.build(...)  ──→ 解析配置 (code > CLI > default)
    │
    ▼
L2 SchemaManager(settings)   ──→ 注册 EmbSetting
L2 SchemaManager.fit(lf)     ──→ 扫描训练集，构建词表/参数
L2 SchemaManager.transform() ──→ 转换全量数据
    │
    ▼
L2 DataScanner + DataTransformer + ShuffleBuffer + TensorFormatter
    │
    ▼  Interaction (Tensor dict)
    │
L4 OmniEmbLayer.whole(interaction)  ──→ 统一特征注入
L4 Backbone.forward(x, domain_ids)   ──→ 骨干网络
L4 DomainTowerHead.forward(x, domain_ids) ──→ 每 Domain 独立输出
    │
    ▼  logits
    │
L3 EvaluatorManager.collect(...)     ──→ 收集预测结果
L3 Evaluator.summary()              ──→ 计算指标
    │
    ▼
L5 BaseTrainer.train_epoch()        ──→ 训练
L5 EarlyStopper.step(summary_dict)  ──→ 判断早停
L5 BaseTrainer.save_checkpoint()    ──→ 保存模型
    │
    ▼
L6 TrainingTracker.log_metrics()    ──→ 记录实验
L6 GridSearchEngine.run()           ──→ 多 GPU 并行搜索
```

## 整体数据流

```
原始 Parquet/CSV
     │
     ▼
┌──────────────────┐
│  SchemaManager   │  ← EmbSetting 定义每个特征怎么处理
│  ┌──────────────┐│
│  │ fit(训练集)   ││  → 扫描统计量，构建词表/边界/归一化参数
│  │ transform( ) ││  → 用固化规则变换任意 Split
│  └──────────────┘│
│  → feature_meta  │  → 固化状态到 JSON
└────────┬─────────┘
         │ encoded parquet
         ▼
┌──────────────────┐
│  DataScanner     │  → 流式读取 Parquet，支持多 Worker 分片
│  DataTransformer │  → 实时 transform (RawParquetStreamDataset 模式)
│  ShuffleBuffer   │  → 大容量 Shuffle Buffer (200w+行)
│  TensorFormatter │  → 每列根据 setting 决定 IntFormatter/DenseFormatter/PaddedFormatter
└────────┬─────────┘
         │ Interaction (dict of Tensor)
         ▼
┌──────────────────┐
│  OmniEmbLayer    │  → 接收 Interaction，通过 EmbView 路由到指定字段
│  │               │  → 每个 Setting.compute_tensor() 取出对应 Tensor
│  │               │    并通过 nn.Embedding 查表
│  ▼               │
│  Concatenated    │  → 按 source/name/none 拼接 Embedding
└────────┬─────────┘
         │ embedding tensor
         ▼
┌──────────────────┐
│   Backbone       │  → SharedBottom / MMoE / PLE / STAR
│   (Tower)        │  → 每个 Domain 独立 Tower
└────────┬─────────┘
         │ logits
         ▼
┌──────────────────┐
│  Trainer         │  → train_epoch → evaluate_epoch → early_stop
│  Evaluator       │  → AUC / LogLoss / GAUC / HR@K / NDCG@K
└──────────────────┘
```

## 核心设计模式

### 1. 声明式特征工程 (`EmbSetting`)

每个字段的特征处理规则被封装为一个 [`EmbSetting`](../../src/betterbole/emb/schema/base.py) 子类：

```
get_fit_exprs()        → 扫描训练集，收集统计信息
parse_fit_result()     → 从统计结果构建词表/参数
get_transform_expr()   → 用词表/参数转换原始数据
get_formatters()       → 定义该列如何转为 PyTorch Tensor
compute_tensor()       → 前向传播时，从 Interaction 中取 Tensor
```

### 2. 调度编排 (`SchemaManager`)

[`SchemaManager`](../../src/betterbole/emb/manager.py) 是特征工程的总调度：

- `prepare_data()` — 全自动流程：fit + transform 一把梭
- `fit()` — 仅在训练集上拟合
- `transform()` — 对任意 Split 应用固化规则
- `split_dataset()` — 4 种切分策略
- `save_schema()` / `load_schema()` — 固化 / 恢复

### 3. 统一 Embedding 层 (`OmniEmbLayer`)

[`OmniEmbLayer`](../../src/betterbole/emb/emblayer.py) 替代了传统的 SideEmb / UserSideEmb / ItemSideEmb 等子类：

- 内部纳管所有 `EmbSetting`
- `forward()` 通过 `target_sources` / `include_fields` / `exclude_fields` 动态路由
- 预置了 `user_all` / `item_all` / `inter` / `whole` / `domain` 等 `EmbView`

### 4. 多场景多任务 (MSR)

所有 MSR 模型继承自 [`MSRModel`](../../src/betterbole/models/msr/base.py)，接收：

- `interaction`：宽表交互数据
- `domain_ids`：域标识

内部结构：
```
OmniEmbLayer → Backbone(共享) → Domain Towers(独立)
```

### 5. 流式数据管线

```
DataScanner (Polars LazyFrame / PyArrow)
    → DataTransformer (SchemaManager.transform)
    → ShuffleBuffer (大容量 buffer)
    → TensorFormatter (列→Tensor)
    → Interaction
```

支持 `torch.utils.data.DataLoader` 多进程，且 Worker 间自动按文件/行号分片。

## 模块依赖关系

```
experiment/  ← 顶层入口
    │
models/      ← 需要 emb/ + core/interaction
    │
core/train/  ← 需要 models/ + evaluate/ + data/
    │
data/        ← 需要 emb/ (SchemaManager)
    │
emb/         ← 自包含 + core/enum_type
    │
evaluate/    ← 自包含 (sklearn)
    │
utils/       ← 独立工具函数
```
