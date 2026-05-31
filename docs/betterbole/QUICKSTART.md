# Quickstart

这份 quickstart 只使用当前源码里真实存在的 API，目标是跑通一条最小链路：

`raw parquet -> SchemaManager -> encoded parquet -> ParquetStreamDataset -> build_model -> BaseTrainer`

如果你只想先理解调用顺序，这一页够用。更细的模块解释分别在 [`details/EMB.md`](details/EMB.md)、[`details/DATA.md`](details/DATA.md)、[`details/CORE.md`](details/CORE.md)。

## 依赖

```bash
pip install torch numpy polars pyarrow scikit-learn psutil matplotlib
```

## 1. 定义配置

`ParamManager` 的优先级是：

`命令行参数 > build(...) 传入参数 > dataclass 默认值`

```python
from dataclasses import dataclass
import torch

from betterbole.experiment.param import ConfigBase, ParamManager


@dataclass
class DemoConfig(ConfigBase):
    experiment_name: str = "demo_mmoe"
    dataset_name: str = "demo"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs: int = 3

    batch_size: int = 1024
    id_emb_dim: int = 16
    side_emb_dim: int = 8
    model: str = "mmoe"


cfg = ParamManager(DemoConfig).build()
print(cfg)
```

## 2. 定义特征规则

这里的重点不是“把所有列都做成 embedding”，而是明确哪些列要进入模型、哪些列只作为上下文保留。

```python
import polars as pl

from betterbole.core.enum_type import FeatureSource
from betterbole.emb import SchemaManager
from betterbole.emb.schema import (
    MinMaxDenseSetting,
    MultiSparseSetting,
    SparseEmbSetting,
)


settings = [
    SparseEmbSetting("user_id", FeatureSource.USER_ID, embedding_dim=cfg.id_emb_dim, min_freq=1),
    SparseEmbSetting("item_id", FeatureSource.ITEM_ID, embedding_dim=cfg.id_emb_dim, min_freq=1),
    SparseEmbSetting("city_id", FeatureSource.USER, embedding_dim=cfg.side_emb_dim, min_freq=1),
    SparseEmbSetting("category_id", FeatureSource.ITEM, embedding_dim=cfg.side_emb_dim, min_freq=1),
    MinMaxDenseSetting("price", FeatureSource.INTERACTION),
    MultiSparseSetting(
        "tags",
        FeatureSource.ITEM,
        embedding_dim=cfg.side_emb_dim,
        max_tag_len=5,
        is_string_format=True,
        separator=",",
    ),
]

manager = SchemaManager(
    settings_list=settings,
    work_dir="./workspace/demo",
    time_field="timestamp",
    label_fields="label",
    domain_fields="domain_id",
)
```

说明：

- `label_fields` 和 `domain_fields` 会被保留到输出数据中，即使它们不是 `EmbSetting`。
- 如果 `domain_id` 只用于场景路由，不参与 embedding，可以不注册成 `SparseEmbSetting`。
- 如果同一个 `work_dir` 里已经有 `feature_meta.json`，`manager.fit()` 会优先加载旧 schema。

## 3. 切分、拟合、转换并落盘

推荐的主流程是：

1. `split_dataset()`
2. `fit(train_raw)`
3. `transform(...)`
4. `save_as_dataset(...)`

```python
raw_lf = pl.scan_parquet("demo_raw.parquet")

train_raw, valid_raw, test_raw = manager.split_dataset(
    raw_lf,
    strategy="random_ratio",
    train_ratio=0.8,
    valid_ratio=0.1,
    group_by="user_id",
)

manager.fit(train_raw)

train_lf = manager.transform(train_raw)
valid_lf = manager.transform(valid_raw)
test_lf = manager.transform(test_raw)

train_path, valid_path, test_path = manager.save_as_dataset(
    train_lf,
    valid_lf,
    test_lf,
)
```

如果你已经手工切好了 train/valid/test，直接把三个 `LazyFrame` 传给 `save_as_dataset()` 即可。

## 4. 构建流式数据集

```python
from betterbole.data.dataset import ParquetStreamDataset


train_ds = ParquetStreamDataset(
    train_path,
    manager,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
)

valid_ds = ParquetStreamDataset(
    valid_path,
    manager,
    batch_size=cfg.batch_size * 2,
    shuffle=False,
    drop_last=False,
)
```

如果你不想先把编码结果落盘，也可以直接用 `RawParquetStreamDataset` 在读取时做 `transform`：

```python
from betterbole.data.dataset import RawParquetStreamDataset


train_ds = RawParquetStreamDataset(
    parquet_path="demo_raw.parquet",
    manager=manager,
    batch_size=cfg.batch_size,
    shuffle=True,
    raw_filter_expr=pl.col("split") == "train",
)
```

## 5. 构建模型、评估器和 Trainer

```python
import torch

from betterbole.core.train import (
    BaseTrainer,
    EarlyStopper,
    TrainerComponents,
    TrainerDataLoaders,
)
from betterbole.evaluate.evaluator import Evaluator
from betterbole.evaluate.manager import EvaluatorManager
from betterbole.models.msr import build_model
from betterbole.utils.optimize import split_params_by_decay


num_domains = 3
model = build_model(manager, num_domains=num_domains, model_cls=cfg.model)

evaluator_manager = EvaluatorManager(log_path="./workspace/demo/metrics.log", title=cfg.experiment_name)
evaluator_manager.register("overall", Evaluator("auc"))

optimizer = torch.optim.Adam(
    split_params_by_decay(model.named_parameters(), weight_decay=1e-5),
    lr=1e-3,
)

trainer = BaseTrainer(
    model=model,
    optimizer=optimizer,
    manager=manager,
    loaders=TrainerDataLoaders(train=train_ds, valid=valid_ds),
    components=TrainerComponents(
        evaluator_manager=evaluator_manager,
        early_stepper=EarlyStopper(patience=3),
    ),
    cfg=cfg,
)
```

## 6. 训练

```python
trainer.run()
```

`BaseTrainer` 的默认行为是：

1. `train_epoch()`
2. `evaluate_epoch()`
3. 如果配置了 `EarlyStopper`，则根据验证指标判断是否保存最佳 checkpoint / 是否提前停止

## 7. 两个容易踩坑的点

### `predict()` 的输出会直接进入 evaluator

`BaseTrainer.evaluate_epoch()` 里调用的是：

```python
scores = self.model.predict(batch_interaction)
self.evaluator.collect(uids, labels, batch_interaction, batch_preds_1d=scores)
```

这意味着：

- 只算 `auc` 时，raw logits 通常也能用。
- 想算 `logloss` 时，最好让 `predict()` 返回概率，或者覆写 `predict_step()`。

### `ParquetStreamDataset` 已经分好 batch 了

如果你自己额外包一层 `torch.utils.data.DataLoader`，要这样写：

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    train_ds,
    batch_size=None,
    num_workers=0,
)
```

否则你会把“已经是一个 batch 的 `Interaction`”再套一层 batch，形状会错。
