# 🚀 Quickstart — 5 分钟跑通 betterbole

## 安装与依赖

```bash
pip install torch numpy polars pyarrow scikit-learn psutil matplotlib
```

## 完整示例：训练一个 SharedBottom 模型

### 1️⃣ 定义特征规则 + 数据预处理

```python
import polars as pl
from betterbole.emb import SchemaManager
from betterbole.emb.schema import SparseEmbSetting, MinMaxDenseSetting
from betterbole.core.enum_type import FeatureSource

# 注册特征规则
settings = [
    SparseEmbSetting("user_id",  source=FeatureSource.USER_ID, embedding_dim=16),
    SparseEmbSetting("item_id",  source=FeatureSource.ITEM_ID, embedding_dim=16),
    MinMaxDenseSetting("rating", source=FeatureSource.INTERACTION),
]

manager = SchemaManager(
    settings_list=settings,
    work_dir="./outputs/my_exp",
    time_field="timestamp",
    label_fields="label",
)

# 读数据
train_lf = pl.scan_parquet("train.parquet")
valid_lf = pl.scan_parquet("valid.parquet")

# Fit + Transform
manager.fit(train_lf)
train_encoded = manager.transform(train_lf).select(manager.fields())
valid_encoded = manager.transform(valid_lf).select(manager.fields())
```

### 2️⃣ 创建 Dataset

```python
from betterbole.data.dataset import ParquetStreamDataset, RawParquetStreamDataset

# 方式 A：使用已编码好的 Parquet
train_ds = ParquetStreamDataset(
    parquet_path="train.parquet",
    manager=manager,
    batch_size=4096,
    shuffle=True,
)
valid_ds = ParquetStreamDataset(
    parquet_path="valid.parquet",
    manager=manager,
    batch_size=4096,
    shuffle=False,
)

# 方式 B：使用 Raw Parquet（实时 transform）
# train_ds = RawParquetStreamDataset("train_raw.parquet", manager, batch_size=4096)
```

### 3️⃣ 构建模型 + Trainer

```python
from betterbole.experiment.param import ConfigBase, ParamManager
from betterbole.core.train.trainer import BaseTrainer
from betterbole.core.train.context import TrainerDataLoaders, TrainerComponents
from betterbole.evaluate.evaluator import Evaluator
from betterbole.evaluate.manager import EvaluatorManager
from betterbole.core.train.early_stepper import EarlyStopper
from betterbole.models.msr import build_model
import torch

# 配置
cfg = ConfigBase(
    experiment_name="my_first_exp",
    dataset_name="my_data",
    max_epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 模型
model = build_model(manager, num_domains=1, model_cls="sharedbottom")

# 数据加载器
loaders = TrainerDataLoaders(
    train=train_ds,
    valid=valid_ds,
)

# 评估器
evaluator = Evaluator("auc", "logloss")
eval_manager = EvaluatorManager()
eval_manager.register("overall", evaluator)

# 组件
components = TrainerComponents(
    evaluator_manager=eval_manager,
    early_stepper=EarlyStopper(patience=5),
)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Trainer
trainer = BaseTrainer(
    model=model,
    optimizer=optimizer,
    manager=manager,
    loaders=loaders,
    components=components,
    cfg=cfg,
)
```

### 4️⃣ 训练

```python
trainer.run()
```

### 5️⃣ 使用 ParamManager 统一配置

```python
@dataclass
class MyConfig(ConfigBase):
    learning_rate: float = 1e-3
    model_type: str = "sharedbottom"
    num_domains: int = 1

pm = ParamManager(MyConfig)
cfg = pm.build(model_type="sharedbottom", learning_rate=0.001)
print(cfg)
```
