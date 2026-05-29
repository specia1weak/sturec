# 🧪 实验模块 (experiment/)

> **层级**: L6 (实验层) — 最高层
>
> 依赖 L1~L5 所有模块。作为用户入口，协调参数管理、网格搜索、实验追踪。

## 参数管理 — [`param.py`](../../src/betterbole/experiment/param.py)

### `ConfigBase`

所有配置的基类 DataClass：

```python
@dataclass
class ConfigBase:
    experiment_name: str = "untitled"
    dataset_name: str = "unknown"
    seed: int = 2026
    device: str = "cpu"
    max_epochs: int = 100
    ckpt_dir: str = ""
    extras: dict = field(default_factory=dict)  # 兜底字典
```

**特性**：
- `extras` 兜底：任何未声明的字段自动存入 `cfg.extras`，可通过 `cfg.xxx` 访问
- `__str__` 美观打印所有配置项
- 子类必须为所有字段添加**类型注解**

### `ParamManager`

多来源参数合并（代码 > 命令行 > 默认值）：

```python
manager = ParamManager(MyConfig)

# 注册字符串→实例的映射
manager.register("model", {
    "sharedbottom": SharedBottomModel,
    "ple": PLEModel,
})

# 合并参数
cfg = manager.build(
    model="ple",            # 代码传入
    learning_rate=0.001,    # 自动进入 extras
)
# 也支持 python script.py --model ple --learning_rate 0.001
```

**参数优先级**：命令行参数 > `build()` 代码参数 > DataClass 默认值

### `seed_everything(seed)`

一键设置所有随机种子。

---

## 网格搜索 — [`engine.py`](../../src/betterbole/experiment/engine.py)

### `GridSearchEngine`

多 GPU 并行网格搜索：

```python
engine = GridSearchEngine(script_path="run.py")

search_space = {
    "model": ["star", "ple"],
    "seed": [2024, 2025, 2026],
    "device": ["cuda"]
}

# 自动分配 GPU，并行执行
engine.run(
    param_space=search_space,
    available_gpus=[0, 1],
    log_dir="./logs",
)
```

- 自动生成笛卡尔积组合
- 轮询分配 GPU
- 每个实验独立日志文件

---

## 实验跟踪 — [`tracker.py`](../../src/betterbole/experiment/tracker.py)

### `TrainingTracker`

简单但完整的实验跟踪器：

```python
tracker = TrainingTracker(workdir="./outputs")

# 记录指标
tracker.log_metrics({'train_loss': 0.5, 'lr': 0.001})

# 保存断点
tracker.save_checkpoint(model, optimizer, is_best=True, metric_val=0.73)

# 加载恢复
tracker.load_checkpoint(model, optimizer, ckpt_path="./checkpoints/best_model.pth")

# 导出中间向量
tracker.save_vector("user_emb", user_embedding)
tracker.export_vectors()
```

**Checkpoint 内容**：
```python
{
    'global_step': int,
    'epoch': int,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'metric': float,
}
```
