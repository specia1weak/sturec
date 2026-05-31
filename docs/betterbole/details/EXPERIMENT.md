# experiment/

这一层负责参数、脚本启动和实验记录，但它不是训练核心。

## 1. `experiment/__init__.py`

源码在 [`src/betterbole/experiment/__init__.py`](../../src/betterbole/experiment/__init__.py)。

当前实际导出的东西很少：

- `ROOT_DIR`
- `WORKSPACE`
- `ignore_future_warning()`
- `change_root_workdir()`
- `set_all()`

注意：当前源码里没有 `preset_workdir` 这个公共函数，不要把旧文档里的写法继续当真。

## 2. `ConfigBase`

源码在 [`src/betterbole/experiment/param.py`](../../src/betterbole/experiment/param.py)。

```python
@dataclass
class ConfigBase:
    experiment_name: str = "untitled"
    dataset_name: str = "unknown"
    seed: int = 2026
    device: str = "cpu"
    max_epochs: int = 100
    ckpt_dir: str = ""
    extras: dict = field(default_factory=dict)
```

### 特性

- `__getattr__` 会去 `extras` 里找未显式声明的参数。
- `__init_subclass__` 会拦截没有类型注解的字段。
- `__str__` 会输出一个比较完整的配置快照。

## 3. `ParamManager`

```python
pm = ParamManager(MyConfig)
cfg = pm.build(learning_rate=1e-3)
```

### 参数优先级

当前源码真实优先级是：

`命令行参数 > build() 传入参数 > dataclass 默认值`

这点和旧文档写反了，务必按源码记。

### `register(field_name, mapping_dict)`

用于把字符串参数映射到具体类或工厂函数。

```python
pm.register("model", {
    "mmoe": MMoEModel,
    "ple": PLEModel,
})
```

如果命令行或 `build()` 里传进来的是字符串，就会被映射成对应对象。

## 4. `seed_everything`

```python
seed_everything(2026)
```

会同步设置 Python、NumPy、PyTorch 以及 cudnn 相关随机状态。

## 5. `GridSearchEngine`

源码在 [`src/betterbole/experiment/engine.py`](../../src/betterbole/experiment/engine.py)。

```python
engine = GridSearchEngine(script_path="@examples/kuairand-1k/kuairan1k.py")
engine.run(
    param_space={
        "model": ["mmoe", "ple"],
        "seed": [2024, 2025],
        "device": ["cuda"],
    },
    available_gpus=[0, 1],
    log_dir="./logs",
)
```

### 行为

- 先做笛卡尔积展开。
- 每个组合起一个 subprocess。
- 用 `CUDA_VISIBLE_DEVICES` 轮询绑定 GPU。
- 输出重定向到单独日志文件。

### 注意

这不是分布式训练器，只是“脚本级并行启动器”。

## 6. `TrainingTracker`

源码在 [`src/betterbole/experiment/tracker.py`](../../src/betterbole/experiment/tracker.py)。

它能：

- 记录指标历史
- 保存 checkpoint
- 加载 checkpoint
- 导出向量到 `.npz`

但它和 `BaseTrainer` 是两个独立系统。当前默认训练流程不会自动调用它。

## 7. 实际建议

- 需要训练流程时优先用 `BaseTrainer`。
- 需要脚本级并行实验时用 `GridSearchEngine`。
- 需要做长期记录、导出向量、手工恢复时再考虑 `TrainingTracker`。
