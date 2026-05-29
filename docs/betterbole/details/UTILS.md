# 工具模块 (`utils/`)

> **层级**: L1 (基础层)
>
> 自包含，不依赖 betterbole 内部任何模块。可独立使用。

> betterbole 的工具箱，涵盖优化器参数管理、负采样、历史序列提取、大文件排序、性能计时、相对时间编码、激活值可视化、CPU 亲和性管理、任务编排等功能。

## 快速导航

- [优化器工具 (`optimize.py`)](#优化器工具-optimizepy)
- [负采样 (`sample.py`)](#负采样-samplepy)
- [历史序列提取 (`sequential.py`)](#历史序列提取-sequentialpy)
- [大文件排序 (`sort.py`)](#大文件排序-sortpy)
- [性能计时 (`time.py`)](#性能计时-timepy)
- [相对时间编码 (`time_bucket.py`)](#相对时间编码-time_bucketpy)
- [激活值记录器 (`recorder.py`)](#激活值记录器-recorderpy)
- [数据可视化 (`visualize.py`)](#数据可视化-visualizepy)
- [进程管理 (`process.py`)](#进程管理-processpy)
- [任务编排 (`task_chain.py`)](#任务编排-task_chainpy)

---

## 优化器工具 (`optimize.py`)

**文件**: [`utils/optimize.py`](src/betterbole/utils/optimize.py)

### `split_params_by_decay()`

```python
def split_params_by_decay(model, weight_decay=1e-5, skip_list=('bias', 'LayerNorm.bias', 'LayerNorm.weight')):
```

将模型参数分为**应应用 weight_decay 的权重**和**不应应用 weight_decay 的偏置/归一化参数**。

**返回**: `(decay_params, no_decay_params)` 两个列表，可直接传入 `optim.AdamW` 的参数组。

### `create_complex_optimizer_groups()`

```python
def create_complex_optimizer_groups(model, lr=1e-3, weight_decay=1e-5):
```

为复杂模型创建带 weight_decay 分组的优化器参数组。自动识别 embedding 层（更小的 weight_decay）与全连接层。

**核心逻辑**:
- Embedding 层 → 较小 weight_decay
- 其他权重 → 标准 weight_decay
- bias/LayerNorm → 无 weight_decay

---

## 负采样 (`sample.py`)

**文件**: [`utils/sample.py`](src/betterbole/utils/sample.py)

### `AbstractSampler`

```python
class AbstractSampler:
    def __init__(self, distribution, alpha):
```

通用负采样基类，支持两种采样策略：

| 策略 | 方法 | 说明 |
|------|------|------|
| **均匀采样 (Uniform)** | `_uni_sampling()` | 从候选池中均匀随机抽取 |
| **流行度采样 (Popularity)** | `_build_alias_table()` + `_pop_sampling()` | 基于 Alias Method 按分布概率采样 |

**核心方法**:
| 方法 | 说明 |
|------|------|
| `set_distribution(distribution)` | 设置采样分布（频次 → 概率） |
| `sampling(sample_num)` | 采样指定数量的负样本 |
| `get_used_ids()` | 获取已使用的 item ID 集合 |
| `sample_by_key_ids(key_ids, num)` | 按 key 分组采样 |

### `PolarsUISampler`

```python
class PolarsUISampler(AbstractSampler):
    def __init__(self, lf: pl.LazyFrame, item_id_col: str, user_id_col: str, time_col: str = None,
                 sample_from_set: bool = False, ...):
```

基于 Polars LazyFrame 的用户-物品交互负采样器。处理已观测交互 → 从未观测交互中采样。

**参数**:
- `lf`: 交互数据 LazyFrame
- `item_id_col` / `user_id_col`: 列名
- `time_col`: 时间列（可选，用于时序负采样）
- `sample_from_set`: 是否仅从同场景中采样

**`sample_by_key_ids` 输出格式**:
- `"flat"`: 展平为 1D Tensor
- `"listwise"`: 保持分组结构

### `extract_history_dict()`

```python
def extract_history_dict(lf: pl.LazyFrame, key_col: str, candidate_col: str,
                         time_col: str = None, is_sequential: bool = False) -> Dict:
```

将交互历史转换为 Python dict 格式：`{key: (candidates, times)}`。

**适用场景**: 需要构建 item 候选池或用户历史行为词典时使用。

---

## 历史序列提取 (`sequential.py`)

**文件**: [`utils/sequential.py`](src/betterbole/utils/sequential.py)

### `extract_history_sequences()`

```python
def extract_history_sequences(lf: pl.LazyFrame, user_col: str, item_col: str,
                              time_col: str, target_col: str, max_seq_len: int = 50,
                              min_seq_len: int = 2, sort_local: bool = True,
                              sort_global: bool = True, ...) -> pl.LazyFrame:
```

从交互日志中提取用户历史序列，**自动防止标签泄露**。

**核心机制 — ASOF JOIN 反泄露**:
```
原始交互                          历史序列
┌──────┬──────────┐              ┌──────┬─────────────────┐
│ user │  time    │              │ user │  history_items  │
├──────┼──────────┤   ASOF      ├──────┼─────────────────┤
│  A   │ 10:01    │  ────────►  │  A   │ [item_1, item_3]│
│  A   │ 10:02    │   LEFT JOIN │  A   │ [item_1, ...]   │
│  A   │ 10:03    │  time < t   │  A   │ [item_1, ...]   │
└──────┴──────────┘              └──────┴─────────────────┘
```

**关键参数**:
| 参数 | 说明 |
|------|------|
| `max_seq_len` | 最大序列长度，超出截断 |
| `min_seq_len` | 最小序列长度过滤 |
| `sort_local` | 是否在 user 内按时间排序 |
| `sort_global` | 是否全局按时间排序 |
| `agg_cols` | 额外聚合的列（如场景 ID） |

### `extract_seq_len()`

```python
def extract_seq_len(lf: pl.LazyFrame, user_col: str, time_col: str) -> pl.LazyFrame:
```

计算每个用户的交互序列长度分布。

### `extract_history_items()`

```python
def extract_history_items(lf: pl.LazyFrame, user_col: str, item_col: str,
                          time_col: str, max_seq_len: int = 50) -> pl.LazyFrame:
```

简化的历史物品序列提取，不包含标签信息。

---

## 大文件排序 (`sort.py`)

**文件**: [`utils/sort.py`](src/betterbole/utils/sort.py)

### `duckdb_sort_parquet()`

```python
def duckdb_sort_parquet(input_path: str, output_path: str,
                         sort_keys: List[str], ascending: bool = True,
                         temp_dir: str = "./_tmp_duckdb_sort"):
```

使用 **DuckDB** 对 Parquet 文件进行**外排序 (out-of-core sorting)**。支持 TB 级数据，内存友好。

**参数**:
- `input_path`: 输入 Parquet 路径
- `output_path`: 输出 Parquet 路径
- `sort_keys`: 排序列名列表
- `ascending`: 升序/降序
- `temp_dir`: 临时目录

### `sort_parquet_inplace()`

```python
def sort_parquet_inplace(file_path: str, sort_keys: List[str], ascending: bool = True, backup: bool = True):
```

原地排序 —— 直接替换原文件，可选创建备份。

---

## 性能计时 (`time.py`)

**文件**: [`utils/time.py`](src/betterbole/utils/time.py)

### `NamedTimer`

```python
class NamedTimer:
    @classmethod
    def start_record(cls, name)       # 开始计时
    @classmethod
    def stop_record(cls, name)        # 结束计时
    @classmethod
    def report(cls)                   # 打印所有计时统计
    def collect(cls, name=None)       # 装饰器：自动计时函数
    def __call__(cls, name)           # 上下文管理器：with timer("xxx"):
```

**单例全局计时器**，支持三种使用方式：

```python
# 1. 上下文管理器
timer = NamedTimer()
with timer("data_loading"):
    data = load_data()

# 2. 装饰器
@NamedTimer.collect("train_epoch")
def train_epoch():
    ...

# 3. 手动
NamedTimer.start_record("inference")
result = model.predict(data)
NamedTimer.stop_record("inference")

# 输出统计
NamedTimer.report()
# 输出:
# ====== NamedTimer Report ======
# data_loading : 3.42s (1 calls)
# train_epoch  : 45.12s (5 calls, avg 9.02s)
# ================================
```

### `CudaNamedTimer`

```python
class CudaNamedTimer(NamedTimer):
    def start_record(self, name):
        torch.cuda.synchronize()
        super().start_record(name)
    def stop_record(self, name):
        torch.cuda.synchronize()
        super().stop_record(name)
```

在启动/停止时同步 CUDA 流，确保 GPU 操作计时准确。

### `timer()` 装饰器

```python
@timer
def my_function():
    ...
```

简单的函数计时装饰器，使用 `time.time()` 测量。

---

## 相对时间编码 (`time_bucket.py`)

**文件**: [`utils/time_bucket.py`](src/betterbole/utils/time_bucket.py)

将**相对时间差**映射到离散的时间桶 (Time Bucket)，常用于序列推荐中的时间间隔特征编码。

### `TIME_BUCKET_BOUNDARIES`

```python
TIME_BUCKET_BOUNDARIES = [1, 2, 4, 8, 12, 16, 24, 32, ... , 86400]  # 64 个桶
```

预定义的 64 个桶边界（以秒为单位），从 1 秒到 1 天，覆盖短期到长期的时间间隔。

### `bucketize_relative_time()`

```python
def bucketize_relative_time(target_time: pl.Expr, reference_time: pl.Expr,
                            boundaries: Optional[Iterable[int]] = None) -> pl.Expr:
```

Polars 表达式：计算 `target_time - reference_time` 的绝对差值，映射到桶索引。

```python
df = df.with_columns(
    bucketize_relative_time(pl.col("click_time"), pl.col("view_time")).alias("time_gap_bucket")
)
```

### `build_padding_mask()`

```python
def build_padding_mask(seq_lens: torch.Tensor, max_len: int) -> torch.Tensor:
```

构建序列填充掩码。`seq_lens` 为每个序列的实际长度，`max_len` 为最大长度。

**返回**: `(B, max_len)` 的 bool 张量，`True` 表示有效位置，`False` 表示填充。

### `RelativeTimeEmbedding`

```python
class RelativeTimeEmbedding(nn.Module):
    def __init__(self, num_buckets: int = 64, embedding_dim: int = 16,
                 init_method: str = 'normal', init_std: float = 1.0,
                 boundaries: Optional[Iterable[int]] = None):
```

可学习的相对时间嵌入层。

```python
# 前向传播
time_emb = RelativeTimeEmbedding()
embeddings = time_emb(interaction, seq_time_field="click_time", ref_time_field="view_time")
```

**内部流程**:
```
seq_time      ref_time
    │             │
    └──────┬──────┘
       差值计算
           │
     bucketize_relative_time()  →  桶索引 (B, S)
           │
     nn.Embedding(num_buckets, embedding_dim)
           │
      时间嵌入 (B, S, D)
```

---

## 激活值记录器 (`recorder.py`)

**文件**: [`utils/recorder.py`](src/betterbole/utils/recorder.py)

用于调试和监控神经网络各层的激活值分布。

### `IndividualReLURecorder`

```python
class IndividualReLURecorder:
    def __init__(self, model, window_size=50):
```

通过注册 forward hook，逐层记录 ReLU 激活值的统计信息。

**核心机制**:
- 自动查找模型中所有 ReLU 层
- 注册 forward hook 收集激活值
- 维护滑动窗口统计

**方法**:
| 方法 | 说明 |
|------|------|
| `get_layer_stats()` | 返回每层的均值/标准差/稀疏度 |

### `ExplicitFeatureRecorder`

```python
class ExplicitFeatureRecorder:
    def __init__(self, window_size=50):
```

手动记录任意特征/激活值的工具。

**方法**:
| 方法 | 说明 |
|------|------|
| `record(name, tensor_data)` | 记录指定名称的特征值 |
| `get_window_stats()` | 返回窗口内的统计摘要 |

---

## 数据可视化 (`visualize.py`)

**文件**: [`utils/visualize.py`](src/betterbole/utils/visualize.py)

提供四种可视化工具，用于分析推荐系统中的特征分布、稀疏性和数据偏置。

### `plot_bias_distributions()`

```python
def plot_bias_distributions(data: pl.DataFrame, cols: List[str],
                            sort_by: str = 'bias', top_k: int = 30,
                            save_path: Optional[str] = None):
```

绘制特征的偏置分布（点击率/平均值分析）。

- 按 `sort_by` 排序展示 Top-K 特征
- 支持同时分析多个特征列

### `plot_sparsity_distributions()`

```python
def plot_sparsity_distributions(sparsity_dict: Dict, top_k: int = 50,
                                save_path: Optional[str] = None):
```

绘制特征值的稀疏度分布。`sparsity_dict` 为 `{feature_name: {value: count}}`。

### `plot_sparsity_ecdf()`

```python
def plot_sparsity_ecdf(sparsity_dict: Dict, top_k: int = 15,
                       save_path: Optional[str] = None):
```

绘制稀疏度的**经验累积分布函数 (ECDF)**。

- 子图模式：每个特征一个子图
- 包含内部 `plot_ecdf(ax, data, title)` 辅助函数

### `plot_power2_sparsity()`

```python
def plot_power2_sparsity(sparsity_dict: Dict, top_k: int = 50,
                         save_path: Optional[str] = None):
```

以 2 的幂次为间隔绘制稀疏度分布热力图，突出长尾特性。

---

## 进程管理 (`process.py`)

**文件**: [`utils/process.py`](src/betterbole/utils/process.py)

在多 GPU 或多进程训练环境中管理 CPU 资源和进程优先级。

### `set_priority()`

```python
def set_priority(ps_priority=psutil.HIGH_PRIORITY_CLASS):
```

设置当前进程的优先级。

### `get_cpu_load_rank()`

```python
def get_cpu_load_rank():
```

返回各 CPU 核心的负载排名（升序），用于负载感知的进程分配。

### `get_affinity(pid)`

```python
def get_affinity(pid):
```

获取指定 PID 的 CPU 亲和性。

### `get_idle_cpus()`

```python
def get_idle_cpus(nums, groups=1, exclude_cpus: Iterable=None):
```

返回负载最低的 `nums` 个 CPU 核心，可选择分组。用于新进程的亲和性设置。

---

## 任务编排 (`task_chain.py`)

**文件**: [`utils/task_chain.py`](src/betterbole/utils/task_chain.py)

基于 PID 锁的任务排队系统，确保同一时间只有一个实例在运行。

### `auto_queue()`

```python
def auto_queue():
```

使用 PID 文件 (`~/.auto_queue`) 实现进程级互斥：

```python
from betterbole.utils.task_chain import auto_queue

# 确保只有一个实例执行
with auto_queue():
    run_training_pipeline()
```

**工作原理**:
1. 检查锁文件中记录的 PID 是否存活
2. 如果存活，则等待并重试
3. 如果已死，则获取锁并写入当前 PID
4. 退出时自动清理锁文件

**使用场景**: 定时任务 / crontab 中的训练流水线，防止重叠执行导致 OOM。

---

**下一步**: 查看 [`API_REFERENCE.md`](docs/betterbole/API_REFERENCE.md) 获取完整的类/函数索引
