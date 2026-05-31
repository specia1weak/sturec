# utils/

`betterbole.utils` 是一组独立工具。顶层包只重导出了时间桶相关能力，其他函数请直接从对应子模块导入。

## 1. `optimize.py`

源码在 [`src/betterbole/utils/optimize.py`](../../src/betterbole/utils/optimize.py)。

### `split_params_by_decay(named_params, weight_decay=0.01, no_decay_keywords=("embedding", "position_ids"))`

把参数分成两组：

- 需要 weight decay 的参数
- 不需要 weight decay 的参数

返回值可直接喂给优化器。

### `create_complex_optimizer_groups(model, decay_dict, default_decay=0.01)`

按参数名关键字分组设置不同 weight decay。

## 2. `sample.py`

源码在 [`src/betterbole/utils/sample.py`](../../src/betterbole/utils/sample.py)。

### `AbstractSampler`

负采样基类，支持：

- `uniform`
- `popularity`

### `PolarsUISampler`

面向 user-item 交互的负采样器。它会先把 `(user_id, item_id)` 物化到内存，再做采样和去重检查。

### `extract_history_dict(*lfs, user_col="user_id", item_col="item_id", merge=False)`

从一个或多个 `LazyFrame` 提取 user -> item 历史字典。

## 3. `sequential.py`

源码在 [`src/betterbole/utils/sequential.py`](../../src/betterbole/utils/sequential.py)。

### `extract_history_sequences(...)`

```python
extract_history_sequences(
    lf,
    max_seq_len=20,
    user_col="user_id",
    time_col="timestamp",
    feature_mapping={"item_id": "history_items"},
    seq_len_col="seq_len",
    label_col="label",
    positive_label=1,
)
```

这是当前最重要的序列构造工具之一，核心目标是避免标签泄露。

### `extract_seq_len(...)`

只根据现有序列列计算长度。

### `extract_history_items(...)`

`extract_history_sequences()` 的简化版，只抽 item 序列。

## 4. `time.py`

源码在 [`src/betterbole/utils/time.py`](../../src/betterbole/utils/time.py)。

### `timer`

函数级计时装饰器。

### `NamedTimer`

全局单例计时器。

常用方式：

```python
timer = NamedTimer()
with timer("data_loading"):
    ...
```

或者：

```python
@NamedTimer().collect("train_epoch")
def train_epoch():
    ...
```

### `CudaNamedTimer`

在计时前后同步 CUDA，适合 GPU 场景。

## 5. `time_bucket.py`

源码在 [`src/betterbole/utils/time_bucket.py`](../../src/betterbole/utils/time_bucket.py)。

这个模块会被 `betterbole.utils` 顶层直接导出。

### `TIME_BUCKET_BOUNDARIES`

64 个秒级边界，用于把相对时间差映射成离散桶。

### `build_padding_mask(seq_lens, max_len)`

返回 `True` 表示 padding 位置。

### `bucketize_relative_time(curr_ts, hist_ts, ...)`

把当前时间和历史时间的差值映射到 bucket id。

### `RelativeTimeEmbedding`

```python
rel_time = RelativeTimeEmbedding(embedding_dim=16)
emb = rel_time(curr_ts, hist_ts, seq_lens=seq_len)
```

它会先 bucketize，再过一个 `nn.Embedding`。

## 6. `recorder.py`

源码在 [`src/betterbole/utils/recorder.py`](../../src/betterbole/utils/recorder.py)。

### `IndividualReLURecorder`

自动 hook 模型里的 ReLU 激活并记录统计。

### `ExplicitFeatureRecorder`

手动记录任意 tensor 的统计窗口。

## 7. `visualize.py`

源码在 [`src/betterbole/utils/visualize.py`](../../src/betterbole/utils/visualize.py)。

可视化函数包括：

- `plot_bias_distributions()`
- `plot_sparsity_distributions()`
- `plot_sparsity_ecdf()`
- `plot_power2_sparsity()`

## 8. `process.py`

源码在 [`src/betterbole/utils/process.py`](../../src/betterbole/utils/process.py)。

- `set_priority()`
- `get_cpu_load_rank()`
- `set_affinity()`
- `get_affinity()`
- `get_idle_cpus()`

## 9. `task_chain.py`

源码在 [`src/betterbole/utils/task_chain.py`](../../src/betterbole/utils/task_chain.py)。

### `auto_queue()`

这是一个基于 PID 锁的串行执行工具，适合防止定时任务重叠。

## 10. 读这个模块时要记住的事

- 这里大多数函数都是独立工具，不依赖 `SchemaManager`。
- `betterbole.utils` 顶层只导出时间桶能力，不要误以为这里所有工具都被 re-export 了。
- `build_padding_mask()` 返回值语义是“padding 位置为 True”，和很多人直觉里“有效位置为 True”相反。
