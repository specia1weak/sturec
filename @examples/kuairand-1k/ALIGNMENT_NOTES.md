# KuaiRand-1K 对齐备忘

这份文档记录 `@examples/kuairand-1k/kuairan1k.py` 与原版 Crocodile KuaiRand-1K 实验在数据和训练侧的已确认对齐点、已排除项、以及后续重点排查方向。

## 当前确认

- 当前实验目标是对齐原版 `test` 表现，所以评估集直接使用 `test_path` 是刻意选择，不是误用。
- 只训练 `1 epoch` 是当前数据集上的正确做法，不需要为了复现原版配置强行改成多轮。
- `CrocodileV1` 已经补齐了内部 `disentangled loss` 与模型内 `regularization_loss()`，外部 `weight decay` 可独立开关。
- `ps_valid` 必须显式写 `drop_last=False`，否则测试集最后一个不满 batch 的尾部样本会被丢掉。
- 训练集 `ParquetStreamDataset(..., drop_last=False)` 不会因为一轮 epoch 而少样本；数据会被扫完整个 parquet，尾部 residual 也会在 `flush()` 时吐出。

## 已确认的数据事实

### 1. 原始 tag 列是逗号分隔

原始 `video_features_basic_1k.csv` 中，`tag` 的真实格式是：

- 单值：`39`
- 多值：`"20,43"`

也就是说，原始 KuaiRand-1K 数据的 `tag` 不是 `^` 分隔。

### 2. 原版 Crocodile 预处理会把 tag 从 `,` 改成 `^`

原版脚本：

- `reference-projects/Crocodile-main/data/Kuairand1K/0_preprocess.py`

其中有：

```python
def process_tag(t):
    if isinstance(t, float):
        return ''
    else:
        return t.replace(',', '^')

data['tag'] = data['tag'].apply(process_tag)
```

因此：

- 原始 CSV 使用 `,`
- 原版中间产物 `train.gz/valid.gz/test.gz` 使用 `^`
- 原版 `dataset_config.yaml` 中的 `splitter: ^` 与其预处理结果一致，并不矛盾

### 3. 删除 `onehot_feat0` / `visible_status` 不是主要矛盾

这两个字段已经尝试去掉，结果变化不大，说明它们不是当前性能差距的主因。

## OOV / 词表机制结论

目前没有发现足够大的 OOV 机制差异。

原版 FuxiCTR：

- `__PAD__ = 0`
- `__OOV__ = vocab_size()`
- 只在训练集上建 vocab
- valid/test 共用训练集 vocab 编码

当前 BetterBole 实现：

- `padding_zero=True` 时 padding index 为 `0`
- `use_oov=True` 时 OOV 落到最后一个 index
- 同样只在训练集上 `fit()`，随后 transform valid/test

结论：

- `OOV` 机制不是当前最优先的怀疑点

## 当前最值得怀疑的点

### 1. 训练集 shuffle 方式

原版 `H5DataLoader(shuffle=True)` 更接近全局随机打乱。

当前 `ParquetStreamDataset` 的训练流程是：

1. 按 parquet 顺序扫描
2. 累积到 `shuffle_buffer_size`
3. 在 buffer 内随机 permutation
4. 分批吐出
5. residual 留到下一轮 drain / flush

这意味着当前实现是“局部窗口 shuffle”，不是严格的全局 shuffle。

如果 parquet 本身还是按 `time_ms` 排序落盘，那么 epoch 内的样本顺序很可能和原版差异明显。

### 2. 数据划分与中间预处理

虽然 train/valid/test 的日期切分已经和原版脚本一致：

- `train <= 20220506`
- `valid == 20220507`
- `test == 20220508`

但仍需继续检查：

- 原版中间 `train.gz` 是否对字段做了额外转换
- 当前 parquet 落盘顺序是否保留了过强的时间局部性
- `tab` 的 domain 编码语义是否和原版完全一致

## DataLoader 结论

### 评估集

推荐固定写法：

```python
ps_valid = ParquetStreamDataset(
    test_path,
    manager,
    batch_size=4096 * 2,
    shuffle=False,
    drop_last=False,
)
```

原因：

- `drop_last=True` 会丢最后一个不满 batch 的测试尾部样本

### 训练集

当前写法：

```python
ps_dataset = ParquetStreamDataset(
    train_path,
    manager,
    batch_size=cfg.batch_size,
    shuffle=True,
    shuffle_buffer_size=cfg.shuffle_buffer_size,
    drop_last=False,
)
```

性质：

- 不会漏样本
- 但只做局部 shuffle，不是全局 shuffle

## 后续建议

下一步优先排查：

1. 全局 shuffle 是否显著影响测试指标
2. parquet 落盘顺序与训练读取顺序之间的耦合
3. `tab` / prior / domain id 的编码语义是否完全等价

如果要继续逼近原版，最值得做的不是继续纠结 OOV，而是先实现一个“接近全局随机顺序”的训练集读取方案，再看指标是否明显变化。
