# `SchemaManager` 运行逻辑梳理

这份文档面向当前仓库里的这套特征工程链路，主要说明：

1. `betterbole/emb/schema.py` 里各类 `EmbSetting` 做什么；
2. `betterbole/emb/manager.py` 的标准运行流程；
3. 什么时候应该 `fit`，什么时候应该 `transform`；
4. 它和 `reference-projects/kddsample/flatten_raw_for_scheme_manager.py` 的关系。

---

## 1. 总体定位

这套设计可以理解为两层：

- **规则层**：`src/betterbole/emb/schema.py`
  - 每个字段对应一个 `EmbSetting`
  - 负责定义：
    - 怎么统计词表 / 边界
    - 怎么做 OOV
    - 怎么把原始列变成模型可消费列
- **调度层**：`src/betterbole/emb/manager.py`
  - `SchemaManager`
  - 负责把所有 `EmbSetting` 组织成一条可执行的数据处理流水线

一句话说：

- `EmbSetting` 定义“**这个字段怎么变**”
- `SchemaManager` 定义“**整张表怎么跑**”

---

## 2. `EmbSetting` 的核心思想

所有特征规则都继承自 `EmbSetting`。

它抽象了 3 个关键阶段：

- `get_fit_exprs()`
  - 生成 Polars 表达式
  - 用于扫描训练集并收集统计量
- `parse_fit_result(result_df)`
  - 从统计结果里恢复词表 / 分桶边界 / 归一化参数
- `get_transform_expr()`
  - 返回 Polars 表达式
  - 用于把原始列真正转换成训练列

所以一个特征规则本质上就是：

```text
训练集统计 -> 规则固化 -> 全量数据变换
```

---

## 3. 当前已经支持的几类特征规则

### 3.1 `SparseEmbSetting`

适合：

- 单值离散特征
- 例如 `user_id`、`item_id`、`category_id`

特点：

- 按值频统计词表
- 支持 `min_freq`
- 支持 `padding_zero`
- 支持 `use_oov`

默认语义：

- 有效词表里的值映射到正常 id
- 词表外值映射到 `oov_idx`

这是你当前这套逻辑里真正的 **OOV 管理入口**。

### 3.2 `SparseSetEmbSetting`

适合：

- 多值集合特征
- 例如 tags、multi-hot 类目、属性集合

特点：

- 支持字符串格式和 list 格式两种输入
- 会先 explode，再按频次建词表
- transform 时逐元素映射到词表 id
- 支持 `max_len`

### 3.3 `IdSeqEmbSetting`

适合：

- 历史行为序列
- 例如 `history_items`

特点：

- **不自己建词表**
- 直接复用某个目标 `SparseEmbSetting` 的词表
- 因此训练集和测试集上的序列 item id 会自动共享：
  - 词表
  - `padding_zero`
  - `oov_idx`

这个设计非常重要，因为它保证了：

- item 主键和 item 序列的编码空间一致

### 3.4 `QuantileEmbSetting`

适合：

- 连续值离散化成 bucket
- 例如数值分位分桶

特点：

- 训练时拟合 quantile 边界
- transform 时输出 bucket id

### 3.5 `MinMaxDenseSetting`

适合：

- 连续浮点特征

特点：

- 训练时拟合 `min/max`
- transform 时做 min-max 归一化

---

## 4. `SchemaManager` 的初始化信息

`SchemaManager` 初始化时主要接收：

- `settings_list`
  - 所有字段规则
- `work_dir`
  - 工作目录
  - 用来保存：
    - `feature_meta.json`
    - 中间 parquet
    - split 结果
- `time_field`
- `label_fields`
- `domain_fields`

它还会自动从 `settings` 里识别：

- `uid_field`
- `iid_field`

也就是 source 为：

- `FeatureSource.USER_ID`
- `FeatureSource.ITEM_ID`

的字段。

---

## 5. `SchemaManager` 的三种常见使用方式

### 5.1 方式一：`prepare_data()`

这是“全自动一把梭”模式。

执行流程：

1. 如果已经存在 `feature_meta.json` 和输出 parquet，直接加载
2. 对输入 `LazyFrame` 做 checkpoint
3. 收集所有未拟合 setting 的 `fit_exprs`
4. 单次扫描训练数据得到统计量
5. 调用每个 setting 的 `parse_fit_result`
6. 再构建所有 setting 的 `transform_exprs`
7. 流式写出处理后的 parquet
8. 保存 `feature_meta.json`

适合：

- 快速试验
- 单份数据直接 fit + transform

不太适合：

- 你明确区分 train / valid / test，且只想用训练集拟合词表

### 5.2 方式二：`fit()` + `transform()`

这是更标准、也更推荐的比赛 / 研究流程。

执行方式：

#### 第一步：只在训练集上 `fit(train_raw_lf)`

效果：

- 拟合词表
- 拟合 quantile 边界
- 拟合归一化参数
- 保存到 `feature_meta.json`

#### 第二步：对 train / valid / test 分别 `transform(raw_lf)`

效果：

- 所有 split 共用同一套词表与 OOV 规则
- 测试集未见值会根据 setting 的 transform 逻辑进 `oov_idx`

这是最符合你当前诉求的用法：

- **先在训练平台用训练集统计出 OOV 规则**
- **再把 valid/test 统一映射**

### 5.3 方式三：先 `fit`，再 `split_dataset` 或 `save_as_dataset`

如果你的数据还是一个整表，还没切分：

1. 原始表上先 `fit`
2. 再 `transform`
3. 再切分并保存

如果你已经有独立 train / valid / test：

1. train_raw 上 `fit`
2. 分别 `transform`
3. `save_as_dataset(train_lf, valid_lf, test_lf)`

---

## 6. `feature_meta.json` 的角色

`SchemaManager.save_schema()` 会把所有 setting 的状态写到：

- `feature_meta.json`

里面通常包括：

- 字段名
- 类型
- embedding size
- vocab
- `oov_idx`
- `padding_zero`
- `min_freq`
- 分桶边界
- min/max

这个文件就是你整套 Scheme 的“固化快照”。

它的作用相当于：

- 你的词表文件
- 你的 OOV 规则文件
- 你的特征变换配置文件

`load_schema()` 会把这些状态重新灌回每个 setting。

所以在比赛里，最重要的不是手写 `schema.json`，而是保证：

- **训练集上拟合出的 `feature_meta.json` 可重复加载**

---

## 7. OOV 是怎么在这套逻辑里工作的

这套 manager 逻辑和 `reference-projects/kddsample` baseline 的默认 OOV 处理不同。

在这套 manager 里：

- `SparseEmbSetting`
- `SparseSetEmbSetting`
- `IdSeqEmbSetting`

都支持显式的：

- `padding_zero`
- `use_oov`
- `oov_idx`

尤其是：

- `SparseEmbSetting.parse_fit_result()`
  - 会按 `min_freq` 过滤低频值
- `_build_vocab_indices()`
  - 会构建：
    - 正常词表 id
    - `oov_idx`

所以这套逻辑天然适合：

- 用训练集统计词表
- 对 valid/test 未见值映射到 OOV

这正是你前面说“我不想木讷使用 baseline 默认 scheme”的核心原因。

---

## 8. 推荐的实际运行顺序

如果你现在的数据来源是：

- `raw/kdd26cup/sample_data.parquet`

推荐分两层跑。

### 8.1 第一步：只展平 raw

使用：

- `reference-projects/kddsample/flatten_raw_for_scheme_manager.py`

它只做：

- 把嵌套结构打平成宽表
- 不做词表拟合
- 不做 OOV
- 不做 embedding id 编码

输出：

- `flattened.parquet`
- `flatten_columns_report.json`

### 8.2 第二步：在宽表上交给 `SchemaManager`

也就是：

1. 根据 `flatten_columns_report.json` 注册各个 setting
2. 在训练集上 `fit`
3. 在 train/valid/test 上 `transform`
4. 保存处理后的 parquet 与 `feature_meta.json`

这样整条链路就是：

```text
raw nested parquet
    ↓
flatten_raw_for_scheme_manager.py
    ↓
wide parquet
    ↓
SchemaManager.fit(train)
    ↓
feature_meta.json
    ↓
SchemaManager.transform(train/valid/test)
    ↓
encoded parquet
```

---

## 9. 为什么这条链路更适合比赛

相比直接硬套 `kddsample` baseline 的 `schema.json + vocab_size` 思路，这套 `SchemaManager` 链路更灵活：

- 可以显式控制 `min_freq`
- 可以显式控制 `OOV`
- 可以区分：
  - padding
  - OOV
  - 正常值
- 可以让序列和主键共享词表
- 可以把规则固化成 `feature_meta.json`
- 可以先在训练集拟合，再对测试集做一致映射

这更符合比赛常见要求：

- **训练阶段确定词表**
- **测试阶段只允许落到已有词表或 OOV**

---

## 10. 你后面最常用的几个接口

### 10.1 `fit(train_raw_lf)`

用途：

- 用训练集拟合词表、边界、归一化参数

### 10.2 `transform(raw_lf)`

用途：

- 用固定好的规则变换任意 split

### 10.3 `split_dataset(...)`

用途：

- 使用统一的 split strategy 切 train / valid / test

### 10.4 `save_as_dataset(train_lf, valid_lf, test_lf)`

用途：

- 如果你已经自己切好了三份数据，直接落盘

### 10.5 `fields()`

用途：

- 返回当前 manager 需要保留的字段列表
- 对下游 DataLoader / parquet 读取很有帮助

---

## 11. 最后一句话总结

这套 `SchemaManager` 的最佳实践不是：

- “让 baseline 数据集类替你定义词表和 OOV”

而是：

- **先把 raw 数据展平**
- **再让 `SchemaManager` 用训练集拟合自己的词表/OOV 规则**
- **最后把 train/valid/test 全部变换到统一编码空间**

这样你才真正掌握了：

- 低频值处理
- 未见值处理
- 序列共享词表
- 持久化元数据

这些比赛里最关键的特征工程控制权。
