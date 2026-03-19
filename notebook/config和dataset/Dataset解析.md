# Dataset 类文档 (RecBole 数据集基类)
`Dataset` 类（继承自 `torch.utils.data.Dataset`）用于将原始数据集存储在内存中。

它提供了许多用于数据预处理的实用函数，例如 **k-core 数据过滤（k-core filtering）** 和 **缺失值插补（missing value imputation）**。在 `Dataset` 内部，特征是以 `pandas.DataFrame` 的形式存储的。通用推荐模型（General Models）和上下文感知推荐模型（Context-aware Models）都可以使用此类。

通过调用 `build` 方法，它可以根据评估设置（`EvalSetting`）将数据集处理并划分为对应的数据加载器（DataLoaders）。
### 🏷️ 属性 (Attributes)

#### 1. 基本信息
*   **`dataset_name`** (`str`): 此数据集的名称。
*   **`dataset_path`** (`str`): 此数据集所在的本地文件绝对/相对路径。
*   **`feat_name_list`** (`list`): 一个列表，包含了所有特征的名称（`str` 类型），其中也包括通过后缀额外加载的附加特征。

#### 2. 特征元数据 (Metadata)
*   **`field2type`** (`dict`): `Dict[str, FeatureType]`，将特征字段名（`str`）映射到其数据类型（即 `FeatureType` 枚举类，如 `TOKEN`, `FLOAT`, `FLOAT_SEQ` 等）。
*   **`field2source`** (`dict`): `Dict[str, FeatureSource]`，将特征字段名（`str`）映射到其数据来源（即 `FeatureSource` 枚举类，如 `USER`, `ITEM`, `INTERACTION`）。
    *   *特别注意*：如果特征是通过参数 `additional_feat_suffix` 加载的附加特征，则其来源类型为字符串格式（即其本地文件的后缀名，与 `additional_feat_suffix` 中配置的一致）。
*   **`field2seqlen`** (`dict`): `Dict[str, int]`，将特征字段名（`str`）映射到该特征的序列长度（`int`）。
    *   对于 **序列特征 (Sequence features)**，其长度既可以是配置文件中手动设置的值，也可以是自动统计出的该特征在数据中的最大序列长度。
    *   对于 **离散 (token)** 和 **连续型 (float) 特征**，它们的长度固定为 `1`。

#### 3. ID 映射与词表 (Token Remapping)
*   **`field2id_token`** (`dict`): `field2id_token[field] 装的是 id2name的映射表`
*   **`field2token_id`** (`dict`): `field2token_id[field] 装的是 name2id的映射表`
#### 4. 关键字段映射 (Key Fields)
*   **`uid_field`** (`str` 或 `None`): UID的实际字段名 == `config['USER_ID_FIELD']`。
*   **`iid_field`** (`str` 或 `None`): IID的实际字段名 == `config['ITEM_ID_FIELD']`。
*   **`label_field`** (`str` 或 `None`): 标签的实际字段名。等同于 `config['LABEL_FIELD']`。
*   **`time_field`** (`str` 或 `None`): 时间戳的实际字段名。等同于 `config['TIME_FIELD']`。
#### 5. 核心特征表 (Feature Tables)
*   **`inter_feat`** (`Interaction`): 内部数据结构，用于存储真实的**交互特征数据**。它是从 `.inter` 文件中加载生成的。
*   **`user_feat`** (`Interaction` 或 `None`): 内部数据结构，用于存储**用户静态特征数据**。如果本地存在 `.user` 文件，则从中加载，否则为 `None`。
*   **`item_feat`** (`Interaction` 或 `None`): 内部数据结构，用于存储**物品静态特征数据**。如果本地存在 `.item` 文件，则从中加载，否则为 `None`。





### Dataset 类外部可调用方法
#### 1. 核心构建、划分与保存 (Core Processing)
这部分方法主要用于数据集的最终成型、划分（训练/验证/测试）以及序列化保存。
*   **`build(self)`**
    *   **功能**：**[核心方法]** 根据配置文件中的评估设置（如打乱、按时间排序、留一法等）处理数据集，并生成数据加载器所需的子数据集。
    *   **返回值**：`list`，包含划分好的多个 `Dataset` 对象（例如：`[train_dataset, valid_dataset, test_dataset]`）。
*   **`split_by_ratio(self, ratios, group_by=None)`**
    *   **功能**：按比例划分交互记录（如 `[0.8, 0.1, 0.1]`）。
    *   **参数**：
        *   `ratios` (list): 切分比例列表。
        *   `group_by` (str): 按某个字段（如 `user_id`）分组后再切分。
*   **`leave_one_out(self, group_by, leave_one_mode)`**
    *   **功能**：使用“留一法（Leave-One-Out）”划分交互记录。
    *   **参数**：
        *   `leave_one_mode` (str): `'valid_and_test'`, `'valid_only'` 或 `'test_only'`。
*   **`time_based_split(self, ratios, group_by)`**
    *   **功能**：结合全局时间与留一法限制，基于时间戳策略进行划分。
*   **`save(self)`**
    *   **功能**：将当前已经过滤、重映射好的 `Dataset` 对象序列化保存（pickle）到配置文件指定的 `checkpoint_dir` 中，后缀通常为 `.pth`。
*   **`copy(self, new_inter_feat)`**
    *   **功能**：以当前的 Dataset 为模板，替换其交互特征（`inter_feat`），返回一个克隆的 Dataset 对象。（RecBole 内部划分数据时大量使用此方法，节省内存）。
*   **`shuffle(self)`** / **`sort(self, by, ascending=True)`**
    *   **功能**：原地打乱 (shuffle) 或按指定字段排序 (sort) 交互记录。

---

#### 2. 动态统计信息与属性 (Dynamic Properties)

这些属性使用了 `@property` 装饰器，可以像变量一样直接访问，用于获取数据集的统计宏观信息。

*   **`user_num` / `item_num` / `inter_num`** (`int`)
    *   **功能**：分别获取 数据集中用户的总数（包含 padding 的 0）、物品的总数、以及交互记录的总行数。
*   **`avg_actions_of_users` / `avg_actions_of_items`** (`float`)
    *   **功能**：分别获取 平均每个用户的交互次数、平均每个物品被交互的次数。
*   **`sparsity`** (`float`)
    *   **功能**：获取数据集的稀疏度。计算公式为：$1 - \frac{inter\_num}{user\_num \times item\_num}$。
*   **动态特征归类获取** (`list`):
    *   `float_like_fields`: 获取所有连续型浮点特征的字段名列表。
    *   `token_like_fields`: 获取所有离散型 Token 特征的字段名列表。
    *   `seq_fields`: 获取所有序列特征（如 `TOKEN_SEQ`）的字段名列表。
    *   `non_seq_fields`: 获取所有非序列特征的字段名列表。

---

#### 3. ID 映射与词表查询 (ID & Token Mapping)

用于在**外部真实字符串 ID** 和 **内部模型张量 ID** 之间进行转换。

*   **`token2id(self, field, tokens)`**
    *   **功能**：将外部的真实 token（字符串/列表）映射为内部模型使用的数字 ID。
    *   **示例**：`dataset.token2id('item_id', 'B00000JYWQ')` $\rightarrow$ `1024`。
*   **`id2token(self, field, ids)`**
    *   **功能**：将内部数字 ID 映射回外部原始真实的 token 字符串（做推荐结果展示时极其重要）。
    *   **示例**：`dataset.id2token('item_id',[1024, 1025])` $\rightarrow$ `['B00000JYWQ', 'B00005N7P0']`。
*   **`num(self, field)`**
    *   **功能**：**[极其常用]** 获取某个特征字段的“词表大小（Vocabulary Size）”。对于 token 特征，返回不同 token 的数量（重映射后）；对于 float 特征，返回 1。常用于初始化 `nn.Embedding(dataset.num(field), dim)`。
*   **`counter(self, field)`**
    *   **功能**：返回一个 `collections.Counter`，包含某个特征字段在交互记录中出现的频率字典。

---

#### 4. 获取与拼接特征数据 (Feature Operations)

*   **`get_user_feature(self)`** / **`get_item_feature(self)`**
    *   **功能**：获取用户/物品的静态属性特征表，返回 `Interaction` 对象。如果数据集中没有静态特征，则会自动生成一张只有 ID 列的表。
*   **`join(self, df)`**
    *   **功能**：类似于 SQL 的 `LEFT JOIN`。给定一个交互记录 `df`，自动根据里面的 `user_id` 和 `item_id`，将用户的静态属性和物品的静态属性拼接到这个 `df` 中并返回。
*   **`field2feats(self, field)`**
    *   **功能**：返回一个列表，包含所有出现过该 `field` 的数据表（可能是 `inter_feat`，`user_feat`，或 `item_feat`）。
*   **`fields(self, ftype=None, source=None)`**
    *   **功能**：根据给定的类型（如 `FeatureType.TOKEN`）和来源（如 `FeatureSource.USER`）筛选并返回字段名列表。

---

#### 5. 高阶矩阵与图网络构建 (Matrix & Graph Construction)

这部分方法是为 **序列推荐（如 DIN）**、**图神经网络（如 LightGCN）** 乃至传统的**协同过滤**提供直接的数据结构支持。

*   **`inter_matrix(self, form="coo", value_field=None)`**
    *   **功能**：生成 用户-物品 (User-Item) 交互的全局稀疏矩阵。
    *   **参数**：`form` 支持 `'coo'` (默认) 和 `'csr'` 格式。这是各种 GCN 模型计算拉普拉斯矩阵的根基。
*   **`history_item_matrix(self, value_field=None, max_history_len=None)`**
    *   **功能**：**[DIN/SASRec 核心依赖]** 获取所有用户的历史交互密集矩阵（Dense Matrix）。
    *   **返回值** (Tuple):
        1.  `history_matrix`: 形状为 `[user_num, max_history_len]` 的张量，存储用户点过的 `item_id` 序列。
        2.  `history_value`: 如果指定了 `value_field`（比如评分），这里存评分序列，否则为 0。
        3.  `history_len`: 一维张量，记录每个用户真实的序列长度（方便 Attention 算 Mask）。
*   **`history_user_matrix(self, value_field=None, max_history_len=None)`**
    *   **功能**：与上方类似，但是反过来的，获取**每个物品被哪些用户交互过**的历史序列。
*   **`get_preload_weight(self, field)`**
    *   **功能**：如果预先加载了外部的预训练 Embedding（比如给 Item 加载了 Word2Vec 或 Bert 的向量），通过此方法取出对齐后的权重矩阵，用于 `nn.Embedding.from_pretrained()`。

---

#### 6. Python 魔法方法 (Magic Methods)

让 `Dataset` 表现得像一个标准的 Python 序列/PyTorch Dataset。

*   **`__len__(self)`**
    *   **用法**：`len(dataset)`
    *   **功能**：返回数据集中包含的交互记录总数（即 `len(self.inter_feat)`）。
*   **`__getitem__(self, index, join=True)`**
    *   **用法**：`dataset[0:10]`
    *   **功能**：获取指定切片/索引的数据。如果 `join=True`（默认），它不仅返回交互记录，还会自动调用 `join()` 方法，把用户和物品的上下文特征全部拼装好返回。
*   **`__str__(self)`** / **`__repr__(self)`**
    *   **用法**：`print(dataset)`
    *   **功能**：在控制台打印数据集宏观信息的彩色美化字符串（用户数、稀疏度、特征列等信息）。 







```
    # -*- coding: utf-8 -*-
# @Time   : 2020/8/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.utils.enum_type
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``GENERAL``: General Recommendation
    - ``SEQUENTIAL``: Sequential Recommendation
    - ``CONTEXT``: Context-aware Recommendation
    - ``KNOWLEDGE``: Knowledge-based Recommendation
    """

    GENERAL = 1
    SEQUENTIAL = 2
    CONTEXT = 3
    KNOWLEDGE = 4
    TRADITIONAL = 5
    DECISIONTREE = 6


class KGDataLoaderState(Enum):
    """States for Knowledge-based DataLoader.

    - ``RSKG``: Return both knowledge graph information and user-item interaction information.
    - ``RS``: Only return the user-item interaction.
    - ``KG``: Only return the triplets with negative examples in a knowledge graph.
    """

    RSKG = 1
    RS = 2
    KG = 3


class EvaluatorType(Enum):
    """Type for evaluation metrics.

    - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
    - ``VALUE``: Value-based metrics like AUC, etc.
    """

    RANKING = 1
    VALUE = 2


class InputType(Enum):
    """Type of Models' input.

    - ``POINTWISE``: Point-wise input, like ``uid, iid, label``.
    - ``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
    """

    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3


class FeatureType(Enum):
    """Type of features.

    - ``TOKEN``: Token features like user_id and item_id.
    - ``FLOAT``: Float features like rating and timestamp.
    - ``TOKEN_SEQ``: Token sequence features like review.
    - ``FLOAT_SEQ``: Float sequence features like pretrained vector.
    """

    TOKEN = "token"
    FLOAT = "float"
    TOKEN_SEQ = "token_seq"
    FLOAT_SEQ = "float_seq"


class FeatureSource(Enum):
    """Source of features.

    - ``INTERACTION``: Features from ``.inter`` (other than ``user_id`` and ``item_id``).
    - ``USER``: Features from ``.user`` (other than ``user_id``).
    - ``ITEM``: Features from ``.item`` (other than ``item_id``).
    - ``USER_ID``: ``user_id`` feature in ``inter_feat`` and ``user_feat``.
    - ``ITEM_ID``: ``item_id`` feature in ``inter_feat`` and ``item_feat``.
    - ``KG``: Features from ``.kg``.
    - ``NET``: Features from ``.net``.
    """

    INTERACTION = "inter"
    USER = "user"
    ITEM = "item"
    USER_ID = "user_id"
    ITEM_ID = "item_id"
    KG = "kg"
    NET = "net"
```