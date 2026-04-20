Viewed kuairan1k.py:1-262
Listed directory Crocodile-main
Listed directory src
Searched for "SharedBottom"
Viewed MDST_SharedBottom.py:1-94
Listed directory Kuairand1k_MDCTR
Viewed dataset_config.yaml:1-145
Viewed model_config.yaml:1-484
Viewed ple.py:1-109
Viewed dataset_config.yaml:1-145

根据对两个工程代码的深入比对（当前目录基于 `FuxiCTR` 架构的 `Crocodile-main` 项目，与你的 `studybole` 库下的 `kuairan1k.py` 和底层逻辑），不仅是因为你提到的 `user_dim` 和 `item_dim` 设成了 1（这确实极大剥夺了协同过滤的信息），在这背后还在**数据特征工程、PyTorch 优化器陷阱、网络结构设计**方面存在多处核心差异，导致了你极其优秀的 PLE 架构跑不赢简单的 SharedBottom。

我梳理出了以下几个最致命的原因，你可以逐一对照：

### 1. 致命的 Pytorch `weight_decay` 陷阱（优化器层面）
在你的 `kuairan1k.py` 第 209 行，你使用了标准的 Adam 优化器来包含整个模型的参数（这实际上是 PyTorch 中最常见的 CTR 踩坑点）：
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```
**问题极其严重：** PyTorch 自带的 weight_decay 是全局的 L2 正则。对于极其稀疏的 ID Embedding（由于大量 ID 长尾且只出现几次），在它们没有被采样到的 iteration 中，梯度虽然是 0，**但 `weight_decay` 依然在迭代更新中强制衰减这些 Embedding！**
这就意味着你大量稀疏特征的学到的表征被你无差别地每一轮强制向 0 压缩。
**反观当前项目：**
`model_config.yaml` 中明确进行了剥离：
```yaml
net_regularizer: 1.e-6
embedding_regularizer: 0.
```
FuxiCTR 对 Embedding 禁用了 `weight_decay`，仅仅对 DNN 使用了正则化，这保全了稀疏特征的学习能力。

### 2. 遗漏了顶级强特征（特征工程层面）
在 `kuairan1k.py` 的特征配置（第 121-122 行附近）中，你注释掉了两个极具区分度的强特征：
```python
# SparseEmbSetting("author_id", FeatureSource.ITEM, 16),
# SparseEmbSetting("music_id", FeatureSource.ITEM, 16),
```
在视频推荐尤其是 KuaiRand 数据集里，**`author_id`（作者） 几乎是除了 `video_id` 之外最强的泛化特征**（用户粉谁、甚至特定作者只发特定调性的视频，具备极强的类聚效应），缺失了这个特征，你的原始输入信息就少了一大截。
**反观当前项目：**
`dataset_config.yaml` 毫无悬念地接纳了它们：
```yaml
{ name: [ 'video_id','author_id', 'video_type','music_id', 'music_type'] ... }
```

### 3. 长尾数值特征处理不当归一化（特征工程层面）
在处理由连续值构成的用户属性特征时，你的做法：
```python
MinMaxDenseSetting("follow_user_num", ...),
MinMaxDenseSetting("fans_user_num", ...),
MinMaxDenseSetting("register_days", ...),
```
CTR 领域的连续数值往往呈极大的**长尾分布（Power Law）**，绝大多数人粉丝是个位数，但极少部分人粉丝超几万。如果直接使用 MinMax，大量的用户数值直接被压缩到非常贴近于 `0` 的极小值内，神经网络对这种被暴击挤压的稠密输入不敏感。
**反观当前项目：**
它使用了清洗好的离散化**分桶特征**（Bucketized Features）：
```yaml
... 'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range', 'register_days_range' ...
```
并且将其作为 `categorical` 进行 32 维的独立 Embedding。分桶 Embedding 直接赋予了神经网络非线性的阶跃抓取能力，而非线性往往比稠密 MLP 更容易学习这种长尾特征。

### 4. 共享底座的信息与维度碾压（模型容量层面）
就算你使用了 PLE 这个更高级的模型，但是在规模上，当前目录被调得更暴力：
* **特征维度优势：** 目前所有 `Crocodile-main` 特征（不管是 ID 还是 user_active_degree）都被映射成了广阔的 **32 维**（`embedding_dim=32`），而你的实现除了 `user/item = 1` 之外，其他的也仅仅只有 **16 维**。
* **底部参数碾压：** FuxiCTR 的 SharedBottom 中 `bottom_hidden_units=[256, 128, 64]`；叠加所有强特征 32维 的宽输入后，信息被非常庞大地交叉。你的 PLE 的 expert_dims 是 `[256, 128]` 并在第二层再传 `128`，再加上输入的维度太小，实际上并没有发挥出多专家分化路线的优势。

### 5. Domain 监督逻辑的细微差异（架构设计）
在 `Crocodile-main` 项目下的 `MDST_SharedBottom.py` 中，我们可以看到它针对 Multi-Scenario 是这样计算的：
```python
y_preds = torch.stack([self.output_activation[i](tower_output[i]) for i in range(self.num_domains)],dim=1)
y_preds_flattened = torch.flatten(y_preds, start_dim=0, end_dim=1)
y_pred = torch.index_select(y_preds_flattened, 0, indices)  # index 是属于哪个 domain
```
它通过硬截断（`index_select`）精准抽取当前 domain 对应的 Tower 输出来做二分类交叉熵 Loss 等计算。
虽然你的 PLE 也做了 `domain_ids` 抽取逻辑进行门控（Gate）分配，但在上述前 3 点的累积摧残下，本该精雕细琢的 PLE Routing 学习过程完全受到了梯度和维度的干扰（被 Weight Decay 毒打，又缺失了最关键维度和 Author 关联）。

**总结：**
并不是你的模型本身实现得很差。把 `kuairan1k.py` 下的 `weight_decay` 设为 `0` （只给 linear 层加正则），解放 `user_dim/item_dim` 并加回被你注释的 `author_id`，换个好的数值分桶方案，你的结果会立刻出现飞跃式反超。