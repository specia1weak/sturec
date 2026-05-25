# TAAC 全量 120 个特征深度诊断与处理策略表

根据实地探查日志，本数据集共包含 **120** 个特征列。以下梳理涵盖了每一个具体字段的物理意义、类型、基数范围以及相对应的算法层级处理建议。

| 特征名 | 类型 | 范围 / 唯一值基数 | 处理建议 (🚨标红为极高危项，需优先处理) |
| :--- | :--- | :--- | :--- |
| **全局与上下文目标 (5个)** | | | |
| `user_id` | Int64 | 1,010,000 | 🛑 **直接丢弃**。这是单次曝光请求流水号，绝对不可作为特征入模，否则会强行记住样本导致严重过拟合。 |
| `item_id` | Int64 | 170,559 | ✅ 目标候选物料，标准 Embedding (建议 64 维)。 |
| `label_type` | Int32 | 2 (值为1或2) | ✅ 训练目标 (Target)，映射为 `0` (无转化) 和 `1` (有转化)，做 BCE / Focal Loss。 |
| `label_time` | Int64 | 155,058 (时间戳) | ⚠️ 标签时间，可不用作特征输入。 |
| `timestamp` | Int64 | 151,558 (时间戳) | 💡 极其重要。这是全局请求时间点，用于减去序列特征中的历史时间戳，计算时间差并做 Time Bucket Embedding。 |
| **物品基础画像 - Item Meta (14个)** | | | |
| `item_int_feats_5 ~ 10`<br>`item_int_feats_12, 13, 16` | Int64 | 个位 ~ 2.1万 | ✅ 单值离散特征(类目/品牌/作者)。直接查表 Embedding。 |
| `item_int_feats_81`<br>`item_int_feats_83 ~ 85` | Int64 | 3 ~ 1,038 | ✅ 统计分箱/状态标志。直接查表 Embedding。 |
| `item_int_feats_11` | List(Int64) | 19,792 | ✅ 多值离散标签(Item Tags)。查表后需使用 Masked Mean/Sum Pooling 融合。 |
| **用户静态与上下文特征 - User Int (46个)** | | | |
| `user_int_feats_1, 3, 4`<br>`user_int_feats_48 ~ 59`<br>`user_int_feats_82, 86`<br>`user_int_feats_92 ~ 109` | Int64 | 2 ~ 2,842 | ✅ 实时上下文/设备/网络特征 (标量)。全部做标准 Embedding，可分组 (Group) 拼接。 |
| `user_int_feats_15, 60, 80` | List(Int64) | 2 ~ 1,166 | ✅ 近期多值离散态。查表 Embedding 后进行 Mean Pooling。 |
| `user_int_feats_62 ~ 66` | List(Int64) | 11 ~ 1,389 | ✅ 近期交互标签轨迹。查表 Embedding 后进行 Mean Pooling。 |
| `user_int_feats_89 ~ 91` | List(Int64) | 8 ~ 10 | ✅ 定长状态序列。查表 Embedding 后进行 Mean/Concat 处理。 |
| **用户数值与高阶向量 - User Dense (10个)** | | | |
| `user_dense_feats_62 ~ 66` | List(Float32) | 120万 ~ 233万<br>🚨 **极值高达几百万** | 🛑 **梯度爆炸预警！** 这是历史累积次数/时长统计，**必须使用 `torch.log1p(x)` 或 Z-score 标准化**，切忌原样输入 MLP。 |
| `user_dense_feats_61`<br>`user_dense_feats_87` | List(Float32) | 8,772万 (Len=256)<br>3.1万 (Len=320) | 💡 **高纯度预训练向量**。无需 Embedding 查表，直接接一层 `LayerNorm + Linear` 降维，与其他特征 Concat。 |
| `user_dense_feats_89 ~ 91` | List(Float32) | 1.6万 ~ 1.7万 (连续值) | ✅ 已归一化的数值向量，直接 Concat 输入 MLP。 |
| **序列 A：中短视频/文章 (9个)** | | | |
| `domain_a_seq_38` | List(Int64) | 676,058 | ✅ 序列 A 核心 ID。长度适中 (100~1200)，基数安全，标准 Embedding。 |
| `domain_a_seq_39` | List(Int64) | 15,213,946 (时间戳) | 💡 序列 A 时间轴。计算 `timestamp - seq_39`，分箱获取 Time Embedding。 |
| `domain_a_seq_40 ~ 46` | List(Int64) | 10 ~ 12,341 | ✅ 序列 A 对齐属性 (是否点赞/完播等)。标准 Embedding，与主 ID 拼接算 Attention。 |
| **序列 B：高频域 (14个)** | | | |
| `domain_b_seq_67` | List(Int64) | 15,746,263 (时间戳) | 💡 序列 B 时间轴。同样减去全局 `timestamp` 算时间衰减。 |
| `domain_b_seq_69` | List(Int64) | 🚨 **54,413,688** | 🛑 **OOM显存溢出预警！** 主ID基数超 5400 万，直接全量 Embedding 必爆显存。**必须使用 Hash Trick (映射到300万以内) 或通过参数 `emb_skip_threshold` 截断丢弃**。 |
| `domain_b_seq_68, 70 ~ 79`<br>`domain_b_seq_88` | List(Int64) | 28 ~ 441,131 | ✅ 序列 B 侧属性。正常建表 Embedding 拼接。 |
| **序列 C：极高频域 (12个)** | | | |
| `domain_c_seq_27` | List(Int64) | 54,809,501 (时间戳) | 💡 序列 C 时间轴。按时间差分桶处理。 |
| `domain_c_seq_47` | List(Int64) | 🚨 **70,965,655** | 🛑 **OOM显存溢出预警！** 全场最大基数特征 (超 7000 万)，**强烈建议 Hash 化或低频截断**。 |
| `domain_c_seq_29` | List(Int64) | 🚨 **5,352,970** | ⚠️ 基数依然极高 (530万)，如果显存紧张，请对其同样应用 Hash 截断。 |
| `domain_c_seq_28, 30 ~ 37` | List(Int64) | 4 ~ 915,696 | ✅ 序列 C 其他侧属性，标准 Embedding。 |
| **序列 D：终身超长序列 (10个)** | | | |
| `domain_d_seq_23` | List(Int64) | 591,638 | 🛑 **计算卡死预警！** 该序列的主 ID 基数正常，但**单样本序列长度高达 3800+**。 |
| `domain_d_seq_17 ~ 22`<br>`domain_d_seq_24 ~ 26` | List(Int64) | 4 ~ 376,689 | 💡 **长序列截断策略**：切勿对长度达数千的序列直接算 Target Attention (O(N²) 复杂度)。必须在 Dataset 层截断至 Top-256 (配置 `seq_max_lens`)，或者采用 SIM 检索策略，先按类目 Hard Search 保留前 50 个相关行为再计算 Attention。 |

## 序列分析

| 序列 | 平均间隔 (取绝对值) | 换算时间 | 业务水平分析 |
| :--- | :--- | :--- | :--- |
| **序列 D** | 7,830 秒 | **约 2.1 小时** | **极高密度行为**。这是非常典型的高频沉浸式操作（如连续刷短视频、快速点击）。结合之前特征诊断中提到的“终身超长序列”以及单样本序列长度高达 3800+，说明这是一个高频且跨度极大的完整生命周期轨迹。[cite: 1] |
| **序列 A** | 102,199 秒 | **约 28.3 小时** | **常规日活行为**。用户平均隔 1.2 天产生一次交互，属于标准的中短视频或文章的每日浏览频率。 |
| **序列 B** | 143,539 秒 | **约 39.8 小时** | **中低频行为**。平均间隔 1.6 天左右，说明该域的触发门槛稍高（可能是某种特定业务线的点击或互动）。 |
| **序列 C** | 230,563 秒 | **约 64.0 小时** | **低频长效行为**。用户平均每隔 2.6 天才在这个域产生一次行为，属于相对稀疏的交互链路。 |