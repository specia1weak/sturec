import pandas as pd
from recbole.data.dataset import Dataset
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import ContextSeqEmbAbstractLayer
from recbole.trainer import Trainer
from recbole.utils import InputType, EvaluatorType, set_color, get_gpu_usage
import torch
from torch import nn, optim
import torch.nn.functional as F
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from tqdm import tqdm

from betterbole.utils import set_all
from betterbole.utils import BoleEmbLayer, EmbSetting, SparseEmbSetting, ItemSeqEmbSetting, UserSideEmb, ItemSideEmb, \
    SparseSeqEmbSetting
from betterbole.utils import NamedTimer
timer = NamedTimer()
set_all()

"""
Gemini said
实现一个完整的 DIN 就像搭建一条精密的流水线。为了让你能在不看代码的情况下顺利写出整个架构，我们将整个前向传播（Forward Pass）拆解为 6 个核心模块。
1. 特征体系的定义与拆解（输入层）
在构建模型之前，你的模型需要明确接收哪些类型的输入。DIN 的输入通常被严格划分为四大阵营：
User Profile（用户画像特征）：例如性别、年龄、职业的 ID 编码。
Context（上下文特征）：例如当前时间、设备类型、所在城市的 ID 编码。
Target Item（目标物品特征）：当前准备预测是否会点击的物品。不仅包含 Item ID，还包含该物品的 Category ID、Brand ID 等附属特征。
History Sequence（历史行为序列）：用户过去点击过的 N 个物品。重点：它不仅是一个 Item ID 的列表，如果目标物品有 Category ID，历史序列中的每一个物品也必须携带对应的 Category ID。

2. Embedding 字典与查表（表示层）
这一步的核心是降维与特征共享。
为每一个离散特征（ID 类特征）初始化一个 Embedding 矩阵。
避坑指南（关键）：Target Item 和 History Sequence 中的同类特征必须共享同一个 Embedding 矩阵。也就是说，目标物品的类别查的是 Category_Embedding_Table，历史物品的类别查的也是同一个 Category_Embedding_Table。
组装物品向量：查表后，将目标物品的所有特征 Embedding（如 ID + Category + Brand）在特征维度上拼接（Concat），形成一个完整的 Target Item Vector。对历史序列中的每一个物品做同样的操作，形成一个矩阵 History Sequence Matrix [Batch,Seq_Len,Embedding_Dim]。

3. 局部激活单元（Attention 层）
这里直接插入你已经写好的 SequenceAttLayer。
输入：上一步得到的 Target Item Vector、History Sequence Matrix，以及记录每个用户真实历史长度的 Sequence Lengths。
交互与计算：将 Target 与 History 的每一个物品进行拼接、相减、相乘，送入一个微型 MLP，得到一组权重。
Masking（掩码）：利用 Sequence Lengths 将填充（Padding）位置的权重强行置为极小值（或 0）。
加权求和：根据算出的权重，对 History Sequence Matrix 在序列长度维度上进行加权求和（不经过 Softmax）。
输出：得到一个与 Target Item Vector 维度相同的向量，这就是动态捕捉到的 User Interest Vector（用户兴趣向量）。

4. 终极信息拼装（Fusion 层）
这是进入最终决策前的最后集结。你需要将所有的信息拼接成一个超长的一维向量（针对单个样本而言）：
上一步得到的 User Interest Vector。
组装好的 Target Item Vector。
查表得到的 User Profile Vector 拼接组合。
查表得到的 Context Vector 拼接组合。
（可选但强烈推荐的 DIN 常用操作）：将 User Interest Vector 和 Target Item Vector 进行逐元素相乘（Element-wise Product）的结果也拼接入内，显式提供匹配信号。

5. 顶层多层感知机（Final MLP）
将第 4 步得到的超长拼接向量，送入一个深层的前馈神经网络。
网络形状：通常是典型的塔型结构，神经元数量逐层递减，例如 [1024, 512, 256]。
Dice 激活函数（DIN 的独门秘籍）：原版 DIN 在这里的隐藏层不使用 ReLU，而是使用 Dice (Data Adaptive Rectified Linear Unit)。它会根据当前 Batch 数据的均值和方差，自适应地调整激活函数的截断点。(建议你第一版手写时先用 PReLU 或 ReLU 跑通，第二版再挑战手写 Dice)。

6. 预测与损失计算（输出层）
线性映射：将 Final MLP 的最后一层输出，通过一个单神经元的线性层（nn.Linear(256, 1)），映射为一个实数（Logit）。
概率转换：使用 Sigmoid 函数将实数压缩到 (0,1) 区间，这就是最终的 CTR（点击率）预估值。
损失函数：在训练阶段，使用这个预估值与真实的点击标签（0 或 1），计算二元交叉熵损失（BCE Loss），然后反向传播更新所有 Embedding 和 MLP 的权重。
"""


## 用不上
class SuperSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For Deep Interest Network, return all features (including user features and item features) embedding matrices."""

    def __init__(self, dataset, embedding_size, pooling_mode, device):
        super(SuperSeqEmbLayer, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.dataset: Dataset = dataset
        self.user_feat = self.dataset.get_user_feature().to(self.device)
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {
            "user": list(self.user_feat.interaction.keys()),
            "item": list(self.item_feat.interaction.keys()),
        }

        self.types = ["user", "item"]
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ["mean", "max", "sum"]
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()


class MLP(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)

class DinAttn(nn.Module):
    def __init__(self, item_dim):
        super(DinAttn, self).__init__()
        self.mlp = MLP(item_dim * 4, item_dim * 2, 1)

    def forward(self, target, item_seq, seq_len):
        """
        :param target: B D
        :param item_seq: B L D
        :param seq_len: B
        :return: B D
        """
        B, L, D = item_seq.shape
        device = target.device
        # B D 与 B L D分别拼接得到 B L 4*D
        target = target.unsqueeze(1).expand(B, L, D) # B 1 D
        sub = item_seq - target # B L D
        dot = item_seq * target # B L D
        # 输入mlp
        mlp_input = torch.cat([target, item_seq, sub, dot], dim=-1)
        mlp_output = self.mlp(mlp_input).view(B, L, 1) # B L 1
        # BLD BL1加权求和 B D, mask呢 怎么制作 B L 1的mask
        tmp = torch.arange(L, device=device).view(1, L).expand(B, L)
        mask: torch.BoolTensor = tmp < seq_len.unsqueeze(1)
        score = mlp_output * item_seq # B L D
        return torch.sum(score * mask.float().unsqueeze(-1), dim=1)


class MyDin(SequentialRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(MyDin, self).__init__(config, dataset)
        self.LABEL = config["LABEL_FIELD"]
        dataset: Dataset = dataset
        # ['user_id', 'item_id', 'timestamp', 'age', 'gender', 'occupation', 'label']
        user_side_settings = [
            SparseEmbSetting(dataset, self.USER_ID, 64),
            SparseEmbSetting(dataset, "age", 16),
            SparseEmbSetting(dataset, "gender", 16),
            SparseEmbSetting(dataset, "occupation", 16),
        ]

        item_side_settings = [
            SparseEmbSetting(dataset, self.ITEM_ID, 64),
            SparseEmbSetting(dataset, "release_year", 16),
            SparseSeqEmbSetting(dataset, "genre", 16)
        ]
        self.user_side_embedding = UserSideEmb(user_side_settings)
        self.item_side_embedding = ItemSideEmb(item_side_settings)

        self.attn = DinAttn(self.item_side_embedding.embedding_size)
        self.mlp = MLP(self.item_side_embedding.embedding_size * 2 + self.user_side_embedding.embedding_size, 64 * 2, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, user_emb, item_emb_seq, seq_len, target_item_emb, ctx_emb=None):
        """
        :param user_emb: 用户侧emb拼接
        :param item_emb_seq: 物品侧历史emb序列
        :param ctx_emb: 交互emb拼接
        :param target_item_emb: 目标物品emb
        :return: logits
        """
        # Attn -> interest_emb
        interest_emb = self.attn(target_item_emb, item_emb_seq, seq_len)
        # Fusion: [interest, user, target, ctx, cross]
        features = [interest_emb, user_emb, target_item_emb]
        if ctx_emb is not None:
            features.append(ctx_emb)
        fusion = torch.cat(features, dim=-1)
        # MLP -> logits
        logits = self.mlp(fusion).squeeze()
        return logits

    def predict(self, interaction):
        user_emb = self.user_side_embedding(interaction, flat2tensor=True)
        item_emb_seq = self.item_side_embedding(interaction[self.ITEM_SEQ], flat2tensor=True)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        target_item_emb = self.item_side_embedding(interaction[self.ITEM_ID], flat2tensor=True)
        logits = self.forward(user_emb, item_emb_seq, item_seq_len, target_item_emb)
        return F.sigmoid(logits)

    def calculate_loss(self, interaction):
        user_emb = self.user_side_embedding(interaction, flat2tensor=True)
        item_emb_seq = self.item_side_embedding(interaction[self.ITEM_SEQ], flat2tensor=True)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        target_item_emb = self.item_side_embedding(interaction[self.ITEM_ID], flat2tensor=True)
        label = interaction[self.LABEL]
        logits = self.forward(user_emb, item_emb_seq, item_seq_len, target_item_emb).squeeze()
        loss = self.criterion(logits, label)
        return loss

    def full_sort_predict(self, interaction):
        raise NotImplementedError


class InterQuery:
    def __init__(self, dataset: Dataset, max_len=10):
        self.max_len = max_len
        valid_col = (dataset.uid_field, dataset.iid_field, dataset.time_field, dataset.label_field)
        self.USER_ID, self.ITEM_ID, self.TIME, self.LABEL = valid_col
        df_data = {k: v for k, v in dataset.inter_feat.interaction.items() if k in valid_col}
        self.df = self._prepare_df(pd.DataFrame(df_data))

    def _prepare_df(self, df):
        df = df.sort_values(self.TIME).reset_index(drop=True)

        def get_user_history_series(user_df, max_len=self.max_len):
            items = user_df[self.ITEM_ID].tolist()
            labels = user_df[self.LABEL].tolist()

            histories = []
            current_hist = []

            for item, label in zip(items, labels):
                histories.append(list(current_hist[-max_len:]))
                if label >= 1:
                    current_hist.append(item)
            return pd.Series(histories, index=user_df.index)

        # 2. 直接赋值给原表的新列
        # 加了 include_groups=False，明确告诉 Pandas 里面不需要 user_id，消除警告
        df['history'] = df.groupby(self.USER_ID, group_keys=False).apply(
            get_user_history_series,
            include_groups=False
        )
        df = df.set_index([self.USER_ID, self.ITEM_ID, self.TIME]).sort_index()
        return df

    def query_history(self, user_id_list, item_id_list, timestamp_list):
        # 1. 构造目标查询表
        target_df = pd.DataFrame(
            {
                self.USER_ID: user_id_list,
                self.ITEM_ID: item_id_list,
                self.TIME: timestamp_list
            }
        )
        # 这是底层高度优化的索引匹配，速度极快且不会有 reindex 的类型推导 bug
        query_res = target_df.join(self.df, on=[self.USER_ID, self.ITEM_ID, self.TIME], how="left")

        history_list = query_res["history"].to_list()
        history_len = []

        batch_size = len(history_list)
        res_tensor = torch.zeros((batch_size, self.max_len), dtype=torch.long)
        for i, seq in enumerate(history_list):
            seq_len = 0
            if isinstance(seq, list):
                seq_len = min(len(seq), self.max_len)
                res_tensor[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            history_len.append(seq_len)
        return res_tensor, torch.tensor(history_len, dtype=torch.long)


"""
全量softmax， 全量Loss， 无neg采样， 不关注用户ID
"""

# 1. 初始化配置
# 注意：传 GRU4Rec 或 SASRec，Config 就会去加载序列模型的预处理逻辑
cfg_override = {
    "MAX_ITEM_LIST_LENGTH": 10,
    # ===== 数据集标签设置 =====
    "LABEL_FIELD": "label",  # 必须指定你的数据集中代表 0/1 标签的列名，默认通常是 'label'

    # ===== 负采样设置 =====
    "train_neg_sample_args": None,  # CTR 数据集自身已经包含了正负样本(0和1)，不需要框架在训练时动态生成负样本

    # ===== 评估设置 (核心修改) =====
    "eval_args": {
        "split": {'RS': [0.8, 0.1, 0.1]},  # CTR 常用按比例随机划分 (Random Split, 8:1:1)
        "order": 'TO',  # 配合随机划分使用随机打乱 (Random Order)
        "mode": 'labeled'  # 核心：必须改成 labeled。表示只在有明确标签的样本对上计算损失和指标，而不是去全库排序
    },

    # ===== 评估指标 =====
    "metrics": ['AUC', 'LogLoss'],  # CTR 任务的常用指标，不再使用 CF 的 Recall/NDCG
    "valid_metric": 'AUC'  # 根据哪个指标来保存 Early Stopping 的最优模型
}


def make_LS_dataset():
    config = Config(
        model='GRU4Rec',
        dataset='ml-1m',
        config_file_list=['dataset/ml-1m/m1-1m.yaml'],
        config_dict=cfg_override
    )

    # 2. 创建并加载数据集
    dataset = create_dataset(config)
    return config, dataset

class CustomTrainer(Trainer):
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        self.model.eval()
        eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            batch_data = batched_data[0]
            user_id_list = batch_data.interaction["user_id"]
            item_id_list = batch_data.interaction["item_id"]
            timestamp_list = batch_data.interaction["timestamp"]
            history, seq_len = iq.query_history(user_id_list, item_id_list, timestamp_list)
            batch_data[model.ITEM_SEQ] = history
            batch_data[model.ITEM_SEQ_LEN] = seq_len
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result


if __name__ == '__main__':
    config, dataset = make_LS_dataset()
    dataset: Dataset = dataset
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = MyDin(config, dataset).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    iq = InterQuery(dataset, config["MAX_ITEM_LIST_LENGTH"])
    trainer = CustomTrainer(config, model)

    print(f"训练集大小: {len(train_data.DATASET)}, 序列最大长度: {config['MAX_ITEM_LIST_LENGTH']}")

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        timer.start_record("epoch")
        for batch_idx, batch_data in enumerate(train_data):
            # break
            # 将数据推到正确的设备 (CPU/GPU)
            with timer("query"):
                user_id_list = batch_data.interaction["user_id"]
                item_id_list = batch_data.interaction["item_id"]
                timestamp_list = batch_data.interaction["timestamp"]
                history, seq_len = iq.query_history(user_id_list, item_id_list, timestamp_list)
            batch_data[model.ITEM_SEQ] = history
            batch_data[model.ITEM_SEQ_LEN] = seq_len

            batch_data = batch_data.to(config['device'])

            with timer("train"):
                loss = model.calculate_loss(batch_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        timer.stop_record("epoch")
        timer.report()
        avg_train_loss = total_loss / len(train_data)

        # 验证阶段
        result = trainer.evaluate(
            valid_data,
            load_best_model=False,
            show_progress=False
        )
        print(f"Epoch {epoch + 1:02d} | 训练 Loss: {avg_train_loss:.4f} | 验证集指标: {result}")
        print("-" * 50)