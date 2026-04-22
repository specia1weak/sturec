import math
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Trainer
from recbole.utils import InputType
import torch
from torch import nn, optim
import torch.nn.functional as F
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from betterbole.utils import set_all
set_all()

def compute_attn(q, k: torch.Tensor, v, mask=None):
    """
    :param q,k,v: B L D
    :param mask: B L L
    :return: B L D
    """
    B, L, D = q.shape
    # q @ kT 获得排列组合的相似度score B L L 1
    # softmax(score)/sqrt(d) @ v 将让权重附加到v加权求和 v应该是 B L D
    attn_score = q @ torch.transpose(k, 1, 2)  / math.sqrt(D) # L D @ D L, B L L
    if mask is not None:
        attn_score = torch.masked_fill(attn_score, mask=mask, value=-1e9)
    attn_weight = F.softmax(attn_score, dim=-1) # 最后一个维度才是权重
    attn_out = attn_weight @ v
    return attn_out

class QKVProject(nn.Module):
    def __init__(self, embedding_size):
        super(QKVProject, self).__init__()
        self.fc = nn.Linear(embedding_size, embedding_size)
    def forward(self, x):
        return self.fc(x)

class FFN(nn.Module):
    def __init__(self, embedding_size, inner_size):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, inner_size),
            nn.ReLU(),
            nn.Linear(inner_size, embedding_size)
        )
    def forward(self, x):
        return self.ffn(x)

"""
全量softmax， 全量Loss， 无neg采样， 不关注用户ID
"""

class MySASRec(SequentialRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(MySASRec, self).__init__(config, dataset)
        """ from SequentialRecommender: 
        
        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]
        """
        self.embedding_size = config["embedding_size"]
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)

        self.q_project = QKVProject(self.embedding_size)
        self.k_project = QKVProject(self.embedding_size)
        self.v_project = QKVProject(self.embedding_size)
        self.ffn = FFN(self.embedding_size, self.embedding_size * 4)
        self.layer_norm1 = nn.LayerNorm(self.embedding_size)
        self.layer_norm2 = nn.LayerNorm(self.embedding_size)

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def extract_from_interaction(self, interaction):
        item_seq = self.item_embedding(interaction[self.ITEM_SEQ])
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        if self.ITEM_ID in interaction:
            nxt_item = interaction[self.ITEM_ID]
        else:
            nxt_item = None
        return item_seq, item_seq_len, nxt_item

    def make_mask(self, seq_len):
        """
        :param seq_len: B
        :return: B, L, L
        """
        B, L = seq_len.shape[0], self.max_seq_length
        # 每行大于等于seq_len的元素变成True
        tmp = torch.arange(L, device=seq_len.device).unsqueeze(0).expand(B, -1) # BL
        mask1: torch.BoolTensor = tmp >= seq_len.unsqueeze(1) #B L  到时候要追加一个维度并和1 L L 相互位或
        mask2 = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq_len.device), diagonal=1) # L, L
        mask = mask1.unsqueeze(1) | mask2.unsqueeze(0)
        return mask

    def forward(self, seq, seq_len):
        """
        :param seq: B L D
        :return: full_out_seq B L D
        """
        # position
        position_idx = torch.arange(self.max_seq_length, device=self.device) # L
        position_bias = self.position_embedding(position_idx) #LD
        seq = position_bias + seq
        # project
        q, k, v = self.q_project(seq), self.k_project(seq), self.v_project(seq) # BLD
        # attn
        attn_out = compute_attn(q, k, v, self.make_mask(seq_len))
        # layer_norm and ffn
        res1 = self.layer_norm1(attn_out + seq)
        res2 = self.layer_norm2(res1 + self.ffn(res1))
        return res2

    def predict(self, interaction):
        """
        :param interaction
        :return: B, 所有序列的预测nxt物品id？
        """
        item_seq, item_seq_len, nxt_item = self.extract_from_interaction(interaction) # B, L, D / B / B
        obj_seq = self.forward(item_seq, item_seq_len) # B L D

        B, L, D = obj_seq.shape
        # 拿取最后一个obj -> B D
        batch_idx = torch.arange(B, device=obj_seq.device)
        nxt_predict = obj_seq[batch_idx, item_seq_len-1] # B D
        nxt_item_emb = self.item_embedding(nxt_item) # B D
        scores = (nxt_predict * nxt_item_emb).sum(dim=1)  # B D * B D
        return scores


    def calculate_loss(self, interaction):
        """
        需要将有效的所有下一预测全部纳入Loss计算
        """
        item_seq, item_seq_len, nxt_item = self.extract_from_interaction(interaction)  # B, L, D / B / B
        obj_seq = self.forward(item_seq, item_seq_len) # B L D
        # 除了头一个和无效的token，其他都要加进来，首先可能得到 B L的Loss矩阵，接着要做 B L的Loss mask
        # Loss矩阵并不好算，B L D 要和 N D进行一次排列组合得到 B L N 然后构造label label又是B L N
        candidate_items = torch.transpose(self.item_embedding.weight, 0, 1) # D N
        logits_seq = obj_seq @ candidate_items # B L N
        # prob_seq = F.softmax(logits_seq, dim=-1) # B L N

        # 目标label怎么构造呢 B L N
        B, L, N = logits_seq.shape
        item_id_seq = interaction[self.ITEM_SEQ] # B L
        n_idx = torch.arange(N, device=self.device).view(1, 1, N) # 1, 1, N

        label_item_id = item_id_seq.roll(shifts=-1, dims=1) # 集体左移
        label_item_id[torch.arange(B, device=self.device), item_seq_len-1] = nxt_item # 目标位置放上最终的nxt_item # B L
        # label_item_id = label_item_id.unqueeze(-1) # B L 1
        # label_prob = torch.where(label_item_id == n_idx, 1.0, 0.0)

        # Loss mask的构造 B L, 需要排除第一个预测
        tmp = torch.arange(L, device=self.device).unsqueeze(0).expand(B, -1)  # BL
        mask: torch.BoolTensor = tmp < item_seq_len.unsqueeze(1)  # B L 这里是有效的地方为True

        # 算总得Loss矩阵
        loss_mat = self.criterion(logits_seq.view(-1, N), label_item_id.view(-1)) # B*L,
        mask_flat = mask.view(-1)
        final_loss = torch.mean(loss_mat[mask_flat])
        return final_loss


    def full_sort_predict(self, interaction):
        item_seq, item_seq_len, _ = self.extract_from_interaction(interaction) # 为每一个预测
        obj_seq = self.forward(item_seq, item_seq_len)  # B L D

        B, L, D = obj_seq.shape
        # 拿取最后一个obj -> B D
        batch_idx = torch.arange(B, device=obj_seq.device)
        nxt_predict = obj_seq[batch_idx, item_seq_len - 1]  # B D
        # 接下来还需要算softmax
        candidate_items = torch.transpose(self.item_embedding.weight, 0, 1)  # D N
        logits = nxt_predict @ candidate_items  # B N
        return logits



# 1. 初始化配置
# 注意：传 GRU4Rec 或 SASRec，Config 就会去加载序列模型的预处理逻辑
cfg_override = {
    # "gpu_id": "",
    "MAX_ITEM_LIST_LENGTH": 10,
    "eval_args": {
        "split": {'LS': 'valid_and_test'},  # 留一法 LS
        "order": 'TO',
        "mode": 'full'
    },
    "val_interval": {
        "rating": '[4.0, 5.0]'
    },
    # ===== 新增这一行 =====
    "train_neg_sample_args": None  # 明确告诉框架：CE 损失不需要训练负采样
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

from recbole.model.sequential_recommender.sasrec import SASRec
def sasrec_dataset():
    config = Config(
        "SASRec", 'ml-1m', config_file_list=['dataset/ml-1m/m1-1m.yaml'], config_dict=cfg_override
    )
    dataset = create_dataset(config)
    return config, dataset

if __name__ == '__main__':
    if test_mysas := False:
        config, dataset = make_LS_dataset()
        ModelCls = MySASRec
    else:
        config, dataset = sasrec_dataset()
        ModelCls = SASRec
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = ModelCls(config, dataset).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    trainer = Trainer(config, model)

    print(f"训练集大小: {len(train_data.DATASET)}, 序列最大长度: {config['MAX_ITEM_LIST_LENGTH']}")

    for epoch in range(50):
        model.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(train_data):
            # 将数据推到正确的设备 (CPU/GPU)
            batch_data = batch_data.to(config['device'])
            loss = model.calculate_loss(batch_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_data)

        # 验证阶段
        result = trainer.evaluate(
            valid_data,
            load_best_model=False,
            show_progress=False
        )
        print(f"Epoch {epoch + 1:02d} | 训练 Loss: {avg_train_loss:.4f} | 验证集指标: {result}")
        print("-" * 50)