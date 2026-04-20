import math

import torch
import torch.nn as nn
from numpy.array_api import arange
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.trainer import Trainer
from recbole.utils import InputType
from torch import optim

class MySimpleSeqRecommender(SequentialRecommender):
    input_type = InputType.POINTWISE  # <--- 补上这一句！
    def __init__(self, config, dataset):
        super(MySimpleSeqRecommender, self).__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.embedding_size,
            batch_first=True
        )
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, item_seq_len):
        seq_emb = self.item_embedding(item_seq)  # [batch_size, max_seq_len, emb_size]
        gru_out, _ = self.gru(seq_emb)  # gru_out shape: [batch_size, max_seq_len, emb_size]
        seq_output = self.gather_indexes(gru_out, item_seq_len-1)
        return seq_output

    def calculate_loss(self, interaction):
        # 【关键 2】从 interaction 中取出序列相关特征。这里的 KEY 是基类初始化好的。
        item_seq = interaction[self.ITEM_SEQ]  # 用户的历史序列
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # 序列的真实长度
        pos_items = interaction[self.ITEM_ID]  # 目标 (Next Item)
        seq_output = self.forward(item_seq, item_seq_len)  # [batch_size, emb_size]
        # 2. 计算当前序列与**所有物品**的匹配得分 (内积)
        test_item_emb = self.item_embedding.weight  # [n_items, emb_size]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [batch_size, n_items]
        # 3. 计算交叉熵损失
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)

        # 点积得到特定 item 的打分
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight

        # 计算与全集物品的得分，用于 evaluate 排序指标 (NDCG, Recall 等)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores.view(-1)

import torch.nn.functional as F


class SeqAttn(nn.Module):
    def __init__(self, token_size=64):
        super(SeqAttn, self).__init__()
        self.token_size = token_size
        # 补全 Q, K, V 三个线性映射
        self.q_project = nn.Linear(token_size, token_size)
        self.k_project = nn.Linear(token_size, token_size)
        self.v_project = nn.Linear(token_size, token_size)
        self.attn_layer_norm = nn.LayerNorm(token_size)
        self.ffn_layer_norm = nn.LayerNorm(token_size)

        self.ffn = nn.Sequential(
            nn.Linear(token_size, token_size),
            nn.ReLU(),
            nn.Linear(token_size, token_size)
        )

    def forward(self, seq, mask=None):
        # seq 维度: (B, L, D)
        B, L, D = seq.shape
        q = self.q_project(seq)  # (B, L, D)
        k = self.k_project(seq)  # (B, L, D)
        v = self.v_project(seq)  # (B, L, D)
        mat = torch.matmul(q, k.transpose(-1, -2))
        mat = mat / math.sqrt(self.token_size)
        if mask is not None:
            mat = mat.masked_fill(mask, -1e9)  # mask只负责将无效地方变成-inf
        attn_score = F.softmax(mat, dim=-1)  # (B, L, L)
        attn_output = torch.matmul(attn_score, v)
        seq = self.attn_layer_norm(seq + attn_output)
        seq = self.ffn_layer_norm(seq + self.ffn(seq))
        return seq

    def forward_with_seq_len(self, seq, seq_len):
        # mask: B L L, 根据seq_len决定mask的取值, sl: B
        B, L, D = seq.shape
        with torch.no_grad():
            mask1 = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)
            mask1 = mask1.expand(B, L, L)
            tmp = torch.arange(L, dtype=torch.int, device=seq.device).expand(B, L) # B, L
            mask2 = tmp >= seq_len.unsqueeze(-1)
            mask = mask1 | mask2.unsqueeze(1).expand(-1, L, -1)
        return self.forward(seq, mask)



class SimpleSASRec(SequentialRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(SimpleSASRec, self).__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.max_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.embedding_size)
        self.seq_attn = SeqAttn(self.embedding_size)
        self.loss_fct = nn.CrossEntropyLoss()
        self.loss = nn.MSELoss()

    def forward(self, seq, seq_len):
        B, L = seq.shape[:2]
        seq_emb = self.item_embedding(seq)
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        seq_emb = seq_emb + self.position_embedding(positions)
        new_seq = self.seq_attn.forward_with_seq_len(seq_emb, seq_len) # B L D
        return new_seq
        # last_tokens = new_seq[torch.arange(B, device=seq.device), seq_len-1] # B D
        # return last_tokens

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        new_seq = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        target_seq = torch.zeros_like(new_seq)
        target_seq = target_seq[:, 1:]
        target_seq = torch.cat([target_seq, test_item_emb.unsqueeze(1)], dim=1) # B L D


        self.loss(new_seq, target_seq)

    def calculate_loss(self, interaction):
        # 【关键 2】从 interaction 中取出序列相关特征。这里的 KEY 是基类初始化好的。
        item_seq = interaction[self.ITEM_SEQ]  # 用户的历史序列
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # 序列的真实长度
        pos_items = interaction[self.ITEM_ID]  # 目标 (Next Item)
        seq_output = self.forward(item_seq, item_seq_len)  # [batch_size, emb_size]
        # 2. 计算当前序列与**所有物品**的匹配得分 (内积)
        test_item_emb = self.item_embedding.weight  # [n_items, emb_size]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [batch_size, n_items]
        # 3. 计算交叉熵损失
        loss = self.loss_fct(logits, pos_items)
        return loss
    

# --- 训练脚本 ---
train_cfg = {
    "train_batch_size": 2048,
    # "gpu_id": "",  # CPU 运行
    "embedding_size": 64,
    "MAX_ITEM_LIST_LENGTH": 10,  # 核心：强制开启并限制序列最大长度
    # 序列推荐通常使用 leave-one-out (留一法) 划分数据集，这里确保显式覆盖配置
    "eval_args": {
        "split": {"LS": "valid_and_test"},
        "order": "TO",  # 严格按时间 Time Order 排序
        "mode": "full"  # 全排序评测
    },
    "train_neg_sample_args": None,  # 明确告诉框架：CE 损失不需要训练负采样
    "val_interval": {
        "rating": '[4.0, 5.0]'
    }
}

if __name__ == '__main__':
    from betterbole.utils import change_root_workdir, ignore_future_warning
    change_root_workdir()
    ignore_future_warning()
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation

    # 【关键 3】直接把模型类 MySimpleSeqRecommender 传给 Config，而不是传字符串
    config = Config(model=MySimpleSeqRecommender,dataset='ml-1m',config_file_list=['dataset/ml-1m/m1-1m.yaml'], config_dict=train_cfg)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = MySimpleSeqRecommender(config, dataset).to(config['device'])
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