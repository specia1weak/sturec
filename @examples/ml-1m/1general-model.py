import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.trainer import Trainer
from recbole.utils import InputType
from torch import optim

# 查表然后算距离的极简模型，但是对付ml-1m仍能生效
class MySimpleBPR(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(MySimpleBPR, self).__init__(config, dataset)
        # 1. 定义 Embedding 层 (n_users 和 n_items 是父类自动算好的)
        self.embedding_size = config['embedding_size']  # 从 yaml 读取
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        # 2. 定义损失函数
        self.loss = BPRLoss()

    def forward(self, user, item):
        # 计算用户向量和物品向量的点积得分
        u_e = self.user_embedding(user)
        i_e = self.item_embedding(item)
        return torch.mul(u_e, i_e).sum(dim=1)  # dot product

    def calculate_loss(self, interaction):
        # RecBole 的 DataLoader 会自动帮你准备好正样本(item_id)和负样本(neg_item_id)
        user = interaction['user_id']
        pos_item = interaction['item_id']
        neg_item = interaction['neg_item_id']
        # 分别计算正负样本得分
        pos_score = self.forward(user, pos_item)
        neg_score = self.forward(user, neg_item)
        # 计算 BPR Loss
        return self.loss(pos_score, neg_score)

    def predict(self, interaction):
        user = interaction['user_id']
        item = interaction['item_id']
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        user = interaction['user_id']
        u_e = self.user_embedding(user)
        all_i_e = self.item_embedding.weight
        scores = torch.matmul(u_e, all_i_e.transpose(0, 1))
        return scores.view(-1)


train_cfg = {
    "embedding_size": 64,
    "train_batch_size": 2048,
    "gpu_id": "", # 实际上只有这种方法才能让device=cpu
    'val_interval': {
        'rating': '[4.0, 5.0]'
    }
}

if __name__ == '__main__':
    from src.utils import change_root_workdir, ignore_future_warning
    change_root_workdir()
    ignore_future_warning()

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    config = Config(model=MySimpleBPR, dataset='ml-1m', config_file_list=['dataset/ml-1m/m1-1m.yaml'], config_dict=train_cfg)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = MySimpleBPR(config, dataset)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    trainer = Trainer(config, model)

    for epoch in range(50):
        # --- 你的自定义训练阶段 ---
        model.train()  # 确保在训练模式
        total_loss = 0.0
        for batch_idx, batch_data in enumerate(train_data):
            loss = model.calculate_loss(batch_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_data)

        # --- 符合 RecBole 原理的验证阶段 ---
        # 直接调用 Trainer 的 evaluate，它会自动调用你的 full_sort_predict，处理 Tuple 数据，并计算出 NDCG/Recall 等指标
        result = trainer.evaluate(
            valid_data,
            load_best_model=False,
            show_progress=False
        )
        print(f"Epoch {epoch + 1} 完成 | 训练 Loss: {avg_train_loss:.4f}")
        print(f"验证集核心指标: {result}")
        print("-" * 50)