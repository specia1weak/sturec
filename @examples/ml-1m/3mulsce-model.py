import torch
from Tools.scripts.dutree import store
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.abstract_recommender import ContextRecommender
from recbole.trainer import Trainer
from recbole.utils import InputType

from betterbole.utils import ignore_future_warning, change_root_workdir
from torch import nn, optim

change_root_workdir()
ignore_future_warning()

"""
三种模型三种情况，你会发现可能MMoE的核心点是不同的场景由不同的tower来判断，多expert的作用或许没有那么明显. 甚至SharedBottom胜过MMoE
所以可能复杂架构需要匹配复杂数据。不过把男女分开来确实比不分表现更好一点。真是奇怪。
这个问题可能需要看看PLE模型的效果再做定论了。PLE把多专家确实玩的比较厉害
"""

class MultiScenarioModel(ContextRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(MultiScenarioModel, self).__init__(config, dataset)
        self.SCENARIO_FIELD = config['SCENARIO_FIELD']
        self.num_scenarios = dataset.num(self.SCENARIO_FIELD)

        self.input_dim = self.num_feature_field * self.embedding_size
        self.output_dim = 64
        self.num_experts = 3

        # 对一个输入有3个专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim),
                nn.ReLU()
            )
            for _ in range(self.num_experts)
        ])

        # 对每一个场景
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.output_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for _ in range(self.num_scenarios)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.num_experts),
                nn.Softmax(dim=1) # 对最后一维的所有值归一化
            )
            for _ in range(self.num_scenarios)
        ])

    def forward(self, x):
        # [B, 3, D]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # [num_scenarios, B, 3] -> 注意这里修复了语法错误
        gate_scores = [gate(x) for gate in self.gates]

        scenario_scores = []
        for i in range(self.num_scenarios):
            # [B, 3, 1] * [B, 3, D] -> 修复了 unsqueeze 拼写
            weighted_output = gate_scores[i].unsqueeze(-1) * expert_outputs
            sum_output = torch.sum(weighted_output, dim=1)  # [B, D]
            scenario_score = self.towers[i](sum_output)  # [B, 1]
            scenario_scores.append(scenario_score.squeeze(dim=1))

        return torch.stack(scenario_scores, dim=1)  # [B, num_scenarios]

    def predict(self, interaction):
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        scenario_ids = interaction[self.SCENARIO_FIELD]
        # raw logits
        scenario_scores = self.forward(x)
        B = scenario_scores.shape[0]
        # 使用高级索引提取对应场景的分数
        final_logits = scenario_scores[torch.arange(B, device=scenario_scores.device), scenario_ids]
        # predict 方法在评测时调用，通常需要返回真实概率用于计算 AUC/LogLoss
        return torch.sigmoid(final_logits)

    def calculate_loss(self, interaction):
        # 获取标签，确保转换为浮点型以匹配 Loss 计算
        labels = interaction[self.LABEL].float()
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        scenario_ids = interaction[self.SCENARIO_FIELD]
        scenario_scores = self.forward(x)
        B = scenario_scores.shape[0]
        final_logits = scenario_scores[torch.arange(B, device=scenario_scores.device), scenario_ids]
        # 使用带 Logits 的 BCE，数值稳定性更好，且无需提前 Sigmoid
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
        return loss

class RandomModel(ContextRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(RandomModel, self).__init__(config, dataset)
        self.SCENARIO_FIELD = config['SCENARIO_FIELD']
        self.fake_var = nn.Parameter(torch.tensor(1.0))
    def predict(self, interaction):
        scenario_ids = interaction[self.SCENARIO_FIELD]
        B = scenario_ids.shape[0]
        return torch.randn(B)

    def calculate_loss(self, interaction):
        return  self.fake_var * (1 + 1e-4)

class SimpleModel(ContextRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(SimpleModel, self).__init__(config, dataset)
        self.input_dim = self.num_feature_field * self.embedding_size
        self.output_dim = 64
        self.num_experts = 3
        self.expert = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.expert(x).squeeze()

    def predict(self, interaction):
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x)
        return torch.sigmoid(final_logits)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
        return loss

class TowTowersModel(ContextRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(TowTowersModel, self).__init__(config, dataset)
        self.SCENARIO_FIELD = config['SCENARIO_FIELD']
        self.input_dim = self.num_feature_field * self.embedding_size
        self.num_scenarios = dataset.num(self.SCENARIO_FIELD)
        self.output_dim = 64
        self.num_experts = 3
        self.expert = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.ReLU(),
        )
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.output_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(self.num_scenarios)
        ])
    def forward(self, x):
        expert_output = self.expert(x)
        tower_scores = [tower(expert_output).squeeze(-1) for tower in self.towers] # [2, B]
        return torch.stack(tower_scores, dim=1) # [B, 2]

    def predict(self, interaction):
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        scenario_ids = interaction[self.SCENARIO_FIELD]
        B = scenario_ids.shape[0]
        scenario_scores = self.forward(x)
        final_logits = scenario_scores[torch.arange(B, device=scenario_scores.device), scenario_ids]
        return torch.sigmoid(final_logits)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        scenario_ids = interaction[self.SCENARIO_FIELD]
        B = scenario_ids.shape[0]
        scenario_scores = self.forward(x)
        final_logits = scenario_scores[torch.arange(B, device=scenario_scores.device), scenario_ids]
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
        return loss

if __name__ == '__main__':
    cfg_override = {
        "embedding_size": 64,
        "train_batch_size": 2048,
        "SCENARIO_FIELD": "gender",  # 自定义cfg
        "eval_args": {
            "group_by": "none",
            "order": "RO", # 随机排序
            "split": {
                "RS": [0.8, 0.1, 0.1]
            },
            "mode": "labeled" # 使用数据label
        },
        "metrics": ["AUC", "LogLoss"],
        "valid_metric": "AUC",
    }
    model_cls = TowTowersModel
    device = "cuda"
    if device == "cpu":
        cfg_override.update(gpu_id="")

    config = Config(model=model_cls, dataset='ml-1m', config_file_list=['dataset/ml-1m/m1-1m.yaml'],
                    config_dict=cfg_override)
    # 2. 创建并加载数据集
    # 这一步会读取原子文件，进行 ID 映射 (Remapping)，并打印数据集的统计信息
    dataset = create_dataset(config)
    print("========== 数据集基本信息 ==========")
    print(dataset)
    print("===================================\n")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = model_cls(config, dataset).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    trainer = Trainer(config, model)

    for epoch in range(50):
        model.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(train_data):
            batch_data = batch_data.to(device)
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


    result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=False
    )
    print(f"测试集核心指标: {result}")
