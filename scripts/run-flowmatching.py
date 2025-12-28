import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm


# ================= 配置部分 =================
class Config:
    # 路径配置 (请确保与你实际路径一致)
    BASE_DIR = Path("D:/pyprojects/recommend-study/studybole/dataset/Amazon_Processed_10Core/Pretrained_Embeddings")
    SOURCE_EMB_PATH = BASE_DIR / "Source_Books_user_emb.pkl"
    TARGET_EMB_PATH = BASE_DIR / "Target_Movies_user_emb.pkl"

    # 模型保存路径
    MODEL_SAVE_DIR = Path("./saved_models")

    # 训练参数
    EMBEDDING_SIZE = 64
    HIDDEN_DIM = 256
    BATCH_SIZE = 1024
    LR = 1e-3
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Flow Matching 参数
    SIGMA_MIN = 1e-4  # 避免除零的极小值


# ================= 1. 数据集定义 =================
class CrossDomainUserDataset(Dataset):
    def __init__(self, source_path, target_path):
        print(f"[-] Loading embeddings...")
        with open(source_path, 'rb') as f:
            self.source_dict = pickle.load(f)
        with open(target_path, 'rb') as f:
            self.target_dict = pickle.load(f)

        # 寻找重叠用户 (Intersection)
        self.common_users = list(set(self.source_dict.keys()) & set(self.target_dict.keys()))
        self.common_users.sort()  # 固定顺序

        print(f"[-] Found {len(self.common_users)} overlapping users for training.")

        # 预先转换为 Tensor 以加速
        self.source_matrix = []
        self.target_matrix = []

        for uid in self.common_users:
            self.source_matrix.append(self.source_dict[uid])
            self.target_matrix.append(self.target_dict[uid])

        self.source_matrix = torch.tensor(np.array(self.source_matrix), dtype=torch.float32)
        self.target_matrix = torch.tensor(np.array(self.target_matrix), dtype=torch.float32)

    def __len__(self):
        return len(self.common_users)

    def __getitem__(self, idx):
        return self.source_matrix[idx], self.target_matrix[idx]


# ================= 2. Flow Matching 模型 =================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


import math


class FlowMatchingNet(nn.Module):
    def __init__(self, in_dim=64, cond_dim=64, hidden_dim=256):
        super().__init__()

        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 条件编码 (源域用户 Embedding)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 主网络 (输入 x_t)
        self.input_mlp = nn.Linear(in_dim, hidden_dim)

        # 融合层
        self.mid_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim)  # 输出速度 v
        )

    def forward(self, x, t, condition):
        # x: [B, 64] 当前状态
        # t: [B]     时间 (0~1)
        # condition: [B, 64] 源域用户Emb

        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(condition)
        x_emb = self.input_mlp(x)

        # 简单的相加融合 (也可以用 Concat)
        h = x_emb + t_emb + c_emb
        velocity = self.mid_block(h)

        return velocity


# ================= 3. 训练逻辑 =================
def train():
    Config.MODEL_SAVE_DIR.mkdir(exist_ok=True)

    # 1. 准备数据
    dataset = CrossDomainUserDataset(Config.SOURCE_EMB_PATH, Config.TARGET_EMB_PATH)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 2. 初始化模型
    model = FlowMatchingNet(in_dim=Config.EMBEDDING_SIZE,
                            cond_dim=Config.EMBEDDING_SIZE,
                            hidden_dim=Config.HIDDEN_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    print("[-] Start Training Optimal Transport Flow Matching...")

    # 3. 训练循环
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0

        for source_emb, target_emb in dataloader:
            source_emb = source_emb.to(Config.DEVICE)  # Condition
            target_emb = target_emb.to(Config.DEVICE)  # Ground Truth (x1)
            batch_size = source_emb.shape[0]

            # --- Flow Matching 核心逻辑 (OT-CFM) ---

            # A. 采样初始噪声 x0 (Standard Gaussian)
            x0 = torch.randn_like(target_emb)

            # B. 采样时间 t [0, 1]
            t = torch.rand(batch_size).to(Config.DEVICE)

            # C. 构建中间状态 x_t (最优传输路径：直线插值)
            # x_t = (1 - t) * x0 + t * x1
            t_expand = t.view(-1, 1)
            x_t = (1 - t_expand) * x0 + t_expand * target_emb

            # D. 计算目标速度 (Target Velocity)
            # v = dx/dt = x1 - x0
            v_target = target_emb - x0

            # E. 模型预测速度
            v_pred = model(x_t, t, source_emb)

            # F. Loss: 回归速度场
            loss = torch.mean((v_pred - v_target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # --- 简单验证: 生成 20 个样本看多样性和准确性 ---
        if (epoch + 1) % 10 == 0:
            val_dist = validate_one_step(model, dataset)
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.6f} | Val Dist(MSE): {val_dist:.6f}")

            # 保存模型
            torch.save(model.state_dict(), Config.MODEL_SAVE_DIR / "flow_model_latest.pth")


@torch.no_grad()
def validate_one_step(model, dataset):
    """
    随机取一个用户，生成 20 个样本，计算平均 MSE
    模拟 'Set Generation' 策略
    """
    model.eval()
    # 随机取一个数据
    idx = np.random.randint(0, len(dataset))
    src, tgt = dataset[idx]
    src = src.unsqueeze(0).to(Config.DEVICE)  # [1, 64]
    tgt = tgt.unsqueeze(0).to(Config.DEVICE)  # [1, 64]

    # 1. 复制 20 份
    num_samples = 20
    src_batch = src.repeat(num_samples, 1)  # [20, 64]

    # 2. 采样 20 个不同的初始噪声
    x_current = torch.randn(num_samples, Config.EMBEDDING_SIZE).to(Config.DEVICE)

    # 3. Euler 求解 (1步生成)
    # Flow Matching 的强大之处：只要一步 dt=1 就能大致到位
    t = torch.zeros(num_samples).to(Config.DEVICE)
    v_pred = model(x_current, t, src_batch)
    x_final = x_current + v_pred * 1.0  # x1 = x0 + v * 1

    # 4. 计算与真实 Target 的距离
    mse = torch.mean((x_final - tgt) ** 2).item()
    return mse


if __name__ == "__main__":
    os.chdir("..")
    train()