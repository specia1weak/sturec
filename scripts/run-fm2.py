import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import math


# ================= 配置部分 =================
class Config:
    # 路径配置
    BASE_DIR = Path("D:/pyprojects/recommend-study/studybole/dataset/Amazon_Processed_10Core/Pretrained_Embeddings")
    SOURCE_EMB_PATH = BASE_DIR / "Source_Books_user_emb.pkl"
    TARGET_EMB_PATH = BASE_DIR / "Target_Movies_user_emb.pkl"  # 目标是 User Emb，对应 CrossDomainUserDataset

    MODEL_SAVE_DIR = Path("./saved_models")

    EMBEDDING_SIZE = 64
    HIDDEN_DIM = 256
    BATCH_SIZE = 1024
    LR = 1e-3
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= 1. 数据集定义 (必须带修复！) =================
class CrossDomainUserDataset(Dataset):
    def __init__(self, source_path, target_path):
        print(f"[-] Loading embeddings...")
        with open(source_path, 'rb') as f:
            raw_source = pickle.load(f)
        with open(target_path, 'rb') as f:
            raw_target = pickle.load(f)

        # === 【关键修复】强制转字符串，解决 Key 类型不匹配问题 ===
        self.source_dict = {str(k).strip(): v for k, v in raw_source.items()}
        self.target_dict = {str(k).strip(): v for k, v in raw_target.items()}
        # ===================================================

        # 寻找重叠用户
        self.common_users = list(set(self.source_dict.keys()) & set(self.target_dict.keys()))
        self.common_users.sort()

        print(f"[-] Found {len(self.common_users)} overlapping users for training.")

        if len(self.common_users) == 0:
            raise RuntimeError("Overlap is 0! IDs mismatch.")

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
        # x 这里是 [B]，x[:, None] 是 [B, 1]
        # 乘法后是 [B, half_dim]
        # Cat 后是 [B, dim] -> 正常，不会广播爆炸
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowMatchingNet(nn.Module):
    def __init__(self, in_dim=64, cond_dim=64, hidden_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_mlp = nn.Linear(in_dim, hidden_dim)
        self.mid_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x, t, condition):
        # x: [B, D]
        # t: [B]
        # c: [B, D]
        t_emb = self.time_mlp(t)  # [B, H]
        c_emb = self.cond_mlp(condition)  # [B, H]
        x_emb = self.input_mlp(x)  # [B, H]

        # 这里的加法是 [B, H] + [B, H] + [B, H]，完全安全
        h = x_emb + t_emb + c_emb
        velocity = self.mid_block(h)
        return velocity


# ================= 3. 训练逻辑 (简单 MSE 版) =================
def train():
    Config.MODEL_SAVE_DIR.mkdir(exist_ok=True)
    dataset = CrossDomainUserDataset(Config.SOURCE_EMB_PATH, Config.TARGET_EMB_PATH)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    model = FlowMatchingNet(in_dim=Config.EMBEDDING_SIZE,
                            cond_dim=Config.EMBEDDING_SIZE,
                            hidden_dim=Config.HIDDEN_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    print("[-] Start Training MSE Flow Matching...")

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0

        for source_emb, target_emb in dataloader:
            source_emb = source_emb.to(Config.DEVICE)
            target_emb = target_emb.to(Config.DEVICE)
            batch_size = source_emb.shape[0]

            # 1. 采样 x0 (噪声)
            x0 = torch.randn_like(target_emb)

            # 2. 采样 t
            # 注意：这里 t 是 [B]，不是 [B, 1]，配合上面的模型定义是安全的
            t = torch.rand(batch_size).to(Config.DEVICE)

            # 3. 插值 x_t
            # t.view(-1, 1) 变成 [B, 1] 用于广播乘法
            t_expand = t.view(-1, 1)
            x_t = (1 - t_expand) * x0 + t_expand * target_emb

            # 4. 目标速度
            v_target = target_emb - x0

            # 5. 预测
            v_pred = model(x_t, t, source_emb)

            # 6. MSE Loss
            loss = torch.mean((v_pred - v_target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.6f}")
            torch.save(model.state_dict(), Config.MODEL_SAVE_DIR / "flow_model_latest.pth")


if __name__ == "__main__":
    os.chdir("..") # 视你的运行目录而定
    train()