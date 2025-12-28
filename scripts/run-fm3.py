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
    BASE_DIR = Path("D:/pyprojects/recommend-study/studybole/dataset/Amazon_Processed_10Core/Pretrained_Embeddings")
    SOURCE_EMB_PATH = BASE_DIR / "Source_Books_user_emb.pkl"
    TARGET_EMB_PATH = BASE_DIR / "Target_Movies_user_emb.pkl"
    MODEL_SAVE_DIR = Path("./saved_models")

    EMBEDDING_SIZE = 64
    HIDDEN_DIM = 256

    # 关键参数
    BATCH_SIZE = 1024
    LR = 1e-3
    EPOCHS = 500
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # === 核心策略 ===
    TRAIN_SAMPLES = 20  # 训练时每次采样 5 条路径，只奖励最好的那条


# ================= 1. 数据集 (带字符串修复) =================
class CrossDomainUserDataset(Dataset):
    def __init__(self, source_path, target_path):
        print(f"[-] Loading embeddings...")
        with open(source_path, 'rb') as f:
            raw_source = pickle.load(f)
        with open(target_path, 'rb') as f:
            raw_target = pickle.load(f)

        # 强制转字符串，确保对齐
        self.source_dict = {str(k).strip(): v for k, v in raw_source.items()}
        self.target_dict = {str(k).strip(): v for k, v in raw_target.items()}

        self.common_users = list(set(self.source_dict.keys()) & set(self.target_dict.keys()))
        self.common_users.sort()
        print(f"[-] Found {len(self.common_users)} overlapping users.")

        if len(self.common_users) == 0: raise RuntimeError("Overlap is 0!")

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


# ================= 2. 模型 (带广播修复) =================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # x: [N] -> [N, 1] -> [N, Dim]
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowMatchingNet(nn.Module):
    def __init__(self, in_dim=64, cond_dim=64, hidden_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.input_mlp = nn.Linear(in_dim, hidden_dim)
        self.mid_block = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, in_dim))

    def forward(self, x, t, condition):
        # 输入维度必须是对齐的 [Total_N, D]
        h = self.input_mlp(x) + self.time_mlp(t) + self.cond_mlp(condition)
        return self.mid_block(h)


# ================= 3. Best-of-N 训练逻辑 =================
# ================= 修复后的训练逻辑 =================
def train_best_of_n():
    Config.MODEL_SAVE_DIR.mkdir(exist_ok=True)
    dataset = CrossDomainUserDataset(Config.SOURCE_EMB_PATH, Config.TARGET_EMB_PATH)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    model = FlowMatchingNet(in_dim=Config.EMBEDDING_SIZE, cond_dim=Config.EMBEDDING_SIZE,
                            hidden_dim=Config.HIDDEN_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    print(f"[-] Start Training with Best-of-{Config.TRAIN_SAMPLES} Strategy + Cosine Repulsion...")

    # 调整权重：因为 Cosine 只有 0~1，所以权重可以稍微大一点，但不要太大
    LAMBDA_DIV = 0.5

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        total_mse = 0
        total_div = 0

        for source_emb, target_emb in dataloader:
            source_emb = source_emb.to(Config.DEVICE)
            target_emb = target_emb.to(Config.DEVICE)
            B = source_emb.shape[0]
            K = Config.TRAIN_SAMPLES
            D = Config.EMBEDDING_SIZE

            # === 数据扩充 ===
            target_K = target_emb.unsqueeze(1).repeat(1, K, 1)
            source_K_flat = source_emb.unsqueeze(1).repeat(1, K, 1).view(-1, D)

            x0 = torch.randn(B, K, D).to(Config.DEVICE)
            t = torch.rand(B, 1).to(Config.DEVICE)
            t_K = t.repeat(1, K)
            t_K_flat = t_K.view(-1)

            x_t = (1 - t_K.unsqueeze(-1)) * x0 + t_K.unsqueeze(-1) * target_K
            x_t_flat = x_t.view(-1, D)

            v_target = target_K - x0

            # === 模型预测 ===
            v_pred_flat = model(x_t_flat, t_K_flat, source_K_flat)
            v_pred = v_pred_flat.view(B, K, D)

            # === Loss 1: 准确性 (Best-of-N) ===
            mse_per_sample = torch.mean((v_pred - v_target) ** 2, dim=-1)  # [B, K]
            loss_best, _ = torch.min(mse_per_sample, dim=1)  # [B]
            loss_acc = torch.mean(loss_best)

            # === Loss 2: 多样性 (Cosine Repulsion) ===
            # 【关键修改】不要用方差，用余弦相似度
            # 1. 归一化预测向量 (只看方向，忽略模长，防止数值爆炸)
            v_norm = torch.nn.functional.normalize(v_pred, dim=-1)  # [B, K, D]

            # 2. 计算相似度矩阵: [B, K, K]
            # sim[b, i, j] 表示第 b 个用户的第 i 个生成结果和第 j 个生成结果的相似度
            sim_matrix = torch.bmm(v_norm, v_norm.transpose(1, 2))

            # 3. 惩罚非对角线元素
            # 我们希望 i != j 时，sim(i, j) 越小越好 (越接近 0 甚至 -1)
            # 对角线全是 1，我们不管它
            eye = torch.eye(K, device=Config.DEVICE).unsqueeze(0)  # [1, K, K]

            # 只取非对角线部分的平方和
            # 最小化这个值 = 强迫向量之间正交 (分开)
            loss_div = torch.mean((sim_matrix - eye) ** 2)

            # === 总 Loss ===
            loss = loss_acc + LAMBDA_DIV * loss_div

            optimizer.zero_grad()
            loss.backward()

            # 【保险措施】梯度裁剪，防止炸裂
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_mse += loss_acc.item()
            total_div += loss_div.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            avg_mse = total_mse / len(dataloader)
            avg_div = total_div / len(dataloader)

            print(f"Epoch {epoch + 1:03d} | Total: {avg_loss:.6f} | Acc: {avg_mse:.6f} | Div(Cos): {avg_div:.6f}")
            torch.save(model.state_dict(), Config.MODEL_SAVE_DIR / "flow_model_best_of_n_v3.pth")

if __name__ == "__main__":
    os.chdir("..")
    train_best_of_n()