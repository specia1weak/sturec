import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # x: [B] -> [B, 1] -> [B, Dim]
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowMatchingNetCFG(nn.Module):
    def __init__(self, in_dim=64, cond_dim=64, hidden_dim=256):
        super().__init__()

        # 1. 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. 条件编码 (User Emb)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. 输入编码 (Noisy Item)
        self.input_mlp = nn.Linear(in_dim, hidden_dim)

        # 4. 主干网络 (ResNet-like Block)
        self.mid_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim)  # 输出速度向量 v
        )

        # === 核心修改: Null Condition (空条件) ===
        # 用于 Classifier-Free Guidance 的无条件生成
        self.null_cond = nn.Parameter(torch.randn(1, cond_dim))

    def forward(self, x, t, condition, cond_mask=None):
        """
        x: [B, D] 当前状态
        t: [B] 时间
        condition: [B, D] 用户 Embedding
        cond_mask: [B] 1.0 = Keep Condition, 0.0 = Drop Condition (Use Null)
        """
        # 处理条件 Drop
        if cond_mask is not None:
            # [B] -> [B, 1]
            mask = cond_mask.view(-1, 1)
            # 广播 Null Condition [1, D] -> [B, D]
            # 如果 mask=1，保留 condition；如果 mask=0，替换为 null_cond
            condition = condition * mask + self.null_cond * (1 - mask)

        t_emb = self.time_mlp(t)  # [B, H]
        c_emb = self.cond_mlp(condition)  # [B, H]
        x_emb = self.input_mlp(x)  # [B, H]

        # 融合
        h = x_emb + t_emb + c_emb
        velocity = self.mid_block(h)

        return velocity


import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm

# ================= 配置 =================
class Config:
    BASE_DIR = Path("D:/pyprojects/recommend-study/studybole/dataset/Amazon_Processed_10Core")
    PRETRAIN_DIR = BASE_DIR / "Pretrained_Embeddings"
    SOURCE_EMB_PATH = PRETRAIN_DIR / "Source_Books_user_emb.pkl"
    TARGET_EMB_PATH = PRETRAIN_DIR / "Target_Movies_item_emb.pkl"  # Item Emb
    TARGET_INTER_PATH = BASE_DIR / "Target_Movies/Target_Movies.inter"

    MODEL_SAVE_DIR = Path("./saved_models")

    EMBEDDING_SIZE = 64
    HIDDEN_DIM = 256
    BATCH_SIZE = 1024
    LR = 1e-3
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # CFG 参数
    DROP_PROB = 0.1  # 10% 概率丢弃条件


# ================= Dataset (Item-based) =================
class CrossDomainItemDataset(Dataset):
    def __init__(self, source_user_path, target_item_path, target_inter_path):
        print(f"[-] Loading Embeddings & Interactions...")
        with open(source_user_path, 'rb') as f:
            self.source_user_dict = pickle.load(f)
        with open(target_item_path, 'rb') as f:
            self.target_item_dict = pickle.load(f)

        # 强制 Key 转字符串
        self.source_user_dict = {str(k).strip(): v for k, v in self.source_user_dict.items()}
        self.target_item_dict = {str(k).strip(): v for k, v in self.target_item_dict.items()}

        # 加载交互
        import pandas as pd
        df = pd.read_csv(target_inter_path, sep='\t', dtype=str)
        uid_col = [c for c in df.columns if 'user_id' in c][0]
        iid_col = [c for c in df.columns if 'item_id' in c][0]

        self.data_pairs = []
        valid_users = set(self.source_user_dict.keys())
        valid_items = set(self.target_item_dict.keys())

        print("[-] Filtering pairs...")
        for _, row in df.iterrows():
            uid = row[uid_col].strip()
            iid = row[iid_col].strip()
            if uid in valid_users and iid in valid_items:
                # 存 Key，省内存
                self.data_pairs.append((uid, iid))

        print(f"[-] Training Pairs: {len(self.data_pairs)}")
        if len(self.data_pairs) == 0: raise RuntimeError("No pairs found!")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        uid, iid = self.data_pairs[idx]
        return (torch.tensor(self.source_user_dict[uid], dtype=torch.float32),
                torch.tensor(self.target_item_dict[iid], dtype=torch.float32))


# ================= 训练循环 =================
def train_cfg():
    Config.MODEL_SAVE_DIR.mkdir(exist_ok=True)

    # 1. 数据
    dataset = CrossDomainItemDataset(Config.SOURCE_EMB_PATH, Config.TARGET_EMB_PATH, Config.TARGET_INTER_PATH)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    # 2. 模型
    model = FlowMatchingNetCFG(Config.EMBEDDING_SIZE, Config.EMBEDDING_SIZE, Config.HIDDEN_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    print(f"[-] Start Training CFG Flow Matching (Drop Rate: {Config.DROP_PROB})...")

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0

        for source_emb, target_emb in dataloader:
            source_emb = source_emb.to(Config.DEVICE)  # Condition
            target_emb = target_emb.to(Config.DEVICE)  # Target (Item)
            B = source_emb.shape[0]

            # --- CFG Mask 生成 ---
            # 1 = Keep, 0 = Drop
            # torch.bernoulli 生成 0/1 分布
            keep_mask = torch.bernoulli(torch.full((B,), 1 - Config.DROP_PROB)).to(Config.DEVICE)

            # --- Flow Matching 流程 ---
            # 1. 采样 x0 (Noise)
            x0 = torch.randn_like(target_emb)

            # 2. 采样 t
            t = torch.rand(B).to(Config.DEVICE)

            # 3. 插值 x_t (Rectified Flow)
            t_expand = t.view(-1, 1)
            x_t = (1 - t_expand) * x0 + t_expand * target_emb

            # 4. 目标速度 v = x1 - x0
            v_target = target_emb - x0

            # 5. 预测 (带 Mask)
            v_pred = model(x_t, t, source_emb, cond_mask=keep_mask)

            # 6. Loss (MSE)
            loss = torch.mean((v_pred - v_target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | CFG Loss: {avg_loss:.6f}")
            torch.save(model.state_dict(), Config.MODEL_SAVE_DIR / "flow_model_cfg.pth")


if __name__ == "__main__":
    os.chdir("..")
    train_cfg()