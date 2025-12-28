import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import math


# ================= 配置 =================
class Config:
    BASE_DIR = Path("D:/pyprojects/recommend-study/studybole/dataset/Amazon_Processed_10Core")
    EMB_DIR = BASE_DIR / "Pretrained_Embeddings"

    # 路径
    SOURCE_USER_PATH = EMB_DIR / "Source_Books_user_emb.pkl"
    TARGET_ITEM_PATH = EMB_DIR / "Target_Movies_item_emb.pkl"
    TARGET_INTER_PATH = BASE_DIR / "Target_Movies/Target_Movies.inter"
    MODEL_SAVE_DIR = Path("./saved_models")

    # 参数
    EMBEDDING_SIZE = 64
    HIDDEN_DIM = 256
    BATCH_SIZE = 1024
    LR = 1e-3
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 【关键】训练时，每个用户采样多少个真实物品作为 Target？
    # 建议设为 1 或 5。设为 5 意味着一次 Batch 训练覆盖用户的 5 个兴趣点
    SAMPLES_PER_USER = 5


# ================= 新的数据集: User -> Items =================
class CrossDomainItemDataset(Dataset):
    def __init__(self, cfg):
        print("[-] Loading Source User Embeddings...")
        with open(cfg.SOURCE_USER_PATH, 'rb') as f:
            self.src_user_emb = pickle.load(f)  # {uid: vec}

        print("[-] Loading Target Item Embeddings...")
        with open(cfg.TARGET_ITEM_PATH, 'rb') as f:
            self.tgt_item_emb = pickle.load(f)  # {iid: vec}

        print("[-] Loading Interactions...")
        df = pd.read_csv(cfg.TARGET_INTER_PATH, sep='\t', dtype=str)
        # 识别列名
        uid_col = [c for c in df.columns if 'user_id' in c][0]
        iid_col = [c for c in df.columns if 'item_id' in c][0]

        # 构建 User -> List[Items] 映射
        # 这一步只保留重叠用户且有 Item Embedding 的记录
        self.user_history = {}

        # 预先过滤有效的 Item ID
        valid_item_ids = set(self.tgt_item_emb.keys())
        valid_user_ids = set(self.src_user_emb.keys())

        grouped = df.groupby(uid_col)[iid_col].apply(list)

        count = 0
        for uid, items in tqdm(grouped.items(), desc="Aligning Data"):
            if uid in valid_user_ids:
                # 过滤掉没有 Embedding 的 Item
                valid_items = [i for i in items if i in valid_item_ids]
                if len(valid_items) > 0:
                    self.user_history[uid] = valid_items
                    count += 1

        self.user_ids = list(self.user_history.keys())
        print(f"[-] Dataset Ready. {count} users found with valid cross-domain history.")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]

        # 1. Source Condition
        src_vec = self.src_user_emb[uid]

        # 2. Target Sampling (核心逻辑)
        # 从该用户的历史物品中，随机采样 K 个物品向量
        # 如果历史物品不足 K 个，就允许重复采样 (Replacement=True)
        history_items = self.user_history[uid]

        # 随机选择索引
        indices = np.random.choice(len(history_items), size=Config.SAMPLES_PER_USER, replace=True)
        sampled_iids = [history_items[i] for i in indices]

        # 获取 Item Vectors
        tgt_vecs = np.array([self.tgt_item_emb[iid] for iid in sampled_iids])

        return torch.tensor(src_vec, dtype=torch.float32), \
            torch.tensor(tgt_vecs, dtype=torch.float32)


# ================= 模型 (Flow Matching) =================
# ... 保持你之前的 FlowMatchingNet 代码不变 ...
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
        h = self.input_mlp(x) + self.time_mlp(t) + self.cond_mlp(condition)
        return self.mid_block(h)