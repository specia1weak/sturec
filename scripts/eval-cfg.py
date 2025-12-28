import math
import os

import torch
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch.nn as nn
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


class Config:
    # 路径需一致
    BASE_DIR = Path("D:/pyprojects/recommend-study/studybole/dataset/Amazon_Processed_10Core")
    EMB_DIR = BASE_DIR / "Pretrained_Embeddings"
    SOURCE_USER_PATH = EMB_DIR / "Source_Books_user_emb.pkl"
    TARGET_ITEM_PATH = EMB_DIR / "Target_Movies_item_emb.pkl"
    TARGET_INTER_PATH = BASE_DIR / "Target_Movies/Target_Movies.inter"

    MODEL_PATH = Path("./saved_models/flow_model_cfg.pth")

    # 推理参数
    NUM_SAMPLES = 20  # 20 个样本
    GUIDANCE_SCALE = 3.0  # CFG 强度 (核心超参，可调 1.5 - 4.0)
    BATCH_SIZE = 64
    TOP_K = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EMBEDDING_SIZE = 64
    HIDDEN_DIM = 256


def load_ground_truth(inter_path):
    print(f"[-] Loading Ground Truth...")
    df = pd.read_csv(inter_path, sep='\t', dtype=str)
    uid_col = [c for c in df.columns if 'user_id' in c][0]
    iid_col = [c for c in df.columns if 'item_id' in c][0]
    groups = df.groupby(uid_col)[iid_col].apply(set)
    return groups.to_dict()


def get_metrics(sorted_items, ground_truth, k=20):
    hit = 0
    idcg = 0
    dcg = 0
    rank_list = sorted_items[:k]
    for i, item_id in enumerate(rank_list):
        if item_id in ground_truth:
            hit += 1
            dcg += 1.0 / np.log2(i + 2)
    num_pos = len(ground_truth)
    ideal_len = min(num_pos, k)
    for i in range(ideal_len): idcg += 1.0 / np.log2(i + 2)
    recall = hit / num_pos if num_pos > 0 else 0
    ndcg = dcg / idcg if idcg > 0 else 0
    return recall, ndcg


@torch.no_grad()
def evaluate_cfg():
    print(f"[-] Loading Model CFG (Scale={Config.GUIDANCE_SCALE})...")
    model = FlowMatchingNetCFG(Config.EMBEDDING_SIZE, Config.EMBEDDING_SIZE, Config.HIDDEN_DIM).to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()

    with open(Config.SOURCE_USER_PATH, 'rb') as f:
        src_user_emb = pickle.load(f)
    with open(Config.TARGET_ITEM_PATH, 'rb') as f:
        tgt_item_emb = pickle.load(f)
    ground_truth = load_ground_truth(Config.TARGET_INTER_PATH)

    item_ids = list(tgt_item_emb.keys())
    item_vecs = np.array([tgt_item_emb[k] for k in item_ids])
    item_tensor = torch.tensor(item_vecs, dtype=torch.float32).to(Config.DEVICE)

    # 找到重叠测试用户
    test_users = list(set(src_user_emb.keys()) & set(ground_truth.keys()))
    test_users.sort()

    print(f"[-] Users to Eval: {len(test_users)}")

    metrics = {'recall': [], 'ndcg': []}
    batch_size = Config.BATCH_SIZE
    num_batches = (len(test_users) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Evaluating"):
        batch_users = test_users[i * batch_size: (i + 1) * batch_size]

        # 1. 构造输入 (Batch * 20)
        batch_src_vecs = []
        for u in batch_users:
            vec = src_user_emb[u]  # [D]
            # 复制 NUM_SAMPLES 份
            batch_src_vecs.append(np.tile(vec, (Config.NUM_SAMPLES, 1)))

        # Shape: [B * 20, 64]
        cond_input = torch.tensor(np.vstack(batch_src_vecs), dtype=torch.float32).to(Config.DEVICE)

        # 噪声
        x_curr = torch.randn_like(cond_input)

        # 2. CFG 推理
        # t = 0 (Euler 1-step)
        t_zeros = torch.zeros(x_curr.shape[0]).to(Config.DEVICE)

        # (A) 有条件预测
        # mask = 1
        mask_keep = torch.ones(x_curr.shape[0]).to(Config.DEVICE)
        v_cond = model(x_curr, t_zeros, cond_input, cond_mask=mask_keep)

        # (B) 无条件预测
        # mask = 0 (模型会自动使用 null_cond)
        mask_drop = torch.zeros(x_curr.shape[0]).to(Config.DEVICE)
        v_uncond = model(x_curr, t_zeros, cond_input, cond_mask=mask_drop)

        # (C) CFG 公式
        velocity = v_uncond + Config.GUIDANCE_SCALE * (v_cond - v_uncond)

        # 3. 更新位置 (x1 = x0 + v * 1)
        x_final = x_curr + velocity * 1.0

        # 4. 检索 (Max Pooling)
        x_final = x_final.view(len(batch_users), Config.NUM_SAMPLES, Config.EMBEDDING_SIZE)

        for u_idx, u_id in enumerate(batch_users):
            user_vectors = x_final[u_idx]  # [20, 64]

            # [20, Num_Items]
            scores_20 = torch.matmul(user_vectors, item_tensor.T)

            # 取 20 个样本中最大的分数
            final_scores, _ = torch.max(scores_20, dim=0)

            _, topk_indices = torch.topk(final_scores, Config.TOP_K)
            topk_items = [item_ids[idx] for idx in topk_indices.cpu().numpy()]

            r, n = get_metrics(topk_items, ground_truth[u_id], k=Config.TOP_K)
            metrics['recall'].append(r)
            metrics['ndcg'].append(n)

    print(f"\n[CFG Result | Scale={Config.GUIDANCE_SCALE}]")
    print(f"Recall@20: {np.mean(metrics['recall']):.4f}")
    print(f"NDCG@20  : {np.mean(metrics['ndcg']):.4f}")


if __name__ == "__main__":
    os.chdir("..")
    evaluate_cfg()