import os

import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import math


# ================= 配置部分 =================
class Config:
    # 路径需与你实际环境一致
    BASE_DIR = Path("D:/pyprojects/recommend-study/studybole/dataset/Amazon_Processed_10Core")
    EMB_DIR = BASE_DIR / "Pretrained_Embeddings"

    # 输入：源域用户 Emb
    SOURCE_USER_PATH = EMB_DIR / "Source_Books_user_emb.pkl"
    # 检索库：目标域物品 Emb
    TARGET_ITEM_PATH = EMB_DIR / "Target_Movies_item_emb.pkl"
    # Ground Truth：目标域交互文件
    TARGET_INTER_PATH = BASE_DIR / "Target_Movies/Target_Movies.inter"

    # 模型路径
    # MODEL_PATH = Path("./saved_models/flow_model_latest.pth")
    # MODEL_PATH = Path("./saved_models/flow_model_best_of_n.pth")
    # MODEL_PATH = Path("./saved_models/flow_model_best_of_n_v3.pth")
    MODEL_PATH = Path("./saved_models/flow_item_dist.pth")

    # 推理参数
    NUM_SAMPLES = 5  # 你的核心 Idea: 一次生成 20 个
    BATCH_SIZE = 64  # 用户批次大小
    TOP_K = 20  # 评估指标 K
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型结构参数 (需与训练时一致)
    EMBEDDING_SIZE = 64
    HIDDEN_DIM = 256


# ================= 模型定义 (必须与训练一致) =================
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
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(condition)
        x_emb = self.input_mlp(x)
        h = x_emb + t_emb + c_emb
        return self.mid_block(h)


# ================= 工具函数 =================
def load_ground_truth(inter_path):
    """读取交互文件，构建 {raw_user_id: set(raw_item_ids)}"""
    print(f"[-] Loading Ground Truth from {inter_path}...")
    df = pd.read_csv(inter_path, sep='\t', dtype=str)

    # 识别列名
    uid_col = [c for c in df.columns if 'user_id' in c][0]
    iid_col = [c for c in df.columns if 'item_id' in c][0]

    gt = {}
    # Groupby 稍微慢一点，但代码清晰
    # 为了速度，转换成 dict
    groups = df.groupby(uid_col)[iid_col].apply(set)
    gt = groups.to_dict()

    print(f"    Loaded {len(gt)} users with history.")
    return gt


def get_metrics(sorted_items, ground_truth, k=20):
    """计算单个用户的 Recall@K 和 NDCG@K"""
    hit = 0
    idcg = 0
    dcg = 0

    # 截取 Top-K
    rank_list = sorted_items[:k]

    for i, item_id in enumerate(rank_list):
        if item_id in ground_truth:
            hit += 1
            dcg += 1.0 / np.log2(i + 2)

    # 计算 IDCG (理想情况下的 DCG)
    # 理想情况是把所有 GT 都排在最前面
    num_pos = len(ground_truth)
    ideal_len = min(num_pos, k)
    for i in range(ideal_len):
        idcg += 1.0 / np.log2(i + 2)

    recall = hit / num_pos if num_pos > 0 else 0
    ndcg = dcg / idcg if idcg > 0 else 0
    return recall, ndcg


# ================= 核心推理 =================
@torch.no_grad()
def evaluate():
    # 1. 加载资源
    print("[-] Loading model and embeddings...")
    model = FlowMatchingNet(Config.EMBEDDING_SIZE, Config.EMBEDDING_SIZE, Config.HIDDEN_DIM).to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()

    with open(Config.SOURCE_USER_PATH, 'rb') as f:
        src_user_emb = pickle.load(f)
    with open(Config.TARGET_ITEM_PATH, 'rb') as f:
        tgt_item_emb = pickle.load(f)  # {item_id: vector}

    ground_truth = load_ground_truth(Config.TARGET_INTER_PATH)

    # 准备 Item Pool (转换为 Tensor 以便矩阵运算)
    item_ids = list(tgt_item_emb.keys())
    item_vecs = np.array([tgt_item_emb[k] for k in item_ids])
    item_tensor = torch.tensor(item_vecs, dtype=torch.float32).to(Config.DEVICE)
    # 归一化 Item，方便算 Cosine (可选，如果 Embedding 已经是 Normalized 可跳过，LightGCN 通常未归一化)
    # 这里建议做一下归一化，内积变 Cosine，通常更稳
    # item_tensor = torch.nn.functional.normalize(item_tensor, dim=1)

    # 确定测试用户集 (必须是既有 Embedding 又有 GT 的用户)
    test_users = list(set(src_user_emb.keys()) & set(ground_truth.keys()))
    test_users.sort()

    print(f"[-] Starting Evaluation on {len(test_users)} overlapping users...")
    print(f"    Set Generation Strategy: Generate {Config.NUM_SAMPLES} vectors per user.")

    metrics = {'recall': [], 'ndcg': []}

    # 2. 批量推理
    batch_size = Config.BATCH_SIZE
    num_batches = (len(test_users) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Evaluating"):
        batch_users = test_users[i * batch_size: (i + 1) * batch_size]

        # 构造 Batch 输入
        batch_src_vecs = []
        for u in batch_users:
            # 复制 NUM_SAMPLES 份
            vec = src_user_emb[u]
            batch_src_vecs.append(np.tile(vec, (Config.NUM_SAMPLES, 1)))

        # Shape: [B * 20, 64]
        batch_input_cond = torch.tensor(np.vstack(batch_src_vecs), dtype=torch.float32).to(Config.DEVICE)

        # 采样噪声
        x_curr = torch.randn_like(batch_input_cond)

        # Flow Matching 生成 (Euler 1-Step)
        # 从 t=0 到 t=1
        # 去掉那个 1
        t_zeros = torch.zeros(x_curr.shape[0]).to(Config.DEVICE)
        velocity = model(x_curr, t_zeros, batch_input_cond)
        x_final = x_curr + velocity * 1.0

        # 此时 x_final 包含了 [u1_1, u1_2... u1_20, u2_1...]

        # 3. 检索与打分
        # 为了显存安全，我们还是逐个用户处理打分矩阵
        # x_final reshape: [Batch, 20, 64]
        x_final = x_final.view(len(batch_users), Config.NUM_SAMPLES, Config.EMBEDDING_SIZE)

        for u_idx, u_id in enumerate(batch_users):
            # 取出该用户的 20 个生成向量 [20, 64]
            user_vectors = x_final[u_idx]

            # 计算分数: [20, Num_Items]
            # score = user_vectors @ item_tensor.T
            scores_20 = torch.matmul(user_vectors, item_tensor.T)

            # === 关键步骤：Max Pooling ===
            # 对于每个物品，取 20 个生成向量中给出的最高分
            # 逻辑：只要这 20 个猜测中有一个猜中了这个物品，分就应该高
            final_scores, _ = torch.max(scores_20, dim=0)  # [Num_Items]

            # Top-K
            _, topk_indices = torch.topk(final_scores, Config.TOP_K)
            topk_items = [item_ids[idx] for idx in topk_indices.cpu().numpy()]

            # 计算指标
            r, n = get_metrics(topk_items, ground_truth[u_id], k=Config.TOP_K)
            metrics['recall'].append(r)
            metrics['ndcg'].append(n)

    # 4. 汇总结果
    avg_recall = np.mean(metrics['recall'])
    avg_ndcg = np.mean(metrics['ndcg'])

    print("\n[=== Evaluation Result ===]")
    print(f"Recall@{Config.TOP_K}: {avg_recall:.4f}")
    print(f"NDCG@{Config.TOP_K}  : {avg_ndcg:.4f}")
    print("===========================")


if __name__ == "__main__":
    os.chdir("..")
    evaluate()