import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# ==========================================
# 0. Hack RecBole (防止Windows单机报错)
# ==========================================
import torch.distributed

if not torch.distributed.is_initialized():
    torch.distributed.barrier = lambda *args, **kwargs: None


# ==========================================
# 1. GMFlow Head (生成器核心)
# ==========================================
class GMFlowHead(nn.Module):
    def __init__(self, hidden_dim, k_components=4):
        super().__init__()
        self.k = k_components
        self.hidden_dim = hidden_dim

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 融合层
        self.input_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU()
        )

        # 核心骨架
        self.body = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # 输出层
        self.head_means = nn.Linear(hidden_dim, k_components * hidden_dim)
        self.head_weights = nn.Linear(hidden_dim, k_components)

        # 共享方差网络
        self.s_network = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_t, t, condition):
        batch_size = x_t.size(0)
        t_emb = self.time_mlp(t)

        # 融合 Condition (SASRec输出) 和 x_t (噪声)
        fused_input = torch.cat([x_t, condition], dim=-1)
        h = self.input_fusion(fused_input) + t_emb
        h = self.body(h)

        # 预测参数
        means = self.head_means(h).view(batch_size, self.k, self.hidden_dim)
        # 归一化 Means，保证在单位球面上 (DreamRec 技巧：Embedding 约束)
        means = F.normalize(means, p=2, dim=-1)

        weight_logits = self.head_weights(h)

        s_input = torch.cat([condition, t], dim=-1)
        log_s = self.s_network(s_input)

        return means, weight_logits, log_s


# ==========================================
# 2. GMFlow Loss
# ==========================================
def gm_flow_loss(pred_means, pred_weight_logits, pred_log_s, target_x0):
    batch_size, k, dim = pred_means.shape
    target = target_x0.unsqueeze(1)

    # 欧氏距离平方
    dist_sq = torch.sum((target - pred_means) ** 2, dim=-1)

    # 方差约束
    s = F.softplus(pred_log_s) + 0.05
    s_sq = s ** 2

    log_weights = F.log_softmax(pred_weight_logits, dim=-1)
    log_gauss_exp = -0.5 * dist_sq / s_sq
    log_gauss_norm = -1.0 * dim * torch.log(s)

    log_prob_k = log_weights + log_gauss_exp + log_gauss_norm
    total_log_prob = torch.logsumexp(log_prob_k, dim=1)

    loss = -torch.mean(total_log_prob)
    return loss, s.mean()


# ==========================================
# 3. DreamGMFlow 主模型 (Approximator + Generator)
# ==========================================
class DreamGMFlow(nn.Module):
    def __init__(self, n_items, hidden_dim, k_components=4, max_seq_len=50):
        super().__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim

        # --- Shared Embeddings ---
        self.item_emb = nn.Embedding(n_items, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # --- Part 1: The Approximator (SASRec) ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, batch_first=True)
        self.approximator = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- Part 2: The Generator (GMFlow) ---
        self.generator = GMFlowHead(hidden_dim, k_components)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_normal_(module.weight)

    def get_seq_repr(self, item_seq):
        """SASRec 编码过程"""
        seq_len = item_seq.size(1)
        mask = (item_seq == 0)

        # 技巧：Normalize Embeddings
        # (DiffuRec 建议将 Embedding 约束在球面上，便于扩散)
        seq_emb = F.normalize(self.item_emb(item_seq), p=2, dim=-1)

        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)

        x = seq_emb + pos_emb
        x = self.layer_norm(x)

        # Transformer 输出 [B, L, D]
        output = self.approximator(x, src_key_padding_mask=mask)

        # 取最后一个 Token 作为序列表示
        # 关键修复：必须配合 Pre-padding 使用，这样 -1 才是最后一个真实物品
        seq_repr = output[:, -1, :]  # [B, D]

        # 归一化 Condition
        return F.normalize(seq_repr, p=2, dim=-1)

    def train_step(self, item_seq, target_item, lambda_gen=0.1):
        """
        联合训练：Approximator (CE) + Generator (NLL)
        """
        batch_size = item_seq.size(0)
        device = item_seq.device

        # 1. Approximator Forward
        condition = self.get_seq_repr(item_seq)  # [B, D]

        # 2. Approximator Loss (Standard SASRec Loss)
        # 这确保了 condition 本身具有很强的判别能力
        all_items = F.normalize(self.item_emb.weight, p=2, dim=-1)
        logits = torch.matmul(condition, all_items.t()) / 0.1  # Temp scaling
        loss_ce = F.cross_entropy(logits, target_item)

        # 3. Generator Forward (GMFlow)
        x0 = F.normalize(self.item_emb(target_item), p=2, dim=-1)

        t = torch.rand((batch_size, 1), device=device)
        noise = torch.randn_like(x0)
        x_t = (1 - t) * x0 + t * noise  # Optimal Transport Path

        pred_means, pred_w_logits, pred_log_s = self.generator(x_t, t, condition)

        # 4. Generator Loss
        loss_nll, mean_s = gm_flow_loss(pred_means, pred_w_logits, pred_log_s, x0)

        # 5. Total Loss (Loss Balancing)
        # [CRITICAL FIX] 重新平衡 Loss
        # NLL (-190) 太大了，CE (4.0) 太小了。
        # 我们给 CE 加权，并给 NLL 降权（除以 dim），防止方差抢跑。

        weight_ce = 10.0
        # 将 NLL 归一化到 per-dimension，使其数值量级降到 ~3.0 左右
        norm_nll = loss_nll / self.hidden_dim

        total_loss = weight_ce * loss_ce + lambda_gen * norm_nll

        return total_loss, loss_ce, loss_nll, mean_s

    @torch.no_grad()
    def full_sort_predict(self, item_seq, alpha=0.5):
        """
        DreamRec 的核心 Inference 技巧：Score Fusion
        Score = alpha * Approximator_Score + (1-alpha) * Generator_Score
        """
        batch_size = item_seq.size(0)
        device = item_seq.device

        # 1. Run Approximator (Get Base Scores)
        condition = self.get_seq_repr(item_seq)
        all_items = F.normalize(self.item_emb.weight, p=2, dim=-1)  # [N, D]

        # SASRec Score (Cosine Similarity / Dot Product)
        # [B, D] @ [D, N] -> [B, N]
        scores_approx = torch.matmul(condition, all_items.t())

        # 2. Run Generator (Get GMFlow Scores)
        # GMFlow: 1-step generation from noise
        x_T = torch.randn((batch_size, self.hidden_dim), device=device)
        t = torch.ones((batch_size, 1), device=device)

        means, w_logits, log_s = self.generator(x_T, t, condition)
        weights = torch.softmax(w_logits, dim=-1)  # [B, K]
        s = F.softplus(log_s) + 0.05  # [B, 1]

        # 计算 GMFlow 对全库物品的概率密度得分
        # Score_i = LogSumExp( log(A_k) - ||item_i - mu_k||^2 / 2s^2 )
        # 优化计算: ||A-B||^2 = ||A||^2 + ||B||^2 - 2AB

        k_comp = means.size(1)
        n_items = all_items.size(0)

        # 预计算
        item_norm_sq = torch.sum(all_items ** 2, dim=-1)  # [N] (Should be 1s)
        mean_norm_sq = torch.sum(means ** 2, dim=-1)  # [B, K]

        # Dot Product [B, K, N]
        means_reshaped = means.view(batch_size * k_comp, -1)
        dot_prod = torch.mm(means_reshaped, all_items.t()).view(batch_size, k_comp, n_items)

        # Dist Sq [B, K, N]
        dist_sq = mean_norm_sq.unsqueeze(-1) + item_norm_sq.unsqueeze(0).unsqueeze(0) - 2 * dot_prod

        # Log Prob
        s_sq = s.unsqueeze(2) ** 2
        log_weights = torch.log(weights + 1e-9).unsqueeze(2)
        log_gauss_part = -0.5 * dist_sq / s_sq

        # GMFlow Score [B, N]
        # 注意：这里得到的是 Log Probability (负数)
        # 为了和 Cosine Similarity (正数, -1~1) 融合，通常需要做标准化或者直接加权
        log_scores_gen = torch.logsumexp(log_weights + log_gauss_part, dim=1)

        # DreamRec Trick: 归一化分数以便融合
        # 简单的 Min-Max 归一化或者 Z-Score
        def normalize_score(score):
            mean = score.mean(dim=1, keepdim=True)
            std = score.std(dim=1, keepdim=True) + 1e-6
            return (score - mean) / std

        # [Ablation Support] 如果 alpha=1 或 0，直接返回原始分数
        if alpha >= 0.99:
            final_scores = scores_approx
        elif alpha <= 0.01:
            final_scores = log_scores_gen
        else:
            norm_approx = normalize_score(scores_approx)
            norm_gen = normalize_score(log_scores_gen)
            final_scores = alpha * norm_approx + (1 - alpha) * norm_gen

        # Mask padding
        final_scores[:, 0] = -float('inf')

        return final_scores


# ==========================================
# 4. 数据与评估工具
# ==========================================
class MockRecBoleDataset:
    def __init__(self, n_samples=1000, max_len=20, n_items=200):
        self.data = []
        # [CRITICAL UPDATE 2] Multi-Modal Data Generation
        # 为了测试 GMFlow 的多模态能力，我们构造 "分支" 数据
        # 规则：Seq=[..., n] -> Target 可能是 n+1 (50%) 或者 n+20 (50%)
        # 这种不确定性是 SASRec 难以完美处理的

        for _ in range(n_samples):
            seq_len = np.random.randint(5, max_len)

            # 保证 n+20 不越界
            start_id = np.random.randint(1, n_items - seq_len - 25)
            real_seq = [start_id + i for i in range(seq_len)]

            last_item = real_seq[-1]

            # 分支逻辑
            if np.random.rand() > 0.5:
                target = last_item + 1  # Path A: Next item
            else:
                target = last_item + 20  # Path B: Jump item

            # Pre-padding
            seq = [0] * (max_len - seq_len) + real_seq

            self.data.append({
                'item_id_list': torch.tensor(seq, dtype=torch.long),
                'item_id': torch.tensor(target, dtype=torch.long)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(dataset, batch_size=32, shuffle=True):
    from torch.utils.data import DataLoader
    def collate_fn(batch):
        item_id_list = torch.stack([item['item_id_list'] for item in batch])
        item_id = torch.stack([item['item_id'] for item in batch])
        return {'item_id_list': item_id_list, 'item_id': item_id}

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)


@torch.no_grad()
def evaluate_fusion(model, dataloader, device, k_metric=10, alpha=0.5, is_recbole=False):
    model.eval()
    recall_list = []
    ndcg_list = []
    check_limit = 20 if not is_recbole else float('inf')
    count = 0

    # print(f"  Evaluating (Alpha={alpha})...") # 移除冗余打印

    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        if is_recbole:
            seq = batch['item_id_list'].to(device)
            target = batch['item_id'].to(device)
        else:
            seq = batch['item_id_list'].to(device)
            target = batch['item_id'].to(device)

        batch_size = seq.size(0)

        # 使用融合预测
        scores = model.full_sort_predict(seq, alpha=alpha)

        _, topk_indices = torch.topk(scores, k=k_metric, dim=1)
        topk_indices = topk_indices.cpu().numpy()
        targets = target.cpu().numpy()

        for i in range(batch_size):
            pred_list = topk_indices[i]
            true_item = targets[i]

            if true_item in pred_list:
                recall_list.append(1.0)
                rank = np.where(pred_list == true_item)[0][0]
                ndcg_list.append(1.0 / np.log2(rank + 2))
            else:
                recall_list.append(0.0)
                ndcg_list.append(0.0)

        count += 1
        if count >= check_limit: break

    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)
    return avg_recall, avg_ndcg


# ==========================================
# 5. Main Training Loop
# ==========================================
def main():
    # 配置
    USE_RECBOLE = False
    N_ITEMS = 1000  # 稍微增加物品数，体现跳转难度
    HIDDEN_DIM = 64
    K_COMPONENTS = 4
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 1e-3
    MODEL_SAVE_PATH = "best_dream_gmflow.pth"
    # ALPHA = 0.6 # 移动到循环内部动态测试

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    test_dataloader = None
    if USE_RECBOLE:
        print("Loading MovieLens-1M via RecBole...")
        try:
            from recbole.quick_start import load_data_and_model
            from recbole.data import create_dataset, data_preparation
            from recbole.config import Config

            config_dict = {
                'model': 'SASRec',
                'dataset': 'ml-1m',
                'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
                'MAX_ITEM_LIST_LENGTH': 50,
                'train_neg_sample_args': None,
                'eval_batch_size': 256,
            }
            config = Config(model='SASRec', dataset='ml-1m', config_dict=config_dict)
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
            test_dataloader = test_data
            N_ITEMS = dataset.item_num
            print(f"Data loaded. Items: {N_ITEMS}")
        except ImportError:
            print("RecBole not found. Falling back to Mock Data.")
            USE_RECBOLE = False

    if not USE_RECBOLE:
        print(f"Generating Multi-Modal Mock Data (Items={N_ITEMS})...")
        print("Pattern: Target is either (Last+1) or (Last+20) with 50/50 prob.")
        # 增加样本量到 10000 保证覆盖率
        dataset = MockRecBoleDataset(n_samples=10000, n_items=N_ITEMS)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_data = get_dataloader(train_ds, batch_size=BATCH_SIZE)
        test_dataloader = get_dataloader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Model Init
    model = DreamGMFlow(N_ITEMS, HIDDEN_DIM, K_COMPONENTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nStarting Joint Training (DreamRec Architecture)...")
    print("Approximator (SASRec) + Generator (GMFlow)")
    best_recall = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss_acc = 0
        ce_acc = 0
        nll_acc = 0
        steps = 0

        for batch in train_data:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            if USE_RECBOLE:
                item_seq = batch['item_id_list'].to(device)
                target_item = batch['item_id'].to(device)
            else:
                item_seq = batch['item_id_list'].to(device)
                target_item = batch['item_id'].to(device)

            optimizer.zero_grad()

            # 联合训练 Loss
            loss, ce, nll, _ = model.train_step(item_seq, target_item, lambda_gen=0.1)

            loss.backward()
            optimizer.step()

            total_loss_acc += loss.item()
            ce_acc += ce.item()
            nll_acc += nll.item()
            steps += 1

        avg_loss = total_loss_acc / steps
        avg_ce = ce_acc / steps
        avg_nll = nll_acc / steps

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | NLL: {avg_nll:.4f}")

        # Ablation Eval
        if (epoch + 1) % 1 == 0:
            print("  [Ablation Test]")

            # 1. Pure SASRec (Alpha=1.0)
            r_sas, n_sas = evaluate_fusion(model, test_dataloader, device, k_metric=10, alpha=1.0,
                                           is_recbole=USE_RECBOLE)
            print(f"    SASRec Only (Alpha=1.0) : Recall={r_sas:.4f}, NDCG={n_sas:.4f}")

            # 2. Pure GMFlow (Alpha=0.0)
            r_gm, n_gm = evaluate_fusion(model, test_dataloader, device, k_metric=10, alpha=0.0, is_recbole=USE_RECBOLE)
            print(f"    GMFlow Only (Alpha=0.0) : Recall={r_gm:.4f}, NDCG={n_gm:.4f}")

            # 3. Fusion (Alpha=0.6)
            r_mix, n_mix = evaluate_fusion(model, test_dataloader, device, k_metric=10, alpha=0.6,
                                           is_recbole=USE_RECBOLE)
            print(f"    DreamRec    (Alpha=0.6) : Recall={r_mix:.4f}, NDCG={n_mix:.4f}")

            if r_mix > best_recall:
                best_recall = r_mix
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\nDone. Check the Ablation Test to see if Fusion helps.")


if __name__ == "__main__":
    main()