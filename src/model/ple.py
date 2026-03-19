import torch.nn as nn
import torch
from tqdm import tqdm


class PLEFramework(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64):
        super(PLEFramework, self).__init__()

        # Embedding
        self.user_embedding = nn.Embedding(n_users, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)

        # [模拟 PLE 核心层]
        # 输入维度: embed_dim * 2 (user + item)
        # 输出维度: embed_dim * 2 (对应两个 Task 的表达)
        # TODO: 未来这里替换为 CGC/PLE Layer
        self.ple_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim * 2)
        )

        # [Towers] Task-Specific Heads
        # 假设 Task 0 (Book) 和 Task 1 (Movie)
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(2)
        ])

    def forward(self, user_id, item_id):
        # 1. Embedding
        u_emb = self.user_embedding(user_id)
        i_emb = self.item_embedding(item_id)
        x = torch.cat([u_emb, i_emb], dim=-1)  # [B, 128]

        # 2. PLE Extraction
        ple_out = self.ple_layer(x)  # [B, 128]

        # 3. Split for Tasks
        # 简单切分模拟多任务输出 (实际 PLE 会有不同的 Gate 输出)
        # 假设前64维给 Task A，后64维给 Task B
        split_dim = ple_out.shape[1] // 2
        task_reps = [ple_out[:, :split_dim], ple_out[:, split_dim:]]

        # 4. Towers
        preds = []
        for i, tower in enumerate(self.towers):
            preds.append(tower(task_reps[i]))

        return preds  # 返回 [Pred_Task0, Pred_Task1]

def train_one_epoch(model, dataloader, optimizer, device, domain_item_dict):
    model.train()
    total_loss = 0
    steps = 0

    # 初始化 Loss 函数
    loss_fn = nn.BCELoss(reduction='none')
    pbar = tqdm(dataloader, desc="  Training", leave=False, ncols=100)

    for batch in pbar:
        user = batch['user'].to(device)
        pos_item = batch['item'].to(device)
        domain = batch['domain'].to(device)
        
        # 构造正样本 Label (全1)
        pos_label = torch.ones_like(batch['label']).to(device).view(-1, 1)
        
        # --- Domain-Aware 负采样 ---
        # 预先分配负样本 Tensor
        neg_item = torch.zeros_like(pos_item)
        
        for task_id in range(2):
            # 找出当前 Batch 中属于该 Task 的样本
            task_mask = (domain == task_id)
            count = task_mask.sum().item()
            if count > 0:
                # 获取该 Domain 的候选集
                candidates = domain_item_dict[task_id]
                # 随机采样索引
                rand_idx = torch.randint(0, len(candidates), (count,), device=device)
                # 填入 neg_item
                neg_item[task_mask] = candidates[rand_idx]

        # Debug Print (First Batch)
        if steps == 0:
            print(f"\n[Debug] Batch 0 Check:")
            print(f"  Pos Items (First 5): {pos_item[:5].tolist()}")
            print(f"  Neg Items (First 5): {neg_item[:5].tolist()}")
            # 简单验证 Domain
            # 注意：这里我们没办法直接反查 Item->Domain，但可以通过 domain_item_dict 验证
            # 只要 neg_item 都在 candidates 里就是对的。逻辑上上面填入的就是对的。

        # 构造负样本 Label (全0)
        neg_label = torch.zeros_like(batch['label']).to(device).view(-1, 1)

        # 前向传播 - 正样本
        pos_preds = model(user, pos_item)
        
        # 前向传播 - 负样本
        neg_preds = model(user, neg_item)

        batch_loss = 0
        
        # 遍历 Task (Book=0, Movie=1)
        for task_id in range(2):
            # 1. 生成掩码
            mask = (domain == task_id).float().view(-1, 1)
            
            if mask.sum() == 0:
                continue
            
            # 2. 获取预测值
            current_pos_pred = pos_preds[task_id]
            current_neg_pred = neg_preds[task_id]
            
            # 3. 计算 Loss
            loss_pos = loss_fn(current_pos_pred, pos_label) * mask
            loss_neg = loss_fn(current_neg_pred, neg_label) * mask
            
            # 4. 平均
            total_task_loss = loss_pos.sum() + loss_neg.sum()
            batch_loss += total_task_loss / (mask.sum() * 2 + 1e-8)

        # 反向传播
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        steps += 1

    return total_loss / steps




from sklearn.metrics import roc_auc_score, log_loss
import numpy as np


def evaluate_model(model, dataloader, n_items, device, domain_item_dict, top_k=10, neg_k=99):
    """
    Top-K 评估函数 (1:N Domain-Aware 负采样)
    计算 AUC, HR@K, NDCG@K
    
    [OPTIMIZED] 使用向量化操作替代逐样本 Python 循环
    """
    model.eval()

    # 初始化指标存储
    # Task 0: Book, Task 1: Movie
    metrics_sum = {
        0: {'AUC': 0.0, 'HR': 0.0, 'NDCG': 0.0, 'Count': 0},
        1: {'AUC': 0.0, 'HR': 0.0, 'NDCG': 0.0, 'Count': 0}
    }
    
    total_batches = len(dataloader)
    pbar = tqdm(dataloader, desc="  Evaluating", leave=False, ncols=100)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            user = batch['user'].to(device)
            pos_item = batch['item'].to(device)
            domain = batch['domain'].to(device)
            batch_size = user.size(0)

            # === 1. 正样本打分 ===
            pos_preds = model(user, pos_item)  # list [ [B, 1], [B, 1] ]

            # === 2. 负样本打分 (1:99 Domain-Aware) ===
            neg_items = torch.zeros((batch_size, neg_k), dtype=torch.long, device=device)
            
            for task_id in range(2):
                mask = (domain == task_id)
                count = mask.sum().item()
                if count > 0:
                    candidates = domain_item_dict[task_id]
                    rand_idx = torch.randint(0, len(candidates), (count, neg_k), device=device)
                    neg_items[mask] = candidates[rand_idx]

            # 展平以便通过模型 [B * neg_k]
            flat_users = user.unsqueeze(1).repeat(1, neg_k).view(-1)
            flat_neg_items = neg_items.view(-1)
            
            flat_neg_preds = model(flat_users, flat_neg_items)  # list [ [B*neg_k, 1], ... ]

            # === 3. 逐任务计算指标 (向量化) ===
            for task_id in range(2):
                task_mask = (domain == task_id)
                n_valid = task_mask.sum().item()
                if n_valid == 0:
                    continue
                
                # 取出正样本分数 [N_valid, 1] -> [N_valid]
                scores_pos = pos_preds[task_id][task_mask].squeeze(-1)
                
                # 取出负样本分数 [N_valid, neg_k]
                scores_neg = flat_neg_preds[task_id].view(batch_size, neg_k)[task_mask]
                
                # 转为 NumPy
                p_pos = scores_pos.cpu().numpy()  # [N]
                p_neg = scores_neg.cpu().numpy()  # [N, neg_k]
                
                # === 向量化计算 AUC ===
                # AUC = P(正样本得分 > 随机负样本得分)
                # 对每个样本：计算有多少负样本得分 < 正样本得分
                # AUC = (正样本击败的负样本数) / neg_k
                pos_expanded = p_pos[:, np.newaxis]  # [N, 1]
                wins = (pos_expanded > p_neg).sum(axis=1)  # [N]
                ties = (pos_expanded == p_neg).sum(axis=1)  # [N]
                auc_scores = (wins + 0.5 * ties) / neg_k  # [N]
                
                # === 向量化计算 Ranking Metrics ===
                # rank = 1 + (负样本得分 > 正样本得分).sum()
                ranks = 1 + (p_neg > pos_expanded).sum(axis=1)  # [N]
                
                # HR@K: 1 if rank <= top_k else 0
                hr_scores = (ranks <= top_k).astype(float)  # [N]
                
                # NDCG@K: 1/log2(rank+1) if rank <= top_k else 0
                ndcg_scores = np.where(ranks <= top_k, 1.0 / np.log2(ranks + 1), 0.0)  # [N]
                
                # 累加
                metrics_sum[task_id]['AUC'] += auc_scores.sum()
                metrics_sum[task_id]['HR'] += hr_scores.sum()
                metrics_sum[task_id]['NDCG'] += ndcg_scores.sum()
                metrics_sum[task_id]['Count'] += n_valid
            
            # 每 10 个 batch 打印一次进度详情
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                book_count = metrics_sum[0]['Count']
                movie_count = metrics_sum[1]['Count']
                pbar.set_postfix({
                    'Book': book_count, 
                    'Movie': movie_count
                })

    # === 平均指标 ===
    final_metrics = {}
    task_names = {0: 'Book', 1: 'Movie'}
    
    print(f"\n  [Eval Summary] Book samples: {metrics_sum[0]['Count']}, Movie samples: {metrics_sum[1]['Count']}")
    
    for task_id, name in task_names.items():
        count = metrics_sum[task_id]['Count']
        if count == 0:
            final_metrics[name] = {'AUC': 0.0, 'HR@10': 0.0, 'NDCG@10': 0.0}
        else:
            final_metrics[name] = {
                'AUC': metrics_sum[task_id]['AUC'] / count,
                'HR@10': metrics_sum[task_id]['HR'] / count,
                'NDCG@10': metrics_sum[task_id]['NDCG'] / count
            }
            
    return final_metrics