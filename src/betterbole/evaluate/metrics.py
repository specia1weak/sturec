import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

# ==================== 指标基类 ====================
class BaseMetric:
    @classmethod
    def calculate(cls, *args, **kwargs):
        raise NotImplementedError

# ==================== PointWise 指标簇 ====================
class AUC(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.0

class LogLoss(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        return log_loss(y_true, y_pred)

class GAUC(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, users, **kwargs):
        user_data = {}
        for u, t, p in zip(users, y_true, y_pred):
            user_data.setdefault(u, {'t': [], 'p': []})
            user_data[u]['t'].append(t)
            user_data[u]['p'].append(p)

        total_weight, weighted_auc = 0.0, 0.0
        for data in user_data.values():
            t_arr, p_arr = np.array(data['t']), np.array(data['p'])
            weight = len(t_arr)
            if len(set(t_arr)) > 1:
                try:
                    weighted_auc += roc_auc_score(t_arr, p_arr) * weight
                    total_weight += weight
                except ValueError:
                    continue
        return weighted_auc / total_weight if total_weight > 0 else 0.0

# ==================== TopK 指标簇 ====================
class HR(BaseMetric):
    @classmethod
    def calculate(cls, hits_k, **kwargs):
        # hits_k shape: [Batch_size, K]
        return hits_k.sum().item()

class NDCG(BaseMetric):
    @classmethod
    def calculate(cls, hits_k, k, **kwargs):
        # 针对 Leave-One-Out 的极速 NDCG 计算
        device = hits_k.device
        ranks = torch.arange(1, k + 1, device=device).float()
        dcg_weights = 1.0 / torch.log2(ranks + 1)
        return (hits_k * dcg_weights).sum().item()

# 注册表：将字符串映射到对应的计算类
METRIC_REGISTRY = {
    "auc": AUC, "logloss": LogLoss, "gauc": GAUC,
    "hr": HR, "ndcg": NDCG
}