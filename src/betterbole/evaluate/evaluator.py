import json
from typing import Iterable, Union, Literal

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import torch
class BaseEvaluator:
    VALID_METRICS = ["auc", "logloss", "gauc", "hr", "ndcg", "recall"]
    def __init__(self, metrics):
        # 统一转小写并过滤非法指标
        self.metrics = [m.lower() for m in metrics]
        for m in self.metrics:
            base_m = m.split('@')[0] if '@' in m else m
            assert base_m in self.VALID_METRICS, f"不支持的指标: {m}"
        self.clear()

    def collect(self, batch_users, batch_targets, batch_preds):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


import numpy as np
from betterbole.evaluate.metrics import METRIC_REGISTRY

class PointWiseEvaluator(BaseEvaluator):
    def __init__(self, metrics):
        super().__init__(metrics)

    def collect(self, batch_users, batch_targets, batch_preds):
        # 粗暴地把当前 Batch 的数据怼进大列表
        self.all_users.extend(batch_users.tolist())
        self.all_targets.extend(batch_targets.tolist())
        self.all_preds.extend(batch_preds.tolist())

    def summary(self):
        results = {}
        if not self.all_targets:
            return results

        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)

        # 遍历指标，通过注册表调用静态计算方法
        for m in self.metrics:
            calc_class = METRIC_REGISTRY[m]
            results[m] = calc_class.calculate(y_true=y_true, y_pred=y_pred, users=self.all_users)

        return results

    def clear(self):
        self.all_users = []
        self.all_targets = []
        self.all_preds = []


class TopKEvaluator(BaseEvaluator):
    def __init__(self, metrics, history_dict):
        self.history_dict = history_dict
        super().__init__(metrics)

        # 解析 K 值 (例如将 'hit@5', 'ndcg@10' 解析出 k_list = [5, 10])
        self.k_list = sorted(list(set([int(m.split('@')[1]) for m in self.metrics])))
        self.max_k = max(self.k_list) if self.k_list else 0

    def collect(self, batch_users, batch_targets, batch_scores):
        device = batch_scores.device
        batch_size = batch_users.size(0)
        batch_indices = torch.arange(batch_size, device=device)
        target_original_scores = batch_scores[batch_indices, batch_targets].clone()
        # 1. 瞬间 Mask 历史物品
        rows, cols = [], []
        for i, uid in enumerate(batch_users.tolist()):
            hist = self.history_dict.get(uid, [])
            rows.extend([i] * len(hist))
            cols.extend(hist)
        if rows:
            batch_scores[rows, cols] = -float('inf')
        batch_scores[batch_indices, batch_targets] = target_original_scores # 将目标物品的分数还原
        # 2. 取 Top-Max_K 并生成命中矩阵 [Batch_size, max_k]
        _, topk_indices = torch.topk(batch_scores, self.max_k, dim=-1)
        hits = (topk_indices == batch_targets.unsqueeze(1)).float()

        # 3. 现结现清：调用指标类算总和并累加
        for k in self.k_list:
            hits_k = hits[:, :k]
            for m in self.metrics:
                metric_name, metric_k = m.split('@')
                if int(metric_k) == k:
                    calc_class = METRIC_REGISTRY[metric_name]
                    # 调用静态方法获取当前 Batch 的总得分
                    batch_sum = calc_class.calculate(hits_k=hits_k, k=k)
                    self.metric_sums[m] += batch_sum

        self.total_users += batch_size

    def summary(self):
        results = {}
        if self.total_users == 0:
            return results
        for m in self.metrics:
            results[m] = self.metric_sums[m] / self.total_users
        return results

    def clear(self):
        self.metric_sums = {m: 0.0 for m in self.metrics}
        self.total_users = 0

import polars as pl
class Evaluator:
    def __init__(self, *metrics: Literal["auc", "logloss", "gauc", "HR@*", "NDCG@*"], history_dict=None):
        """
        metrics: list of str, 例如 ['auc', 'logloss', 'hit@10', 'ndcg@20']
        history_dict: dict, TopK 评估必须传入的静态历史黑名单
        """
        self.metrics = [m.lower() for m in metrics]
        # 1. 自动解析和分流指标
        self.point_metrics = [m for m in self.metrics if '@' not in m]
        self.topk_metrics = [m for m in self.metrics if '@' in m]

        # 2. 实例化底层 Evaluator
        self.point_evaluator = None
        self.topk_evaluator = None

        if self.point_metrics:
            self.point_evaluator = PointWiseEvaluator(self.point_metrics)

        if self.topk_metrics:
            assert history_dict is not None, "错误：计算 TopK 指标必须传入 history_dict 以进行历史剔除！"
            self.topk_evaluator = TopKEvaluator(self.topk_metrics, history_dict)

    def collect(self, batch_users, batch_targets, batch_preds_1d=None, batch_scores_2d=None):
        if self.point_evaluator:
            self.collect_pointwise(batch_users, batch_targets, batch_preds_1d)
        if self.topk_evaluator:
            self.collect_topk(batch_users, batch_targets, batch_scores_2d)

    def collect_pointwise(self, batch_users, batch_targets, batch_preds_1d):
        """
        收集 CTR 等 Point-wise 任务的预测结果。
        - batch_preds_1d: shape [Batch_Size]，通常为 0~1 的概率值。
        """
        if self.point_evaluator is None:
            raise RuntimeError("未初始化 PointWise 评估器。请检查是否传入了 auc, logloss 等指标。")
        self.point_evaluator.collect(batch_users, batch_targets, batch_preds_1d)

    def collect_topk(self, batch_users, batch_targets, batch_scores_2d):
        """
        收集序列推荐/召回等 Top-K 任务的全排序得分。
        - batch_scores_2d: shape [Batch_Size, Num_Items]，通常为 Raw Logits。
        """
        if self.topk_evaluator is None:
            raise RuntimeError("未初始化 TopK 评估器。请检查是否传入了 hit@k, ndcg@k 等指标。")
        self.topk_evaluator.collect(batch_users, batch_targets, batch_scores_2d)

    def summary(self):
        """
        Epoch 结束时调用，合并所有底层 Evaluator 的成绩单并保存
        epoch/step: 可选，用于记录当前是第几轮或第几步
        """
        final_results = {}

        if self.point_evaluator:
            final_results.update(self.point_evaluator.summary())

        if self.topk_evaluator:
            final_results.update(self.topk_evaluator.summary())

        return final_results

    def clear(self):
        """
        清空缓存和累加器，准备下一个 Epoch
        """
        if self.point_evaluator:
            self.point_evaluator.clear()

        if self.topk_evaluator:
            self.topk_evaluator.clear()

class EvaluatorDecorator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
    def collect_pointwise(self, batch_users, batch_targets, batch_preds_1d):
        self.evaluator.collect_pointwise(batch_users, batch_targets, batch_preds_1d)
    def collect_topk(self, batch_users, batch_targets, batch_scores_2d):
        self.evaluator.collect_topk(batch_users, batch_targets, batch_scores_2d)
    def collect(self, batch_users, batch_targets, batch_preds_1d=None, batch_scores_2d=None):
        self.evaluator.collect(batch_users, batch_targets, batch_preds_1d, batch_scores_2d)
    def summary(self, epoch=None, step=None):
        self.evaluator.summary()
    def clear(self):
        self.evaluator.clear()

import os
from datetime import datetime
class LogDecorator(EvaluatorDecorator):
    def __init__(self, evaluator, save_path=None, title=None):
        """
        evaluator: 原始的 Evaluator 实例
        save_path: 日志保存路径
        """
        super().__init__(evaluator)
        self.save_path = save_path
        self.title = title
        self._is_first_summary = True

        if self.save_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)

    def summary(self, epoch=None, step=None):
        """
        拦截原有的 summary 方法，在拿到结果后增加纯文本写入的逻辑
        同时扩展了接受 epoch 和 step 的能力
        """
        # 1. 调用原 Evaluator 的 summary 获取干净的指标字典
        final_results = self.evaluator.summary()

        # 2. 将 epoch 和 step 补充进返回给外层的字典里
        if epoch is not None:
            final_results['epoch'] = epoch
        if step is not None:
            final_results['step'] = step

        # 3. 拦截并执行纯文本日志写入逻辑
        if self.save_path:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(self.save_path, 'a', encoding='utf-8') as f:
                # 首次写入打分割线
                if self._is_first_summary:
                    title = self.title or ""
                    f.write(f"\n========== {title}实验进程开始 | 首次评估时间: {current_time} ==========\n")
                    self._is_first_summary = False

                # 拼接易读字符串
                log_parts = [f"[{current_time}]"]
                if epoch is not None:
                    log_parts.append(f"Epoch: {epoch:02d}")
                if step is not None:
                    log_parts.append(f"Step: {step}")

                # 遍历原始指标写入
                for metric_name, value in self.evaluator.summary().items():
                    if isinstance(value, float):
                        log_parts.append(f"{metric_name}: {value:.4f}")
                    else:
                        log_parts.append(f"{metric_name}: {value}")

                f.write(" | ".join(log_parts) + '\n')

        return final_results

if __name__ == '__main__':
    ...