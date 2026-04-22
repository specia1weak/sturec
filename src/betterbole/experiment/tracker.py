import os
from pathlib import Path

import torch
import numpy as np


class TrainingTracker:
    def __init__(self, workdir):
        self.save_dir = Path(workdir) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.current_epoch = 0

        self.metrics_history = {}  # 例如: {'loss': [], 'auc': []}
        self.best_metric_value = -float('inf')  # 假设指标越大越好，如果是 loss 则改为 float('inf')

        self.saved_vectors = {}

    def step(self):
        """推进全局步数"""
        self.global_step += 1

    def log_metrics(self, metrics_dict):
        """
        通用指标记录接口
        tracker.log_metrics({'train_loss': 0.5, 'lr': 0.001})
        """
        for k, v in metrics_dict.items():
            if k not in self.metrics_history:
                self.metrics_history[k] = []
            self.metrics_history[k].append((self.global_step, v))

    def save_checkpoint(self, model, optimizer=None, is_best=False, metric_val=None):
        """
        保存断点（包含模型、优化器、步数）
        """
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metric_val is not None:
            checkpoint['metric'] = metric_val

        # 1. 保存常规 checkpoint (可以每 N 步覆盖保存一次，或者带上 step 后缀)
        save_path = os.path.join(self.save_dir, f"ckpt_step_{self.global_step}.pth")
        torch.save(checkpoint, save_path)

        # 2. 如果是当前最佳模型，单独存一份
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"🌟 [Step {self.global_step}] 发现并保存了新的最佳模型！")

    def load_checkpoint(self, model, optimizer=None, ckpt_path=None):
        """
        断点续训/推理加载接口
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

        print(f"正在加载 Checkpoint: {ckpt_path}")
        # 注意：如果是多卡训练后在单卡加载，可能需要 map_location='cpu'
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

        # 恢复模型权重
        model.load_state_dict(checkpoint['model_state_dict'])

        # 恢复优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复全局状态
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)

        print(f"✅ 加载成功！恢复至 Epoch {self.current_epoch}, Global Step {self.global_step}")
        return model, optimizer

    # --- 向量存取相关的接口保持不变 ---
    def save_vector(self, name, vector_tensor):
        self.saved_vectors[name] = vector_tensor.detach().cpu().numpy()

    def export_vectors(self):
        if not self.saved_vectors:
            return
        file_path = os.path.join(self.save_dir, f"vectors_{self.global_step}.npz")
        np.savez(file_path, **self.saved_vectors)
        self.saved_vectors.clear()