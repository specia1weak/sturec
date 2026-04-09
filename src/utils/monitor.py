import numpy as np
import torch
import torch.nn as nn
from collections import deque
import polars as pl

class IndividualReLUMonitor:
    def __init__(self, model, window_size=50):
        self.window_size = window_size
        self.zero_ratios = {}  # 字典：存放每个 ReLU 层的历史记录队列
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        registered_ids = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                mod_id = id(module)
                if mod_id not in registered_ids:
                    self.zero_ratios[name] = deque(maxlen=self.window_size)
                    hook_handle = module.register_forward_hook(self._make_hook(name))
                    self.hooks.append(hook_handle)

                    registered_ids.add(mod_id)
                else:
                    print(f"[提示] 层 '{name}' 复用了已挂载的 ReLU 实例，跳过重复注册。")

    def _make_hook(self, layer_name):
        def hook(module, input, output):
            # 计算当前 batch 中，输出为 0 的激活值比例
            zero_ratio = (output.detach() == 0).float().mean().item()
            # 准确追加到属于自己的队列中
            self.zero_ratios[layer_name].append(zero_ratio)

        return hook

    def get_layer_stats(self):
        """计算并返回各个层在当前窗口内的平均 0 占比"""
        stats = {}
        for name, ratios in self.zero_ratios.items():
            if ratios:
                stats[name] = sum(ratios) / len(ratios)
            else:
                stats[name] = 0.0
        return stats

    def get_latest_batch_stats(self):
        """只获取当前最新一个 batch 的数据"""
        return {name: ratios[-1] for name, ratios in self.zero_ratios.items() if ratios}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


class ExplicitFeatureMonitor:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.features = {}

    def record(self, name, tensor_data):
        """
        tensor_data 假设是 [batch_size, num_experts] 的分布
        """
        if name not in self.features:
            self.features[name] = deque(maxlen=self.window_size)
        tensor_cpu = tensor_data.detach().float().cpu()
        batch_mean = tensor_cpu.mean(dim=0).numpy()
        batch_var = tensor_cpu.var(dim=0).numpy()
        # 新增：计算整个 batch 的全局均值和方差
        feature_mean = tensor_cpu.mean().item()
        feature_var = tensor_cpu.var().item()
        self.features[name].append({
            'batch_mean': batch_mean,
            'batch_var': batch_var,
            'feature_mean': feature_mean,
            'feature_var': feature_var
        })

    def get_window_stats(self):
        stats = {}
        for name, data_queue in self.features.items():
            if len(data_queue) > 0:
                # 提取出所有的 mean 和 var
                window_means = np.array(
                    [item['batch_mean'] for item in data_queue])  # shape: [window_size, num_experts]
                window_vars = np.array([item['batch_var'] for item in data_queue])  # shape: [window_size, num_experts]
                feature_means = np.array([item['feature_mean'] for item in data_queue])  # shape: [window_size]
                feature_vars = np.array([item['feature_var'] for item in data_queue])  # shape: [window_size]
                stats[name] = {
                    '样本平均值': np.mean(window_means, axis=0),
                    '样本方差': np.mean(window_vars, axis=0),
                    '训练方差': np.var(window_means, axis=0),
                    '特征均值': np.mean(feature_means),
                    '特征方差': np.mean(feature_vars),
                }
            else:
                stats[name] = None

        infos = []
        for name, stat in stats.items():
            infos.append(f"{name:=^20} \n{str(pl.DataFrame(stat))}")
        return "\n".join(infos)