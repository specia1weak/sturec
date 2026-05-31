from collections import deque

import torch.nn as nn


class IndividualReLURecorder:
    def __init__(self, model, window_size=50):
        self.window_size = window_size
        self.zero_ratios = {}
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        registered_ids = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                mod_id = id(module)
                if mod_id in registered_ids:
                    print(f"[提示] 层 '{name}' 复用了已挂载的 ReLU 实例，跳过重复注册。")
                    continue
                self.zero_ratios[name] = deque(maxlen=self.window_size)
                hook_handle = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook_handle)
                registered_ids.add(mod_id)

    def _make_hook(self, layer_name):
        def hook(module, input_args, output):
            del module, input_args
            zero_ratio = (output.detach() == 0).float().mean().item()
            self.zero_ratios[layer_name].append(zero_ratio)
        return hook

    def get_layer_stats(self):
        stats = {}
        for name, ratios in self.zero_ratios.items():
            stats[name] = sum(ratios) / len(ratios) if ratios else 0.0
        return stats

    def get_latest_batch_stats(self):
        return {name: ratios[-1] for name, ratios in self.zero_ratios.items() if ratios}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
