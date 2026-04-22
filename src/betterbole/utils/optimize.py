import torch
import torch.nn as nn
from typing import List, Dict, Any, Union, Iterable, Tuple

def split_params_by_decay(
        named_params: Iterable[Tuple[str, nn.Parameter]],  # 修改这里
        weight_decay: float = 0.01,
        no_decay_keywords: Union[tuple, list] = ("embedding", "position_ids")
) -> List[Dict[str, Any]]:
    """
    基于张量维度和关键字自动分组优化器参数。
    """
    decay_params = []
    no_decay_params = []

    for name, param in named_params: # 修改这里
        if not param.requires_grad:
            continue

        if param.ndim < 2 or any(nd in name.lower() for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

def create_complex_optimizer_groups(
        model: nn.Module,
        decay_dict: Dict[str, float],
        default_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    根据传入的 decay_dict 灵活配置不同网络层的前缀参数衰减。

    参数示例:
    decay_dict = {
        "embeddings": 0.0,       # 推荐系统 Embedding
        "moe_router": 0.05,      # 假设 MoE 门控网络需要更强的正则化
        "bias": 0.0,             # 偏置项
        "norm": 0.0              # 归一化层
    }
    """
    # 构建一个按 weight_decay 分组的字典
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 默认衰减值
        current_decay = default_decay

        # 遍历字典进行匹配（可以根据需求换成正则 re.search）
        for key, decay_value in decay_dict.items():
            if key in name.lower():
                current_decay = decay_value
                break  # 匹配到第一个优先级最高的就跳出

        # 归入对应的 decay 组
        if current_decay not in groups:
            groups[current_decay] = []
        groups[current_decay].append(param)

    # 转换为 Optimizer 需要的格式
    return [
        {"params": params, "weight_decay": wd}
        for wd, params in groups.items()
    ]