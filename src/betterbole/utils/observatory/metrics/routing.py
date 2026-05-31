from typing import Dict

import torch

from betterbole.utils.observatory.analysis import flatten_tensor


def gate_mean(gate_weights: torch.Tensor) -> torch.Tensor:
    return flatten_tensor(gate_weights).float().mean(dim=0)


def gate_var(gate_weights: torch.Tensor) -> torch.Tensor:
    return flatten_tensor(gate_weights).float().var(dim=0, unbiased=False)


def gate_entropy(gate_weights: torch.Tensor, eps: float = 1e-12) -> float:
    flat = flatten_tensor(gate_weights).float()
    probs = flat.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    return float(entropy.item())


def expert_usage(gate_weights: torch.Tensor) -> torch.Tensor:
    flat = flatten_tensor(gate_weights).float()
    return flat.mean(dim=0)


def expert_dead_ratio(gate_weights: torch.Tensor, threshold: float = 1e-3) -> float:
    usage = expert_usage(gate_weights)
    return float((usage < threshold).float().mean().item())


def weighted_feature_var(raw_feature: torch.Tensor, gate_weights: torch.Tensor) -> float:
    raw = flatten_tensor(raw_feature).float()
    gate = flatten_tensor(gate_weights).float()
    weighted = raw * gate.mean(dim=-1, keepdim=True)
    return float(weighted.var(unbiased=False).item())


def pre_post_gate_var(raw_feature: torch.Tensor, gated_feature: torch.Tensor) -> Dict[str, float]:
    raw = flatten_tensor(raw_feature).float()
    gated = flatten_tensor(gated_feature).float()
    return {
        "raw_var": float(raw.var(unbiased=False).item()),
        "gated_var": float(gated.var(unbiased=False).item()),
        "ratio": float((gated.var(unbiased=False) / raw.var(unbiased=False).clamp_min(1e-12)).item()),
    }
