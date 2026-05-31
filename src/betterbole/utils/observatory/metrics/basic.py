from typing import Dict

import torch

from betterbole.utils.observatory.analysis import flatten_tensor


def _flat(x: torch.Tensor) -> torch.Tensor:
    return flatten_tensor(x).float()


def feature_mean(x: torch.Tensor) -> float:
    return float(_flat(x).mean().item())


def feature_var(x: torch.Tensor) -> float:
    return float(_flat(x).var(unbiased=False).item())


def num_samples(x: torch.Tensor) -> int:
    return int(_flat(x).size(0))


def flat_dim(x: torch.Tensor) -> int:
    return int(_flat(x).size(1))


def per_dim_mean(x: torch.Tensor) -> torch.Tensor:
    return _flat(x).mean(dim=0)


def per_dim_var(x: torch.Tensor) -> torch.Tensor:
    return _flat(x).var(dim=0, unbiased=False)


def dead_dim_ratio(x: torch.Tensor, threshold: float = 1e-6) -> float:
    dim_var = per_dim_var(x)
    return float((dim_var < threshold).float().mean().item())


def near_zero_ratio(x: torch.Tensor, threshold: float = 1e-4) -> float:
    return float((_flat(x).abs() < threshold).float().mean().item())


def basic_stats(x: torch.Tensor) -> Dict[str, float]:
    return {
        "feature_mean": feature_mean(x),
        "feature_var": feature_var(x),
        "num_samples": float(num_samples(x)),
        "flat_dim": float(flat_dim(x)),
        "dead_dim_ratio": dead_dim_ratio(x),
        "near_zero_ratio": near_zero_ratio(x),
    }
