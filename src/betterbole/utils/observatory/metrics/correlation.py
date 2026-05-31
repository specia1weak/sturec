from typing import Dict, Tuple

import torch

from betterbole.utils.observatory.analysis import flatten_tensor


def _flat(x: torch.Tensor) -> torch.Tensor:
    return flatten_tensor(x).float()


def mean_abs_corr(x: torch.Tensor, eps: float = 1e-12) -> float:
    flat_x = _flat(x)
    if flat_x.size(0) < 2 or flat_x.size(1) < 2:
        return 0.0
    centered = flat_x - flat_x.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    valid_mask = std > eps
    if int(valid_mask.sum().item()) < 2:
        return 0.0
    normalized = centered[:, valid_mask] / std[valid_mask].clamp_min(eps)
    corr = normalized.t().mm(normalized) / max(1, normalized.size(0))
    eye = torch.eye(corr.size(0), dtype=torch.bool)
    off_diag = corr.masked_select(~eye)
    if off_diag.numel() == 0:
        return 0.0
    return float(off_diag.abs().mean().item())


def max_abs_corr(x: torch.Tensor, eps: float = 1e-12) -> float:
    flat_x = _flat(x)
    if flat_x.size(0) < 2 or flat_x.size(1) < 2:
        return 0.0
    centered = flat_x - flat_x.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    valid_mask = std > eps
    if int(valid_mask.sum().item()) < 2:
        return 0.0
    normalized = centered[:, valid_mask] / std[valid_mask].clamp_min(eps)
    corr = normalized.t().mm(normalized) / max(1, normalized.size(0))
    eye = torch.eye(corr.size(0), dtype=torch.bool)
    off_diag = corr.masked_select(~eye)
    if off_diag.numel() == 0:
        return 0.0
    return float(off_diag.abs().max().item())


def sample_cosine_stats(x: torch.Tensor, eps: float = 1e-12) -> Tuple[float, float]:
    flat_x = _flat(x)
    if flat_x.size(0) < 2 or flat_x.size(1) == 0:
        return 0.0, 0.0
    normalized = torch.nn.functional.normalize(flat_x, p=2, dim=1, eps=eps)
    sim = normalized.mm(normalized.t())
    eye = torch.eye(sim.size(0), dtype=torch.bool)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0:
        return 0.0, 0.0
    return float(off_diag.mean().item()), float(off_diag.abs().mean().item())


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = _flat(x) - _flat(x).mean(dim=0, keepdim=True)
    y = _flat(y) - _flat(y).mean(dim=0, keepdim=True)
    if x.size(0) < 2 or y.size(0) < 2 or x.size(1) == 0 or y.size(1) == 0:
        return 0.0
    cross = x.t().mm(y)
    xx = x.t().mm(x)
    yy = y.t().mm(y)
    numerator = cross.pow(2).sum()
    denominator = torch.sqrt(xx.pow(2).sum().clamp_min(eps) * yy.pow(2).sum().clamp_min(eps))
    return float((numerator / denominator.clamp_min(eps)).item())


def principal_angles(x: torch.Tensor, y: torch.Tensor, rank: int = 8) -> Tuple[float, float]:
    x = _flat(x) - _flat(x).mean(dim=0, keepdim=True)
    y = _flat(y) - _flat(y).mean(dim=0, keepdim=True)
    usable_rank = min(rank, x.size(0), x.size(1), y.size(0), y.size(1))
    if usable_rank <= 0:
        return 0.0, 0.0
    ux = torch.linalg.svd(x, full_matrices=False).U[:, :usable_rank]
    uy = torch.linalg.svd(y, full_matrices=False).U[:, :usable_rank]
    sigma = torch.linalg.svdvals(ux.t().mm(uy)).clamp(0.0, 1.0)
    return float(sigma.mean().item()), float(sigma.min().item())


def subspace_overlap(x: torch.Tensor, y: torch.Tensor, rank: int = 8) -> Tuple[float, float]:
    return principal_angles(x, y, rank=rank)


def correlation_summary(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12, rank: int = 8) -> Dict[str, float]:
    angle_mean, angle_min = principal_angles(x, y, rank=rank)
    return {
        "mean_abs_corr": mean_abs_corr(x, eps=eps),
        "max_abs_corr": max_abs_corr(x, eps=eps),
        "sample_cos_mean": sample_cosine_stats(x, eps=eps)[0],
        "sample_cos_abs_mean": sample_cosine_stats(x, eps=eps)[1],
        "linear_cka": linear_cka(x, y, eps=eps),
        "subspace_mean_cos": angle_mean,
        "subspace_min_cos": angle_min,
    }
