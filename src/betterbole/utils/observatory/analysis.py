from typing import Dict, Tuple

import torch


def flatten_tensor(tensor_data: torch.Tensor) -> torch.Tensor:
    tensor_cpu = tensor_data.detach().float().cpu()
    if tensor_cpu.ndim == 0:
        return tensor_cpu.view(1, 1)
    if tensor_cpu.ndim == 1:
        return tensor_cpu.view(-1, 1)
    return tensor_cpu.reshape(tensor_cpu.shape[0], -1)


def uniform_sample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:
    if x.size(0) <= max_rows:
        return x
    row_index = torch.linspace(0, x.size(0) - 1, steps=max_rows).round().long()
    return x.index_select(0, row_index)


def select_top_variance_dims(x: torch.Tensor, max_dims: int) -> torch.Tensor:
    if x.size(1) <= max_dims:
        return x
    dim_var = x.var(dim=0, unbiased=False)
    topk = min(max_dims, x.size(1))
    dim_index = torch.topk(dim_var, k=topk).indices.sort().values
    return x.index_select(1, dim_index)


def build_sketch(x: torch.Tensor, max_rows: int, max_dims: int) -> torch.Tensor:
    sketch = uniform_sample_rows(x, max_rows)
    sketch = select_top_variance_dims(sketch, max_dims)
    return sketch.contiguous()


def safe_corrcoef(x: torch.Tensor, eps: float) -> Tuple[float, float]:
    if x.size(0) < 2 or x.size(1) < 2:
        return 0.0, 0.0
    centered = x - x.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    valid_mask = std > eps
    if int(valid_mask.sum().item()) < 2:
        return 0.0, 0.0
    normalized = centered[:, valid_mask] / std[valid_mask].clamp_min(eps)
    corr = normalized.t().mm(normalized) / max(1, normalized.size(0))
    eye = torch.eye(corr.size(0), dtype=torch.bool)
    off_diag = corr.masked_select(~eye)
    if off_diag.numel() == 0:
        return 0.0, 0.0
    return float(off_diag.abs().mean().item()), float(off_diag.abs().max().item())


def sample_cosine_stats(x: torch.Tensor, eps: float) -> Tuple[float, float]:
    if x.size(0) < 2 or x.size(1) == 0:
        return 0.0, 0.0
    normalized = torch.nn.functional.normalize(x, p=2, dim=1, eps=eps)
    sim = normalized.mm(normalized.t())
    eye = torch.eye(sim.size(0), dtype=torch.bool)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0:
        return 0.0, 0.0
    return float(off_diag.mean().item()), float(off_diag.abs().mean().item())


def spectral_stats(x: torch.Tensor, eps: float) -> Dict[str, float]:
    if x.numel() == 0 or x.size(1) == 0:
        return {
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "stable_rank": 0.0,
            "top1_energy_ratio": 0.0,
            "top2_energy_ratio": 0.0,
            "dead_dim_ratio": 1.0,
            "mean_dim_var": 0.0,
            "max_dim_var": 0.0,
            "mean_abs_corr": 0.0,
            "max_abs_corr": 0.0,
            "sample_cosine_mean": 0.0,
            "sample_cosine_abs_mean": 0.0,
        }

    centered = x - x.mean(dim=0, keepdim=True)
    dim_var = centered.var(dim=0, unbiased=False)
    dead_dim_ratio = float((dim_var < 1e-6).float().mean().item())
    mean_dim_var = float(dim_var.mean().item())
    max_dim_var = float(dim_var.max().item()) if dim_var.numel() > 0 else 0.0
    mean_abs_corr, max_abs_corr = safe_corrcoef(centered, eps=eps)
    cosine_mean, cosine_abs_mean = sample_cosine_stats(centered, eps=eps)

    if centered.size(0) < 2:
        return {
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "stable_rank": 0.0,
            "top1_energy_ratio": 0.0,
            "top2_energy_ratio": 0.0,
            "dead_dim_ratio": dead_dim_ratio,
            "mean_dim_var": mean_dim_var,
            "max_dim_var": max_dim_var,
            "mean_abs_corr": mean_abs_corr,
            "max_abs_corr": max_abs_corr,
            "sample_cosine_mean": cosine_mean,
            "sample_cosine_abs_mean": cosine_abs_mean,
        }

    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.pow(2)
    total_energy = energy.sum().clamp_min(eps)
    probs = energy / total_energy
    spectral_entropy = -(probs * probs.clamp_min(eps).log()).sum()
    effective_rank = torch.exp(spectral_entropy)
    participation_ratio = total_energy.pow(2) / energy.pow(2).sum().clamp_min(eps)
    stable_rank = total_energy / energy.max().clamp_min(eps)
    top1_ratio = energy[0] / total_energy
    top2_ratio = energy[:2].sum() / total_energy

    return {
        "effective_rank": float(effective_rank.item()),
        "participation_ratio": float(participation_ratio.item()),
        "stable_rank": float(stable_rank.item()),
        "top1_energy_ratio": float(top1_ratio.item()),
        "top2_energy_ratio": float(top2_ratio.item()),
        "dead_dim_ratio": dead_dim_ratio,
        "mean_dim_var": mean_dim_var,
        "max_dim_var": max_dim_var,
        "mean_abs_corr": mean_abs_corr,
        "max_abs_corr": max_abs_corr,
        "sample_cosine_mean": cosine_mean,
        "sample_cosine_abs_mean": cosine_abs_mean,
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float) -> float:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    if x.size(0) < 2 or y.size(0) < 2 or x.size(1) == 0 or y.size(1) == 0:
        return 0.0
    cross = x.t().mm(y)
    xx = x.t().mm(x)
    yy = y.t().mm(y)
    numerator = cross.pow(2).sum()
    denominator = torch.sqrt(xx.pow(2).sum().clamp_min(eps) * yy.pow(2).sum().clamp_min(eps))
    return float((numerator / denominator.clamp_min(eps)).item())


def subspace_overlap(x: torch.Tensor, y: torch.Tensor, rank: int) -> Tuple[float, float]:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    usable_rank = min(rank, x.size(0), x.size(1), y.size(0), y.size(1))
    if usable_rank <= 0:
        return 0.0, 0.0
    ux = torch.linalg.svd(x, full_matrices=False).U[:, :usable_rank]
    uy = torch.linalg.svd(y, full_matrices=False).U[:, :usable_rank]
    sigma = torch.linalg.svdvals(ux.t().mm(uy)).clamp(0.0, 1.0)
    return float(sigma.mean().item()), float(sigma.min().item())
