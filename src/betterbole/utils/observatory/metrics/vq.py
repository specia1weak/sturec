from typing import Dict, Tuple

import torch

from betterbole.utils.observatory.analysis import flatten_tensor


def code_usage_hist(code_indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
    flat = code_indices.view(-1).long()
    return torch.bincount(flat, minlength=int(codebook_size)).float()


def used_code_ratio(code_indices: torch.Tensor, codebook_size: int) -> float:
    hist = code_usage_hist(code_indices, codebook_size)
    return float((hist > 0).float().mean().item())


def code_entropy(code_indices: torch.Tensor, codebook_size: int, eps: float = 1e-12) -> float:
    hist = code_usage_hist(code_indices, codebook_size)
    probs = hist / hist.sum().clamp_min(eps)
    probs = probs[probs > 0]
    if probs.numel() == 0:
        return 0.0
    return float((-(probs * probs.log()).sum()).item())


def per_domain_code_usage(code_indices: torch.Tensor, domain_ids: torch.Tensor, codebook_size: int) -> torch.Tensor:
    code_indices = code_indices.view(-1).long()
    domain_ids = domain_ids.view(-1).long()
    num_domains = int(domain_ids.max().item()) + 1 if domain_ids.numel() > 0 else 0
    usage = torch.zeros(num_domains, int(codebook_size), dtype=torch.float32, device=code_indices.device)
    for domain_idx in range(num_domains):
        mask = domain_ids == domain_idx
        if mask.any():
            usage[domain_idx] = code_usage_hist(code_indices[mask], codebook_size)
    return usage


def code_usage_js_divergence(code_indices: torch.Tensor, domain_ids: torch.Tensor, codebook_size: int, eps: float = 1e-12) -> float:
    usage = per_domain_code_usage(code_indices, domain_ids, codebook_size)
    if usage.size(0) < 2:
        return 0.0
    probs = usage / usage.sum(dim=-1, keepdim=True).clamp_min(eps)
    mean_prob = probs.mean(dim=0).clamp_min(eps)
    js = 0.0
    for i in range(probs.size(0)):
        p = probs[i].clamp_min(eps)
        m = mean_prob
        js += (p * (p / m).log()).sum()
    return float((js / probs.size(0)).item())


def quantized_cos(z: torch.Tensor, quantized: torch.Tensor, eps: float = 1e-12) -> float:
    z = flatten_tensor(z).float()
    quantized = flatten_tensor(quantized).float()
    z_n = torch.nn.functional.normalize(z, dim=-1, eps=eps)
    q_n = torch.nn.functional.normalize(quantized, dim=-1, eps=eps)
    return float((z_n * q_n).sum(dim=-1).mean().item())


def residual_norm(z: torch.Tensor, quantized: torch.Tensor) -> float:
    z = flatten_tensor(z).float()
    quantized = flatten_tensor(quantized).float()
    return float((z - quantized).norm(dim=-1).mean().item())


def codebook_pairwise_cos(codebook: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    codebook = flatten_tensor(codebook).float()
    codebook = torch.nn.functional.normalize(codebook, dim=-1, eps=eps)
    return codebook.mm(codebook.t())


def codebook_min_separation(codebook: torch.Tensor, eps: float = 1e-12) -> float:
    sim = codebook_pairwise_cos(codebook, eps=eps)
    eye = torch.eye(sim.size(0), dtype=torch.bool)
    off_diag = sim.masked_fill(eye, -1.0)
    if off_diag.numel() == 0:
        return 0.0
    return float((1.0 - off_diag.max().item()))


def nearest_second_margin(similarity: torch.Tensor) -> float:
    if similarity.ndim != 2 or similarity.size(1) < 2:
        return 0.0
    top2 = torch.topk(similarity, k=2, dim=1).values
    return float((top2[:, 0] - top2[:, 1]).mean().item())


def vq_summary(code_indices: torch.Tensor, codebook_size: int, z: torch.Tensor = None, quantized: torch.Tensor = None) -> Dict[str, float]:
    summary = {
        "used_code_ratio": used_code_ratio(code_indices, codebook_size),
        "code_entropy": code_entropy(code_indices, codebook_size),
    }
    if z is not None and quantized is not None:
        summary["quantized_cos"] = quantized_cos(z, quantized)
        summary["residual_norm"] = residual_norm(z, quantized)
    return summary
