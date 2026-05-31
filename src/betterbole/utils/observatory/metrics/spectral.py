from dataclasses import dataclass
from typing import Dict

import torch

from betterbole.utils.observatory.analysis import flatten_tensor


@dataclass
class SpectrumStats:
    singular_values: torch.Tensor
    energy: torch.Tensor
    total_energy: torch.Tensor
    effective_rank: float
    participation_ratio: float
    stable_rank: float
    spectral_entropy: float


def compute_spectrum(x: torch.Tensor, eps: float = 1e-12) -> SpectrumStats:
    flat_x = flatten_tensor(x).float()
    centered = flat_x - flat_x.mean(dim=0, keepdim=True)
    if centered.numel() == 0 or centered.size(1) == 0 or centered.size(0) < 2:
        zeros = torch.zeros(1)
        return SpectrumStats(
            singular_values=zeros,
            energy=zeros,
            total_energy=zeros,
            effective_rank=0.0,
            participation_ratio=0.0,
            stable_rank=0.0,
            spectral_entropy=0.0,
        )

    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.pow(2)
    total_energy = energy.sum().clamp_min(eps)
    probs = energy / total_energy
    spectral_entropy = float((-(probs * probs.clamp_min(eps).log()).sum()).item())
    effective_rank = float(torch.exp(torch.tensor(spectral_entropy)).item())
    participation_ratio = float((total_energy.pow(2) / energy.pow(2).sum().clamp_min(eps)).item())
    stable_rank = float((total_energy / energy.max().clamp_min(eps)).item())
    return SpectrumStats(
        singular_values=singular_values,
        energy=energy,
        total_energy=total_energy,
        effective_rank=effective_rank,
        participation_ratio=participation_ratio,
        stable_rank=stable_rank,
        spectral_entropy=spectral_entropy,
    )


def singular_values(x: torch.Tensor) -> torch.Tensor:
    return compute_spectrum(x).singular_values


def effective_rank(x: torch.Tensor) -> float:
    return compute_spectrum(x).effective_rank


def participation_ratio(x: torch.Tensor) -> float:
    return compute_spectrum(x).participation_ratio


def stable_rank(x: torch.Tensor) -> float:
    return compute_spectrum(x).stable_rank


def topk_energy_ratio(x: torch.Tensor, k: int = 1, eps: float = 1e-12) -> float:
    stats = compute_spectrum(x, eps=eps)
    if stats.energy.numel() == 0:
        return 0.0
    k = max(1, min(int(k), int(stats.energy.numel())))
    return float((stats.energy[:k].sum() / stats.total_energy.clamp_min(eps)).item())


def spectral_summary(x: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    stats = compute_spectrum(x, eps=eps)
    return {
        "effective_rank": stats.effective_rank,
        "participation_ratio": stats.participation_ratio,
        "stable_rank": stats.stable_rank,
        "spectral_entropy": stats.spectral_entropy,
        "top1_energy_ratio": topk_energy_ratio(x, k=1, eps=eps),
        "top2_energy_ratio": topk_energy_ratio(x, k=2, eps=eps),
    }
