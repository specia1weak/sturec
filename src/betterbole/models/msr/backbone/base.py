from abc import ABC, abstractmethod
import warnings

import torch
from torch import nn

class MSRBackbone(nn.Module, ABC):
    """
    Deprecated backbone base kept only for compatibility with legacy MSR backbones.

    New model-specific implementations should inherit directly from `nn.Module`
    unless they truly need a shared backbone abstraction.
    """
    def __init__(self, input_dim: int, num_domains: int, output_dim: int):
        super().__init__()
        warnings.warn(
            "MSRBackbone is deprecated and retained only for legacy compatibility. "
            "Do not use it for new model implementations; inherit from nn.Module instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.input_dim = int(input_dim)
        self.num_domains = int(num_domains)
        self.output_dim = int(output_dim)

    @abstractmethod
    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
