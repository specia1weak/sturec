from abc import ABC, abstractmethod

import torch
from torch import nn


class MSRBackbone(nn.Module, ABC):
    def __init__(self, input_dim: int, num_domains: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_domains = int(num_domains)
        self.output_dim = int(output_dim)

    @abstractmethod
    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
