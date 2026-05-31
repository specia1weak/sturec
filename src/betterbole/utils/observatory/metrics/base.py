from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch


@dataclass
class MetricInput:
    name: str
    tensor: torch.Tensor
    step: Optional[int] = None
    domain_ids: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    name: str
    kind: str
    axis_names: Tuple[str, ...]
    values: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSpec:
    name: str
    fn: Callable[..., MetricResult]
    kind: str
    axis_names: Tuple[str, ...]
    default_enabled: bool = True
    default_plot: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
