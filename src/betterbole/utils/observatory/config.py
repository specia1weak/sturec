from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TensorSketchConfig:
    max_samples: int = 256
    max_dims: int = 256
    eps: float = 1e-12


@dataclass
class TensorDisplayConfig:
    show_global_summary: bool = True
    show_per_dim: bool = True
    max_display_dims: int = 16
    topk_display_dims: int = 8
    rank_by: str = "variance"


@dataclass
class TensorMonitorOptions:
    metrics: Tuple[str, ...] = ("basic", "spectral", "correlation", "cosine")
    display: TensorDisplayConfig = field(default_factory=TensorDisplayConfig)
    sketch: TensorSketchConfig = field(default_factory=TensorSketchConfig)


@dataclass
class RelationOptions:
    enabled: bool = False
    rank: int = 8
    max_pairs: int = 12
    names: Optional[Tuple[str, ...]] = None
