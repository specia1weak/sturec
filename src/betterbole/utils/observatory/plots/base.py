from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


PathLike = Union[str, Path]


@dataclass
class PlotSeries:
    name: str
    x: Sequence[float]
    y: Sequence[float]
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class PlotSpec:
    title: str
    xlabel: str
    ylabel: str
    save_path: Optional[PathLike] = None
    figsize: tuple = (8, 5)
    legend: bool = True


def ensure_parent_dir(save_path: Optional[PathLike]) -> None:
    if not save_path:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def finalize_plot(spec: PlotSpec) -> None:
    plt.title(spec.title)
    plt.xlabel(spec.xlabel)
    plt.ylabel(spec.ylabel)
    plt.tight_layout()
    if spec.legend:
        plt.legend()
    if spec.save_path:
        ensure_parent_dir(spec.save_path)
        plt.savefig(spec.save_path, dpi=150, bbox_inches="tight")
        plt.close()


def maybe_numpy(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)
