from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from betterbole.utils.observatory.plots.base import PlotSpec, finalize_plot, maybe_numpy


def plot_dim_profile(
        dim_idx: Sequence[float],
        values: Sequence[float],
        title: str,
        ylabel: str,
        save_path=None,
        kind: str = "line",
):
    plt.figure(figsize=(9, 5))
    x = maybe_numpy(dim_idx)
    y = maybe_numpy(values)
    if kind == "bar":
        plt.bar(x, y)
    else:
        plt.plot(x, y, marker="o", linewidth=1.6)
    finalize_plot(PlotSpec(title=title, xlabel="dim_idx", ylabel=ylabel, save_path=save_path, legend=False))


def plot_topk_bar(
        dim_idx: Sequence[float],
        values: Sequence[float],
        title: str,
        ylabel: str,
        save_path=None,
):
    plot_dim_profile(dim_idx=dim_idx, values=values, title=title, ylabel=ylabel, save_path=save_path, kind="bar")


def plot_ranked_profile(
        values: Sequence[float],
        title: str,
        ylabel: str,
        save_path=None,
        descending: bool = True,
):
    values = maybe_numpy(values)
    ranked = np.sort(values)
    if descending:
        ranked = ranked[::-1]
    dim_idx = np.arange(ranked.size)
    plot_dim_profile(dim_idx=dim_idx, values=ranked, title=title, ylabel=ylabel, save_path=save_path, kind="line")
