from typing import Iterable, Sequence

import matplotlib.pyplot as plt

from betterbole.utils.observatory.plots.base import PlotSeries, PlotSpec, finalize_plot, maybe_numpy


def plot_scalar_series(
        steps: Sequence[float],
        values: Sequence[float],
        title: str,
        ylabel: str,
        save_path=None,
        label: str = None,
):
    plt.figure(figsize=(8, 5))
    plt.plot(maybe_numpy(steps), maybe_numpy(values), marker="o", linewidth=1.8, label=label or ylabel)
    finalize_plot(PlotSpec(title=title, xlabel="step", ylabel=ylabel, save_path=save_path, legend=label is not None))


def plot_multi_series(
        series_list: Iterable[PlotSeries],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path=None,
):
    plt.figure(figsize=(8, 5))
    for series in series_list:
        plt.plot(maybe_numpy(series.x), maybe_numpy(series.y), marker="o", linewidth=1.6, label=series.name)
    finalize_plot(PlotSpec(title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path, legend=True))
