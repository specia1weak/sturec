from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from betterbole.utils.observatory.plots.base import PlotSpec, finalize_plot


def plot_heatmap(
        values: Sequence[Sequence[float]],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path=None,
        xticklabels=None,
        yticklabels=None,
        cmap: str = "viridis",
        colorbar_label: str = None,
):
    plt.figure(figsize=(8, 6))
    matrix = np.asarray(values, dtype=float)
    image = plt.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    if xticklabels is not None:
        plt.xticks(np.arange(len(xticklabels)), xticklabels, rotation=45, ha="right")
    if yticklabels is not None:
        plt.yticks(np.arange(len(yticklabels)), yticklabels)
    cbar = plt.colorbar(image)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)
    finalize_plot(PlotSpec(title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path, legend=False))


def plot_domain_code_usage(values, title: str, save_path=None):
    values = np.asarray(values, dtype=float)
    xticklabels = [str(i) for i in range(values.shape[1])]
    yticklabels = [str(i) for i in range(values.shape[0])]
    plot_heatmap(
        values=values,
        title=title,
        xlabel="code_idx",
        ylabel="domain_idx",
        save_path=save_path,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        colorbar_label="usage",
    )


def plot_step_dim_heatmap(values, title: str, save_path=None):
    values = np.asarray(values, dtype=float)
    xticklabels = [str(i) for i in range(values.shape[1])]
    yticklabels = [str(i) for i in range(values.shape[0])]
    plot_heatmap(
        values=values,
        title=title,
        xlabel="dim_idx",
        ylabel="step_idx",
        save_path=save_path,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        colorbar_label="value",
    )


def plot_similarity_matrix(values, title: str, save_path=None, ticklabels=None):
    values = np.asarray(values, dtype=float)
    plot_heatmap(
        values=values,
        title=title,
        xlabel="tensor_b",
        ylabel="tensor_a",
        save_path=save_path,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cmap="coolwarm",
        colorbar_label="similarity",
    )
