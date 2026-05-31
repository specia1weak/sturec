from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from betterbole.utils.observatory.plots.heatmap import plot_domain_code_usage, plot_step_dim_heatmap
from betterbole.utils.observatory.plots.profile import plot_dim_profile
from betterbole.utils.observatory.plots.series import plot_multi_series, plot_scalar_series
from betterbole.utils.observatory.plots.base import PlotSeries


def build_tensor_report(
        output_dir,
        step_values: Dict[str, Sequence[float]],
        step_axis: Sequence[float],
        dim_profile: Dict[str, Sequence[float]] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, values in step_values.items():
        plot_scalar_series(
            steps=step_axis,
            values=values,
            title=metric_name,
            ylabel=metric_name,
            save_path=output_dir / ("%s_series.png" % metric_name),
        )

    if dim_profile:
        for metric_name, values in dim_profile.items():
            plot_dim_profile(
                dim_idx=np.arange(len(values)),
                values=values,
                title=metric_name,
                ylabel=metric_name,
                save_path=output_dir / ("%s_profile.png" % metric_name),
            )


def build_vq_report(
        output_dir,
        entropy_series: Sequence[float],
        step_axis: Sequence[float],
        domain_code_usage,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scalar_series(
        steps=step_axis,
        values=entropy_series,
        title="code_entropy",
        ylabel="code_entropy",
        save_path=output_dir / "code_entropy_series.png",
    )
    plot_domain_code_usage(
        values=domain_code_usage,
        title="domain_code_usage",
        save_path=output_dir / "domain_code_usage.png",
    )


def build_multi_series_report(output_dir, series_map: Dict[str, Iterable[PlotSeries]], xlabel: str, ylabel: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for report_name, series_list in series_map.items():
        plot_multi_series(
            series_list=series_list,
            title=report_name,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=output_dir / ("%s.png" % report_name),
        )
