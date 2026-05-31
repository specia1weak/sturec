from typing import Dict, List, Optional

import numpy as np
import polars as pl

from betterbole.utils.observatory.config import TensorDisplayConfig


SCALAR_LABELS = {
    "feature_mean": "feat_mean",
    "feature_var": "feat_var",
    "num_samples": "n",
    "flat_dim": "dim",
    "effective_rank": "eff_rank",
    "participation_ratio": "part_ratio",
    "stable_rank": "stable_rank",
    "top1_energy_ratio": "top1_energy",
    "top2_energy_ratio": "top2_energy",
    "dead_dim_ratio": "dead_dim",
    "mean_dim_var": "mean_dim_var",
    "max_dim_var": "max_dim_var",
    "mean_abs_corr": "mean_abs_corr",
    "max_abs_corr": "max_abs_corr",
    "sample_cosine_mean": "cos_mean",
    "sample_cosine_abs_mean": "abs_cos_mean",
}


def _render_df(df: pl.DataFrame) -> str:
    with pl.Config(
        tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=2000,
        fmt_str_lengths=2000,
    ):
        return str(df)


def _global_summary_frame(window_stats: Dict[str, object]) -> pl.DataFrame:
    row = {}
    for key, label in SCALAR_LABELS.items():
        if key in window_stats:
            row[label] = [window_stats[key]]
    return pl.DataFrame(row)


def _top_dimension_frame(window_stats: Dict[str, object], display: TensorDisplayConfig) -> Optional[pl.DataFrame]:
    mean_vec = np.asarray(window_stats["batch_mean"])
    var_vec = np.asarray(window_stats["batch_var"])
    train_var_vec = np.asarray(window_stats["train_var"])
    if mean_vec.ndim == 0:
        mean_vec = mean_vec.reshape(1)
        var_vec = var_vec.reshape(1)
        train_var_vec = train_var_vec.reshape(1)

    if mean_vec.size == 0:
        return None

    if not display.show_per_dim:
        return None

    if display.rank_by == "mean_abs":
        rank_score = np.abs(mean_vec)
    elif display.rank_by == "train_var":
        rank_score = train_var_vec
    else:
        rank_score = var_vec

    if mean_vec.size <= display.max_display_dims:
        selected = np.arange(mean_vec.size)
    else:
        topk = min(display.topk_display_dims, mean_vec.size)
        selected = np.argsort(-rank_score)[:topk]
        selected = np.sort(selected)

    frame = pl.DataFrame({
        "dim_idx": selected.astype(int),
        "batch_mean": mean_vec[selected],
        "batch_var": var_vec[selected],
        "train_var": train_var_vec[selected],
    })
    return frame


def render_tensor_report(name: str, window_stats: Dict[str, object], display: TensorDisplayConfig) -> str:
    sections: List[str] = [f"{name:=^20}"]
    if display.show_global_summary:
        sections.append(_render_df(_global_summary_frame(window_stats)))

    dim_frame = _top_dimension_frame(window_stats, display)
    if dim_frame is not None:
        if len(dim_frame) < int(window_stats["flat_dim"]):
            sections.append("[Top-Dim]")
        else:
            sections.append("[Per-Dim]")
        sections.append(_render_df(dim_frame))
    return " \n".join(sections)


def render_relation_report(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return ""
    return f"{'relation_stats':=^20} \n{_render_df(pl.DataFrame(rows))}"
