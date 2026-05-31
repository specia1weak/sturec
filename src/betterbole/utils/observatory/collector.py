from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from betterbole.utils.observatory.analysis import (
    build_sketch,
    flatten_tensor,
    linear_cka,
    spectral_stats,
    subspace_overlap,
)
from betterbole.utils.observatory.config import RelationOptions, TensorDisplayConfig, TensorMonitorOptions
from betterbole.utils.observatory.formatting import render_relation_report, render_tensor_report


class TensorObservatory:
    def __init__(
            self,
            window_size: int = 50,
            default_options: Optional[TensorMonitorOptions] = None,
            relation_options: Optional[RelationOptions] = None,
    ):
        self.window_size = int(window_size)
        self.default_options = default_options or TensorMonitorOptions()
        self.relation_options = relation_options or RelationOptions()
        self.features: Dict[str, Deque[Dict[str, object]]] = {}
        self.options_by_name: Dict[str, TensorMonitorOptions] = {}
        self._record_counters: Dict[str, int] = {}

    def register(self, name: str, options: Optional[TensorMonitorOptions] = None) -> None:
        if name not in self.features:
            self.features[name] = deque(maxlen=self.window_size)
        if options is not None:
            self.options_by_name[name] = options

    def configure_relations(self, relation_options: RelationOptions) -> None:
        self.relation_options = relation_options

    def _resolve_options(self, name: str, options: Optional[TensorMonitorOptions]) -> TensorMonitorOptions:
        if options is not None:
            self.options_by_name[name] = options
            return options
        if name in self.options_by_name:
            return self.options_by_name[name]
        return self.default_options

    def _resolve_step(self, name: str, step: Optional[int]) -> int:
        if step is not None:
            resolved = int(step)
        else:
            resolved = int(self._record_counters.get(name, 0))
        self._record_counters[name] = resolved + 1
        return resolved

    def record(self, name, tensor_data, options: Optional[TensorMonitorOptions] = None, step: Optional[int] = None):
        resolved = self._resolve_options(name, options)
        self.register(name)
        resolved_step = self._resolve_step(name, step)

        flat_tensor = flatten_tensor(tensor_data)
        sketch = build_sketch(
            flat_tensor,
            max_rows=resolved.sketch.max_samples,
            max_dims=resolved.sketch.max_dims,
        )

        batch_mean = flat_tensor.mean(dim=0).numpy()
        batch_var = flat_tensor.var(dim=0, unbiased=False).numpy()
        feature_mean = float(flat_tensor.mean().item())
        feature_var = float(flat_tensor.var(unbiased=False).item())

        entry = {
            "step": resolved_step,
            "batch_mean": batch_mean,
            "batch_var": batch_var,
            "feature_mean": feature_mean,
            "feature_var": feature_var,
            "num_samples": int(flat_tensor.size(0)),
            "flat_dim": int(flat_tensor.size(1)),
            "sketch": sketch,
        }
        metrics = set(resolved.metrics)
        if any(metric in metrics for metric in ("spectral", "correlation", "cosine")):
            entry.update(spectral_stats(sketch, eps=resolved.sketch.eps))
        self.features[name].append(entry)

    def _aggregate_window(self, name: str) -> Optional[Dict[str, object]]:
        data_queue = self.features.get(name)
        if not data_queue:
            return None

        window_means = np.stack([item["batch_mean"] for item in data_queue], axis=0)
        window_vars = np.stack([item["batch_var"] for item in data_queue], axis=0)

        sample_mean = np.mean(window_means, axis=0)
        sample_var = np.mean(window_vars, axis=0)
        train_var = np.var(window_means, axis=0)

        scalar_keys = [
            "feature_mean",
            "feature_var",
            "num_samples",
            "flat_dim",
            "effective_rank",
            "participation_ratio",
            "stable_rank",
            "top1_energy_ratio",
            "top2_energy_ratio",
            "dead_dim_ratio",
            "mean_dim_var",
            "max_dim_var",
            "mean_abs_corr",
            "max_abs_corr",
            "sample_cosine_mean",
            "sample_cosine_abs_mean",
        ]
        aggregated = {
            "batch_mean": sample_mean,
            "batch_var": sample_var,
            "train_var": train_var,
        }
        for key in scalar_keys:
            values = [item[key] for item in data_queue if key in item]
            if values:
                aggregated[key] = float(np.mean(values))
        return aggregated

    def get_window_stats(
            self,
            names: Optional[Iterable[str]] = None,
            include_relations: Optional[bool] = None,
            relation_limit: Optional[int] = None,
            relation_names: Optional[Iterable[str]] = None,
    ) -> str:
        if names is None:
            names = list(self.features.keys())
        else:
            names = list(names)

        reports: List[str] = []
        for name in names:
            aggregated = self._aggregate_window(name)
            if aggregated is None:
                continue
            options = self.options_by_name.get(name, self.default_options)
            reports.append(render_tensor_report(name, aggregated, options.display))

        if include_relations is None:
            include_relations = self.relation_options.enabled
        if include_relations:
            relation_text = self.get_relation_stats(
                names=relation_names if relation_names is not None else names,
                max_pairs=relation_limit or self.relation_options.max_pairs,
            )
            if relation_text:
                reports.append(relation_text)
        return "\n".join(reports)

    def _pairwise_relations(self, names: Iterable[str], max_pairs: int) -> List[Dict[str, object]]:
        valid_names = [name for name in names if name in self.features and len(self.features[name]) > 0]
        rows = []
        for i in range(len(valid_names)):
            for j in range(i + 1, len(valid_names)):
                x = self.features[valid_names[i]][-1]["sketch"]
                y = self.features[valid_names[j]][-1]["sketch"]
                common_rows = min(x.size(0), y.size(0))
                if common_rows < 2:
                    continue
                x_aligned = x[:common_rows]
                y_aligned = y[:common_rows]
                eps = self.options_by_name.get(valid_names[i], self.default_options).sketch.eps
                cka = linear_cka(x_aligned, y_aligned, eps=eps)
                overlap_mean, overlap_min = subspace_overlap(x_aligned, y_aligned, rank=self.relation_options.rank)
                rows.append({
                    "tensor_a": valid_names[i],
                    "tensor_b": valid_names[j],
                    "n": int(common_rows),
                    "linear_cka": cka,
                    "subspace_mean_cos": overlap_mean,
                    "subspace_min_cos": overlap_min,
                })
        rows.sort(key=lambda row: row["linear_cka"], reverse=True)
        return rows[:max_pairs]

    def get_relation_stats(self, names: Optional[Iterable[str]] = None, max_pairs: Optional[int] = None) -> str:
        if names is None:
            if self.relation_options.names is not None:
                names = self.relation_options.names
            else:
                names = sorted(self.features.keys())
        else:
            names = list(names)
        rows = self._pairwise_relations(names=names, max_pairs=max_pairs or self.relation_options.max_pairs)
        return render_relation_report(rows)

    def get_scalar_history(self, name: str, key: str) -> Tuple[List[int], List[float]]:
        data_queue = self.features.get(name)
        if not data_queue:
            return [], []
        steps = [int(item.get("step", idx)) for idx, item in enumerate(data_queue)]
        values = [float(item[key]) for item in data_queue if key in item]
        if len(values) != len(steps):
            aligned_steps = []
            aligned_values = []
            for idx, item in enumerate(data_queue):
                if key not in item:
                    continue
                aligned_steps.append(int(item.get("step", idx)))
                aligned_values.append(float(item[key]))
            return aligned_steps, aligned_values
        return steps, values

    def get_vector_history(self, name: str, key: str) -> Tuple[List[int], List[np.ndarray]]:
        data_queue = self.features.get(name)
        if not data_queue:
            return [], []
        steps = []
        values = []
        for idx, item in enumerate(data_queue):
            if key not in item:
                continue
            steps.append(int(item.get("step", idx)))
            values.append(np.asarray(item[key]))
        return steps, values

    def get_step_dim_matrix(self, name: str, key: str) -> Tuple[List[int], np.ndarray]:
        steps, vectors = self.get_vector_history(name, key)
        if not vectors:
            return steps, np.zeros((0, 0), dtype=float)
        return steps, np.stack(vectors, axis=0)


def _build_demo_options() -> Dict[str, TensorMonitorOptions]:
    return {
        "shared_hidden": TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=10,
                topk_display_dims=6,
                rank_by="variance",
            )
        ),
        "specific_hidden": TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=10,
                topk_display_dims=6,
                rank_by="variance",
            )
        ),
        "gate_weights": TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=8,
                topk_display_dims=4,
                rank_by="mean_abs",
            )
        ),
    }


def _demo_record_batches(observatory: TensorObservatory) -> None:
    torch.manual_seed(2026)

    for step in range(4):
        batch_size = 128
        latent = torch.randn(batch_size, 12)
        shared_hidden = torch.cat(
            [
                1.4 * latent[:, :4] + 0.10 * torch.randn(batch_size, 4),
                0.25 * torch.randn(batch_size, 8),
            ],
            dim=1,
        )
        specific_hidden = torch.cat(
            [
                0.35 * latent[:, :4] + 0.25 * torch.randn(batch_size, 4),
                1.1 * torch.randn(batch_size, 8),
            ],
            dim=1,
        )
        gate_logits = torch.stack(
            [
                1.2 + 0.15 * latent[:, 0],
                0.8 + 0.20 * latent[:, 1],
                0.4 + 0.35 * latent[:, 2] + 0.05 * step,
            ],
            dim=-1,
        )
        gate_weights = torch.softmax(gate_logits, dim=-1)

        observatory.record("shared_hidden", shared_hidden)
        observatory.record("specific_hidden", specific_hidden)
        observatory.record("gate_weights", gate_weights)


def main() -> None:
    observatory = TensorObservatory(window_size=4)
    options_by_name = _build_demo_options()
    for name, options in options_by_name.items():
        observatory.register(name, options=options)

    observatory.configure_relations(
        RelationOptions(
            enabled=True,
            rank=4,
            max_pairs=6,
            names=("shared_hidden", "specific_hidden", "gate_weights"),
        )
    )
    _demo_record_batches(observatory)
    print(observatory.get_window_stats())


if __name__ == "__main__":
    main()
