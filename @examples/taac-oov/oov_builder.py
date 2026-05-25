"""Build keep-id maps and dense probes for the official TAAC parquet baseline.

This module does not modify the official loader. It scans the raw parquet
files, counts discrete ids exactly under the same truncation rule as the
official dataset code, and returns the ids whose frequency is at least a chosen
threshold.

Supported discrete columns:
- ``user_int`` scalar columns
- ``user_int`` list<int> columns (only the first ``dim`` elements are counted)
- ``item_int`` scalar columns
- ``item_int`` list<int> columns (only the first ``dim`` elements are counted)
- sequence side-info columns from the 4 sequence domains
  (only the first ``seq_max_len`` elements are counted)

Supported dense probes:
- ``user_dense`` list<float> columns
- value-level min/max/mean/std and approximate quantiles
- row-level L2 norm statistics
- a rule-based transform suggestion for downstream schema design

The official dataset treats ids ``>= vocab_size`` as out-of-bound. That means
the future runtime OOV remap should map non-whitelisted ids to
``oov_idx == vocab_size`` only after the current OOB check, or the OOB check
itself must be adjusted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass(frozen=True)
class DiscreteColumnSpec:
    name: str
    source: str
    fid: int
    vocab_size: int
    kind: str
    truncate_len: int
    domain: Optional[str] = None

    @property
    def oov_idx(self) -> int:
        return self.vocab_size


@dataclass(frozen=True)
class DenseColumnSpec:
    name: str
    fid: int
    dim: int


@dataclass
class OOVColumnMap:
    spec: DiscreteColumnSpec
    keep_ids: np.ndarray
    observed_values: int
    counter_kind: str

    @property
    def keep_count(self) -> int:
        return int(self.keep_ids.size)

    def to_jsonable(self, include_ids: bool = True) -> Dict[str, object]:
        data: Dict[str, object] = {
            "name": self.spec.name,
            "source": self.spec.source,
            "fid": self.spec.fid,
            "domain": self.spec.domain,
            "kind": self.spec.kind,
            "truncate_len": self.spec.truncate_len,
            "vocab_size": self.spec.vocab_size,
            "oov_idx": self.spec.oov_idx,
            "observed_values": self.observed_values,
            "keep_count": self.keep_count,
            "counter_kind": self.counter_kind,
        }
        if include_ids:
            data["keep_ids"] = self.keep_ids.tolist()
        return data


@dataclass
class DenseColumnProfile:
    spec: DenseColumnSpec
    rows: int
    observed_values: int
    total_slots: int
    min_value: Optional[float]
    max_value: Optional[float]
    mean_value: Optional[float]
    std_value: Optional[float]
    q50_value: Optional[float]
    q90_value: Optional[float]
    q95_value: Optional[float]
    q99_value: Optional[float]
    q999_value: Optional[float]
    zero_ratio: Optional[float]
    positive_ratio: Optional[float]
    negative_ratio: Optional[float]
    padding_ratio: Optional[float]
    row_l2_mean: Optional[float]
    row_l2_std: Optional[float]
    row_l2_min: Optional[float]
    row_l2_max: Optional[float]
    suggestion: str
    sample_size: int

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "name": self.spec.name,
            "fid": self.spec.fid,
            "dim": self.spec.dim,
            "rows": self.rows,
            "observed_values": self.observed_values,
            "total_slots": self.total_slots,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "q50_value": self.q50_value,
            "q90_value": self.q90_value,
            "q95_value": self.q95_value,
            "q99_value": self.q99_value,
            "q999_value": self.q999_value,
            "zero_ratio": self.zero_ratio,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "padding_ratio": self.padding_ratio,
            "row_l2_mean": self.row_l2_mean,
            "row_l2_std": self.row_l2_std,
            "row_l2_min": self.row_l2_min,
            "row_l2_max": self.row_l2_max,
            "suggestion": self.suggestion,
            "sample_size": self.sample_size,
        }


@dataclass
class OOVBuildResult:
    min_freq: int
    seq_max_lens: Dict[str, int]
    columns: Dict[str, OOVColumnMap]
    dense_profiles: Dict[str, DenseColumnProfile]
    dense_counter_columns: int
    sparse_counter_columns: int
    dense_counter_bytes: int
    files_scanned: int
    column_chunks: int

    def to_jsonable(self, include_ids: bool = True) -> Dict[str, object]:
        return {
            "meta": {
                "min_freq": self.min_freq,
                "seq_max_lens": self.seq_max_lens,
                "dense_counter_columns": self.dense_counter_columns,
                "sparse_counter_columns": self.sparse_counter_columns,
                "dense_counter_bytes": self.dense_counter_bytes,
                "files_scanned": self.files_scanned,
                "column_chunks": self.column_chunks,
                "head_truncation": True,
            },
            "columns": {
                name: col.to_jsonable(include_ids=include_ids)
                for name, col in self.columns.items()
            },
            "dense_profiles": {
                name: profile.to_jsonable()
                for name, profile in self.dense_profiles.items()
            }
        }

    def summary(self, sample_ids: int = 8) -> str:
        total_keep = sum(col.keep_count for col in self.columns.values())
        non_empty = [col for col in self.columns.values() if col.keep_count > 0]
        lines = [
            "=== KEEP-ID Build Summary ===",
            f"min_freq={self.min_freq}",
            f"seq_max_lens={self.seq_max_lens}",
            (
                f"counter_plan=dense:{self.dense_counter_columns} "
                f"sparse:{self.sparse_counter_columns} "
                f"dense_bytes={self.dense_counter_bytes}"
            ),
            (
                f"files_scanned={self.files_scanned} "
                f"column_chunks={self.column_chunks} "
                f"columns={len(self.columns)} "
                f"columns_with_keep={len(non_empty)} "
                f"total_keep_ids={total_keep}"
            ),
        ]
        for col in sorted(non_empty, key=lambda x: (-x.keep_count, x.spec.name)):
            sample = col.keep_ids[:sample_ids].tolist()
            lines.append(
                f"{col.spec.name}: keep={col.keep_count}, observed={col.observed_values}, "
                f"vocab={col.spec.vocab_size}, oov_idx={col.spec.oov_idx}, "
                f"kind={col.spec.kind}, sample={sample}"
            )
        if self.dense_profiles:
            lines.append("=== Dense Probe Summary ===")
            for profile in sorted(self.dense_profiles.values(), key=lambda x: x.spec.fid):
                lines.append(
                    f"{profile.spec.name}: dim={profile.spec.dim}, observed={profile.observed_values}, "
                    f"pad_ratio={_fmt_float(profile.padding_ratio)}, "
                    f"min={_fmt_float(profile.min_value)}, p99={_fmt_float(profile.q99_value)}, "
                    f"max={_fmt_float(profile.max_value)}, mean={_fmt_float(profile.mean_value)}, "
                    f"std={_fmt_float(profile.std_value)}, row_l2_mean={_fmt_float(profile.row_l2_mean)}, "
                    f"suggest={profile.suggestion}"
                )
        return "\n".join(lines)


def resolve_parquet_files(path_str: str) -> List[str]:
    if os.path.isdir(path_str):
        files = sorted(
            os.path.join(path_str, name)
            for name in os.listdir(path_str)
            if name.endswith(".parquet")
        )
        if not files:
            raise FileNotFoundError(f"No .parquet files found under {path_str}")
        return files
    if os.path.isfile(path_str):
        return [path_str]
    raise FileNotFoundError(f"Parquet path does not exist: {path_str}")


def parse_seq_max_lens_arg(raw: str) -> Dict[str, int]:
    parsed: Dict[str, int] = {}
    if not raw:
        return parsed
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, value = pair.split(":")
        parsed[name.strip()] = int(value.strip())
    return parsed


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.6g}"


def load_schema_specs(
    schema_path: str,
    seq_max_lens: Optional[Dict[str, int]] = None,
    default_seq_max_len: int = 50,
) -> Tuple[List[DiscreteColumnSpec], Dict[str, int]]:
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    specs: List[DiscreteColumnSpec] = []

    for source_key, prefix in (("user_int", "user_int_feats"), ("item_int", "item_int_feats")):
        for fid, vocab_size, dim in raw[source_key]:
            if int(vocab_size) <= 0:
                continue
            specs.append(
                DiscreteColumnSpec(
                    name=f"{prefix}_{fid}",
                    source=source_key,
                    fid=int(fid),
                    vocab_size=int(vocab_size),
                    kind="scalar" if int(dim) == 1 else "list",
                    truncate_len=int(dim),
                )
            )

    resolved_seq_max_lens: Dict[str, int] = {}
    seq_cfg = raw["seq"]
    for domain in sorted(seq_cfg.keys()):
        resolved_seq_max_lens[domain] = int(
            (seq_max_lens or {}).get(domain, default_seq_max_len)
        )
        cfg = seq_cfg[domain]
        ts_fid = cfg.get("ts_fid")
        prefix = cfg["prefix"]
        for fid, vocab_size in cfg["features"]:
            if int(fid) == int(ts_fid):
                continue
            if int(vocab_size) <= 0:
                continue
            specs.append(
                DiscreteColumnSpec(
                    name=f"{prefix}_{fid}",
                    source="seq",
                    fid=int(fid),
                    vocab_size=int(vocab_size),
                    kind="seq",
                    truncate_len=resolved_seq_max_lens[domain],
                    domain=domain,
                )
            )

    return specs, resolved_seq_max_lens


def load_dense_specs(schema_path: str) -> List[DenseColumnSpec]:
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    specs: List[DenseColumnSpec] = []
    for fid, dim in raw.get("user_dense", []):
        specs.append(
            DenseColumnSpec(
                name=f"user_dense_feats_{int(fid)}",
                fid=int(fid),
                dim=int(dim),
            )
        )
    return specs


class _ApproxReservoir:
    """Approximate bounded sampler used for quantile probing."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = max(0, int(capacity))
        self._rng = np.random.default_rng(seed)
        self._buf = np.empty(self.capacity, dtype=np.float32) if self.capacity > 0 else np.empty(0, dtype=np.float32)
        self.size = 0
        self.seen = 0

    def update(self, values: np.ndarray) -> None:
        n = int(values.size)
        self.seen += n
        if n == 0 or self.capacity <= 0:
            return

        vals = values.astype(np.float32, copy=False)
        if self.size < self.capacity:
            take = min(self.capacity - self.size, n)
            if n <= take:
                chosen = vals
            else:
                idx = self._rng.choice(n, size=take, replace=False)
                chosen = vals[idx]
            self._buf[self.size:self.size + take] = chosen
            self.size += take

        if self.size < self.capacity:
            return

        replace_k = min(n, self.capacity, 4096)
        if replace_k <= 0:
            return
        batch_idx = (
            self._rng.choice(n, size=replace_k, replace=False)
            if n > replace_k else np.arange(n)
        )
        sample_idx = self._rng.choice(self.capacity, size=replace_k, replace=False)
        self._buf[sample_idx] = vals[batch_idx]

    def get(self) -> np.ndarray:
        return self._buf[:self.size].copy()


class _DenseAccumulator:
    def __init__(self, spec: DenseColumnSpec, sample_size: int) -> None:
        self.spec = spec
        self.rows = 0
        self.total_slots = 0
        self.observed_values = 0

        self.min_value = math.inf
        self.max_value = -math.inf
        self.sum_value = 0.0
        self.sumsq_value = 0.0
        self.zero_count = 0
        self.positive_count = 0
        self.negative_count = 0

        self.row_l2_min = math.inf
        self.row_l2_max = -math.inf
        self.row_l2_sum = 0.0
        self.row_l2_sumsq = 0.0

        self.sample = _ApproxReservoir(sample_size, seed=spec.fid + 7919)

    def update(self, array) -> None:
        sliced = pc.list_slice(array, start=0, stop=int(self.spec.dim))
        offs = sliced.offsets.to_numpy()
        flat = sliced.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        batch_rows = len(offs) - 1

        self.rows += batch_rows
        self.total_slots += batch_rows * self.spec.dim
        self.observed_values += int(flat.size)

        if flat.size > 0:
            flat64 = flat.astype(np.float64, copy=False)
            self.min_value = min(self.min_value, float(flat.min()))
            self.max_value = max(self.max_value, float(flat.max()))
            self.sum_value += float(flat64.sum())
            self.sumsq_value += float(np.dot(flat64, flat64))
            self.zero_count += int(np.count_nonzero(flat == 0.0))
            self.positive_count += int(np.count_nonzero(flat > 0.0))
            self.negative_count += int(np.count_nonzero(flat < 0.0))
            self.sample.update(flat)

        row_l2 = np.zeros(batch_rows, dtype=np.float64)
        for i in range(batch_rows):
            start = int(offs[i])
            end = int(offs[i + 1])
            if start >= end:
                continue
            seg = flat[start:end].astype(np.float64, copy=False)
            row_l2[i] = math.sqrt(float(np.dot(seg, seg)))

        if batch_rows > 0:
            self.row_l2_min = min(self.row_l2_min, float(row_l2.min()))
            self.row_l2_max = max(self.row_l2_max, float(row_l2.max()))
            self.row_l2_sum += float(row_l2.sum())
            self.row_l2_sumsq += float(np.dot(row_l2, row_l2))

    def finalize(self) -> DenseColumnProfile:
        if self.observed_values > 0:
            mean_value = self.sum_value / self.observed_values
            var_value = max(self.sumsq_value / self.observed_values - mean_value * mean_value, 0.0)
            std_value = math.sqrt(var_value)
            zero_ratio = self.zero_count / self.observed_values
            positive_ratio = self.positive_count / self.observed_values
            negative_ratio = self.negative_count / self.observed_values
            min_value = self.min_value
            max_value = self.max_value
            sample_values = self.sample.get().astype(np.float64, copy=False)
            if sample_values.size > 0:
                q50, q90, q95, q99, q999 = np.quantile(
                    sample_values, [0.5, 0.9, 0.95, 0.99, 0.999]
                ).tolist()
            else:
                q50 = q90 = q95 = q99 = q999 = None
        else:
            mean_value = std_value = None
            zero_ratio = positive_ratio = negative_ratio = None
            min_value = max_value = None
            q50 = q90 = q95 = q99 = q999 = None

        if self.rows > 0:
            row_l2_mean = self.row_l2_sum / self.rows
            row_l2_var = max(self.row_l2_sumsq / self.rows - row_l2_mean * row_l2_mean, 0.0)
            row_l2_std = math.sqrt(row_l2_var)
            row_l2_min = self.row_l2_min
            row_l2_max = self.row_l2_max
            padding_ratio = (
                1.0 - (self.observed_values / self.total_slots)
                if self.total_slots > 0 else None
            )
        else:
            row_l2_mean = row_l2_std = row_l2_min = row_l2_max = None
            padding_ratio = None

        return DenseColumnProfile(
            spec=self.spec,
            rows=self.rows,
            observed_values=self.observed_values,
            total_slots=self.total_slots,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            std_value=std_value,
            q50_value=q50,
            q90_value=q90,
            q95_value=q95,
            q99_value=q99,
            q999_value=q999,
            zero_ratio=zero_ratio,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            padding_ratio=padding_ratio,
            row_l2_mean=row_l2_mean,
            row_l2_std=row_l2_std,
            row_l2_min=row_l2_min,
            row_l2_max=row_l2_max,
            suggestion=_suggest_dense_transform(
                dim=self.spec.dim,
                min_value=min_value,
                q50_value=q50,
                q99_value=q99,
                negative_ratio=negative_ratio,
            ),
            sample_size=self.sample.size,
        )


def _suggest_dense_transform(
    dim: int,
    min_value: Optional[float],
    q50_value: Optional[float],
    q99_value: Optional[float],
    negative_ratio: Optional[float],
) -> str:
    if q99_value is None:
        return "skip_or_inspect_manually"
    if negative_ratio is not None and negative_ratio > 0.05:
        return "standardize_or_layernorm"
    if min_value is not None and min_value >= 0.0:
        median = q50_value if q50_value is not None else 0.0
        if q99_value > 100.0 or (median >= 0.0 and q99_value / max(median + 1e-6, 1e-6) > 100.0):
            return "clip_log1p_zscore"
    if dim >= 64:
        return "layernorm_or_standardize"
    return "standardize"


class _BaseCounter:
    def __init__(self, spec: DiscreteColumnSpec, min_freq: int) -> None:
        self.spec = spec
        self.min_freq = int(min_freq)
        self.observed_values = 0

    def _filter_valid(self, ids: np.ndarray) -> np.ndarray:
        if ids.size == 0:
            return ids
        valid = ids[(ids > 0) & (ids < self.spec.vocab_size)]
        self.observed_values += int(valid.size)
        return valid

    def update(self, ids: np.ndarray) -> None:
        raise NotImplementedError

    def finalize(self) -> OOVColumnMap:
        raise NotImplementedError


class _DenseCounter(_BaseCounter):
    def __init__(self, spec: DiscreteColumnSpec, min_freq: int) -> None:
        super().__init__(spec, min_freq)
        self.counts = np.zeros(spec.vocab_size, dtype=np.uint8)

    def update(self, ids: np.ndarray) -> None:
        valid = self._filter_valid(ids)
        if valid.size == 0:
            return
        uniq, freq = np.unique(valid, return_counts=True)
        merged = self.counts[uniq].astype(np.uint16) + freq.astype(np.uint16)
        self.counts[uniq] = np.minimum(merged, self.min_freq).astype(np.uint8)

    def finalize(self) -> OOVColumnMap:
        keep_mask = self.counts >= self.min_freq
        keep_ids = np.flatnonzero(keep_mask).astype(np.int64, copy=False)
        return OOVColumnMap(
            spec=self.spec,
            keep_ids=keep_ids,
            observed_values=self.observed_values,
            counter_kind="dense",
        )


class _SparseCounter(_BaseCounter):
    def __init__(self, spec: DiscreteColumnSpec, min_freq: int) -> None:
        super().__init__(spec, min_freq)
        self.counts: Dict[int, int] = {}

    def update(self, ids: np.ndarray) -> None:
        valid = self._filter_valid(ids)
        if valid.size == 0:
            return
        uniq, freq = np.unique(valid, return_counts=True)
        for key, count in zip(uniq.tolist(), freq.tolist()):
            merged = self.counts.get(key, 0) + int(count)
            self.counts[int(key)] = min(merged, self.min_freq)

    def finalize(self) -> OOVColumnMap:
        keep_ids = np.array(
            sorted(key for key, count in self.counts.items() if count >= self.min_freq),
            dtype=np.int64,
        )
        return OOVColumnMap(
            spec=self.spec,
            keep_ids=keep_ids,
            observed_values=self.observed_values,
            counter_kind="sparse",
        )


def _plan_counters(
    specs: Iterable[DiscreteColumnSpec],
    min_freq: int,
    dense_budget_mb: int,
) -> Tuple[Dict[str, _BaseCounter], int, int, int]:
    budget_bytes = int(dense_budget_mb) * 1024 * 1024
    dense_bytes = 0
    dense_count = 0
    sparse_count = 0
    counters: Dict[str, _BaseCounter] = {}

    planned = sorted(specs, key=lambda spec: spec.vocab_size)
    for spec in planned:
        need_bytes = int(spec.vocab_size)
        if dense_bytes + need_bytes <= budget_bytes:
            counters[spec.name] = _DenseCounter(spec, min_freq=min_freq)
            dense_bytes += need_bytes
            dense_count += 1
        else:
            counters[spec.name] = _SparseCounter(spec, min_freq=min_freq)
            sparse_count += 1

    return counters, dense_count, sparse_count, dense_bytes


def _extract_scalar_ids(array) -> np.ndarray:
    return array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _extract_truncated_list_ids(array, truncate_len: int) -> np.ndarray:
    sliced = pc.list_slice(array, start=0, stop=int(truncate_len))
    flattened = pc.list_flatten(sliced)
    if len(flattened) == 0:
        return np.empty(0, dtype=np.int64)
    return flattened.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _estimate_column_cost(spec: DiscreteColumnSpec) -> int:
    if spec.kind == "scalar":
        return 1
    if spec.kind == "list":
        return max(2, min(spec.truncate_len, 16))
    return max(8, min(spec.truncate_len, 64))


def _plan_column_chunks(
    specs: List[DiscreteColumnSpec],
    max_chunk_cost: int,
) -> List[List[DiscreteColumnSpec]]:
    chunks: List[List[DiscreteColumnSpec]] = []
    current: List[DiscreteColumnSpec] = []
    current_cost = 0

    ordered = sorted(
        specs,
        key=lambda spec: (
            0 if spec.kind == "scalar" else 1 if spec.kind == "list" else 2,
            -_estimate_column_cost(spec),
            spec.name,
        ),
    )
    for spec in ordered:
        cost = _estimate_column_cost(spec)
        if current and current_cost + cost > max_chunk_cost:
            chunks.append(current)
            current = []
            current_cost = 0
        current.append(spec)
        current_cost += cost

    if current:
        chunks.append(current)
    return chunks


def build_dense_profiles(
    parquet_path: str,
    schema_path: str,
    batch_size: int = 256,
    dense_sample_size: int = 200000,
) -> Dict[str, DenseColumnProfile]:
    dense_specs = load_dense_specs(schema_path)
    if not dense_specs:
        return {}

    files = resolve_parquet_files(parquet_path)
    accumulators = {
        spec.name: _DenseAccumulator(spec, sample_size=dense_sample_size)
        for spec in dense_specs
    }
    dense_columns = [spec.name for spec in dense_specs]

    for file_path in files:
        pf = pq.ParquetFile(file_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            for batch in pf.iter_batches(
                batch_size=batch_size,
                row_groups=[rg_idx],
                columns=dense_columns,
            ):
                name_to_idx = {name: i for i, name in enumerate(batch.schema.names)}
                for spec in dense_specs:
                    col = batch.column(name_to_idx[spec.name])
                    accumulators[spec.name].update(col)

    return {
        name: acc.finalize()
        for name, acc in accumulators.items()
    }


def build_oov_mapping(
    parquet_path: str,
    schema_path: str,
    min_freq: int = 10,
    seq_max_lens: Optional[Dict[str, int]] = None,
    default_seq_max_len: int = 50,
    batch_size: int = 256,
    dense_budget_mb: int = 512,
    max_chunk_cost: int = 200,
    include_dense_probe: bool = True,
    dense_sample_size: int = 200000,
) -> OOVBuildResult:
    specs, resolved_seq_max_lens = load_schema_specs(
        schema_path=schema_path,
        seq_max_lens=seq_max_lens,
        default_seq_max_len=default_seq_max_len,
    )
    files = resolve_parquet_files(parquet_path)
    counters, dense_count, sparse_count, dense_bytes = _plan_counters(
        specs=specs,
        min_freq=min_freq,
        dense_budget_mb=dense_budget_mb,
    )
    column_chunks = _plan_column_chunks(specs, max_chunk_cost=max_chunk_cost)

    for file_path in files:
        pf = pq.ParquetFile(file_path)
        for chunk_specs in column_chunks:
            chunk_columns = [spec.name for spec in chunk_specs]
            for rg_idx in range(pf.metadata.num_row_groups):
                for batch in pf.iter_batches(
                    batch_size=batch_size,
                    row_groups=[rg_idx],
                    columns=chunk_columns,
                ):
                    name_to_idx = {name: i for i, name in enumerate(batch.schema.names)}
                    for spec in chunk_specs:
                        col = batch.column(name_to_idx[spec.name])
                        if spec.kind == "scalar":
                            ids = _extract_scalar_ids(col)
                        else:
                            ids = _extract_truncated_list_ids(col, spec.truncate_len)
                        counters[spec.name].update(ids)

    columns = {
        name: counter.finalize()
        for name, counter in counters.items()
    }
    dense_profiles = (
        build_dense_profiles(
            parquet_path=parquet_path,
            schema_path=schema_path,
            batch_size=batch_size,
            dense_sample_size=dense_sample_size,
        )
        if include_dense_probe else {}
    )
    return OOVBuildResult(
        min_freq=min_freq,
        seq_max_lens=resolved_seq_max_lens,
        columns=columns,
        dense_profiles=dense_profiles,
        dense_counter_columns=dense_count,
        sparse_counter_columns=sparse_count,
        dense_counter_bytes=dense_bytes,
        files_scanned=len(files),
        column_chunks=len(column_chunks),
    )


def remap_with_keep_ids_inplace(
    ids: np.ndarray,
    keep_ids_sorted: np.ndarray,
    oov_idx: int,
) -> np.ndarray:
    """Keep ids found in ``keep_ids_sorted`` and map other positive ids to OOV."""
    if ids.size == 0:
        return ids
    positive = ids > 0
    if not positive.any():
        return ids
    if keep_ids_sorted.size == 0:
        ids[positive] = oov_idx
        return ids
    positive_ids = ids[positive]
    positions = np.searchsorted(keep_ids_sorted, positive_ids)
    in_bounds = positions < keep_ids_sorted.size
    matches = np.zeros(positive_ids.shape, dtype=bool)
    matches[in_bounds] = keep_ids_sorted[positions[in_bounds]] == positive_ids[in_bounds]
    positive_ids[~matches] = oov_idx
    ids[positive] = positive_ids
    return ids


def remap_oov_ids_inplace(
    ids: np.ndarray,
    keep_ids_sorted: np.ndarray,
    oov_idx: int,
) -> np.ndarray:
    """Backward-compatible alias for ``remap_with_keep_ids_inplace``."""
    return remap_with_keep_ids_inplace(ids, keep_ids_sorted, oov_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-column OOV id maps")
    parser.add_argument(
        "--parquet_path",
        type=str,
        default=None,
        help=(
            "Parquet directory or file. Defaults to env TRAIN_DATA_PATH, "
            "otherwise the directory containing schema.json."
        ),
    )
    parser.add_argument(
        "--schema_path",
        type=str,
        default=None,
        help=(
            "Schema JSON path. Defaults to env TRAIN_SCHEMA_PATH, "
            "otherwise <script_dir>/schema.json, then ./schema.json."
        ),
    )
    parser.add_argument("--min_freq", type=int, default=10)
    parser.add_argument("--seq_max_lens", type=str, default="")
    parser.add_argument("--default_seq_max_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dense_budget_mb", type=int, default=512)
    parser.add_argument(
        "--max_chunk_cost",
        type=int,
        default=200,
        help="Upper bound of per-pass column cost. Lower is safer, higher is faster.",
    )
    parser.add_argument(
        "--dense_sample_size",
        type=int,
        default=200000,
        help="Approximate sample size per dense column used for quantile probing.",
    )
    parser.add_argument(
        "--skip_dense_probe",
        action="store_true",
        default=False,
        help="Skip user_dense distribution probing and only build keep-id maps.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to write the full JSON payload.",
    )
    parser.add_argument(
        "--print_json",
        action="store_true",
        default=False,
        help="Print the full JSON payload, including every keep-id list.",
    )
    parser.add_argument(
        "--json_summary_only",
        action="store_true",
        default=False,
        help="Print a compact JSON summary without the raw keep-id lists.",
    )
    return parser.parse_args()


def _resolve_default_paths(args: argparse.Namespace) -> Tuple[str, str]:
    env_parquet = os.environ.get("TRAIN_DATA_PATH")
    env_schema = os.environ.get("TRAIN_SCHEMA_PATH")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_candidates = [
        args.parquet_path,
        env_parquet,
        os.getcwd(),
        script_dir,
    ]
    parquet_path = next((p for p in parquet_candidates if p and os.path.exists(p)), None)
    if parquet_path is None:
        raise FileNotFoundError(
            "Parquet path not found. Pass --parquet_path or set TRAIN_DATA_PATH."
        )

    schema_candidates = [
        args.schema_path,
        env_schema,
        os.path.join(parquet_path, "schema.json"),
        os.path.join(script_dir, "schema.json"),
        os.path.abspath("schema.json"),
    ]
    schema_path = next((p for p in schema_candidates if p and os.path.exists(p)), None)
    if schema_path is None:
        raise FileNotFoundError(
            "schema.json not found. Pass --schema_path, set TRAIN_SCHEMA_PATH, "
            "or place it under TRAIN_DATA_PATH as <data_dir>/schema.json."
        )

    return parquet_path, schema_path


def main() -> None:
    args = parse_args()
    parquet_path, schema_path = _resolve_default_paths(args)
    result = build_oov_mapping(
        parquet_path=parquet_path,
        schema_path=schema_path,
        min_freq=args.min_freq,
        seq_max_lens=parse_seq_max_lens_arg(args.seq_max_lens),
        default_seq_max_len=args.default_seq_max_len,
        batch_size=args.batch_size,
        dense_budget_mb=args.dense_budget_mb,
        max_chunk_cost=args.max_chunk_cost,
        include_dense_probe=not args.skip_dense_probe,
        dense_sample_size=args.dense_sample_size,
    )
    payload = result.to_jsonable(include_ids=True)
    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    if args.print_json:
        print(json.dumps(payload, ensure_ascii=False))
        return
    if args.json_summary_only:
        print(json.dumps(result.to_jsonable(include_ids=False), ensure_ascii=False))
        return
    print(result.summary())


if __name__ == "__main__":
    main()
