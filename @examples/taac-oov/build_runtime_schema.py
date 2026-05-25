"""Build a runtime schema from the official schema and probe output.

Input:
- original ``schema.json``
- probe JSON emitted by ``oov_builder.py --output_json ...``

Output:
- ``schema_runtime.json``: official schema plus runtime directives
- ``keep_ids.npz``: per-column keep-id arrays referenced by runtime schema

The runtime schema keeps the original official sections intact:
- ``user_int``
- ``item_int``
- ``user_dense``
- ``seq``

and adds a ``runtime`` section consumed by the dataset adapter.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np


def _resolve_paths(args: argparse.Namespace) -> Tuple[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = args.schema_path or os.path.join(script_dir, "schema.json")
    probe_json = args.probe_json
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema.json not found: {schema_path}")
    if not probe_json or not os.path.exists(probe_json):
        raise FileNotFoundError(f"probe JSON not found: {probe_json}")
    return schema_path, probe_json


def _default_output_paths(args: argparse.Namespace, probe_json: str) -> Tuple[str, str]:
    probe_dir = os.path.dirname(os.path.abspath(probe_json))
    output_schema = args.output_schema or os.path.join(probe_dir, "schema_runtime.json")
    output_keep_npz = args.output_keep_npz or os.path.join(probe_dir, "keep_ids.npz")
    return output_schema, output_keep_npz


def _select_dense_transform(profile: Dict[str, object]) -> Dict[str, object]:
    suggestion = str(profile.get("suggestion", "none"))
    q99 = profile.get("q99_value")

    if suggestion == "clip_log1p_zscore":
        clip_value = float(q99) if q99 is not None else None
        return {
            "transform": "clip_log1p",
            "clip_value": clip_value,
            "clip_quantile": 0.99,
            "suggestion": suggestion,
        }
    if suggestion in {"layernorm_or_standardize", "standardize"}:
        return {
            "transform": "none",
            "suggestion": suggestion,
        }
    return {
        "transform": "none",
        "suggestion": suggestion,
    }


def build_runtime_schema_dict(
    raw_schema: Dict[str, object],
    probe_payload: Dict[str, object],
    keep_ids_artifact_name: str,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    meta = dict(probe_payload.get("meta", {}))
    probe_columns = probe_payload.get("columns", {})
    probe_dense = probe_payload.get("dense_profiles", {})

    keep_arrays: Dict[str, np.ndarray] = {}
    discrete_runtime: Dict[str, object] = {}
    total_compact_vocab = 0
    total_compact_oov_vocab = 0
    total_raw_vocab = 0
    for name, info in probe_columns.items():
        keep_ids = np.asarray(info.get("keep_ids", []), dtype=np.int64)
        keep_arrays[name] = keep_ids
        keep_count = int(info["keep_count"])
        raw_vocab_size = int(info["vocab_size"])
        compact_vocab_size = keep_count
        compact_oov_idx = keep_count + 1
        total_compact_vocab += compact_vocab_size
        total_compact_oov_vocab += compact_oov_idx
        total_raw_vocab += raw_vocab_size
        discrete_runtime[name] = {
            "name": name,
            "fid": int(info["fid"]),
            "source": info["source"],
            "domain": info.get("domain"),
            "kind": info["kind"],
            "truncate_len": int(info["truncate_len"]),
            "raw_vocab_size": raw_vocab_size,
            "raw_oov_idx": int(info["oov_idx"]),
            "keep_ids_key": name,
            "keep_count": keep_count,
            "compact_vocab_size": compact_vocab_size,
            "compact_oov_idx": compact_oov_idx,
            "remap_mode": "keep_ids_to_compact_oov",
        }

    dense_runtime: Dict[str, object] = {}
    for name, profile in probe_dense.items():
        dense_runtime[name] = {
            "name": name,
            "fid": int(profile["fid"]),
            "dim": int(profile["dim"]),
            "rows": int(profile["rows"]),
            "observed_values": int(profile["observed_values"]),
            "padding_ratio": profile.get("padding_ratio"),
            "min_value": profile.get("min_value"),
            "max_value": profile.get("max_value"),
            "mean_value": profile.get("mean_value"),
            "std_value": profile.get("std_value"),
            "q50_value": profile.get("q50_value"),
            "q90_value": profile.get("q90_value"),
            "q95_value": profile.get("q95_value"),
            "q99_value": profile.get("q99_value"),
            "q999_value": profile.get("q999_value"),
            "zero_ratio": profile.get("zero_ratio"),
            "positive_ratio": profile.get("positive_ratio"),
            "negative_ratio": profile.get("negative_ratio"),
            "row_l2_mean": profile.get("row_l2_mean"),
            "row_l2_std": profile.get("row_l2_std"),
            **_select_dense_transform(profile),
        }

    runtime = {
        "version": 1,
        "oov_policy": {
            "enabled": True,
            "mode": "keep_ids_to_compact_oov",
            "min_freq": meta.get("min_freq"),
            "keep_ids_artifact": keep_ids_artifact_name,
        },
        "dense_policy": {
            "enabled": True,
            "mode": "column_transform",
        },
        "embedding_estimate": {
            "raw_total_vocab": total_raw_vocab,
            "compact_total_vocab": total_compact_vocab,
            "compact_total_vocab_with_oov": total_compact_oov_vocab,
            "emb_dim_64_fp32_bytes": total_compact_oov_vocab * 64 * 4,
            "emb_dim_64_fp16_bytes": total_compact_oov_vocab * 64 * 2,
        },
        "seq_max_lens": meta.get("seq_max_lens", {}),
        "discrete": discrete_runtime,
        "dense": dense_runtime,
    }

    runtime_schema = dict(raw_schema)
    runtime_schema["format"] = "raw_parquet_runtime"
    runtime_schema["base_format"] = raw_schema.get("format", "raw_parquet")
    runtime_schema["runtime"] = runtime

    return runtime_schema, keep_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build schema_runtime.json from TAAC probe output")
    parser.add_argument(
        "--schema_path",
        type=str,
        default=None,
        help="Original official schema.json path. Defaults to <script_dir>/schema.json.",
    )
    parser.add_argument(
        "--probe_json",
        type=str,
        required=True,
        help="Full probe JSON emitted by oov_builder.py --output_json ...",
    )
    parser.add_argument(
        "--output_schema",
        type=str,
        default="",
        help="Path to write schema_runtime.json. Defaults next to probe_json.",
    )
    parser.add_argument(
        "--output_keep_npz",
        type=str,
        default="",
        help="Path to write keep_ids.npz. Defaults next to probe_json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema_path, probe_json = _resolve_paths(args)
    output_schema, output_keep_npz = _default_output_paths(args, probe_json)

    with open(schema_path, "r", encoding="utf-8") as f:
        raw_schema = json.load(f)
    with open(probe_json, "r", encoding="utf-8") as f:
        probe_payload = json.load(f)

    output_schema = os.path.abspath(output_schema)
    output_keep_npz = os.path.abspath(output_keep_npz)
    os.makedirs(os.path.dirname(output_schema), exist_ok=True)
    os.makedirs(os.path.dirname(output_keep_npz), exist_ok=True)

    runtime_schema, keep_arrays = build_runtime_schema_dict(
        raw_schema=raw_schema,
        probe_payload=probe_payload,
        keep_ids_artifact_name=os.path.basename(output_keep_npz),
    )

    np.savez_compressed(output_keep_npz, **keep_arrays)
    with open(output_schema, "w", encoding="utf-8") as f:
        json.dump(runtime_schema, f, ensure_ascii=False, indent=2)

    emb_est = runtime_schema["runtime"]["embedding_estimate"]
    print(f"runtime_schema={output_schema}")
    print(f"keep_ids_npz={output_keep_npz}")
    print(f"discrete_columns={len(keep_arrays)}")
    print(f"dense_columns={len(runtime_schema['runtime']['dense'])}")
    print(f"raw_total_vocab={emb_est['raw_total_vocab']}")
    print(f"compact_total_vocab={emb_est['compact_total_vocab']}")
    print(f"compact_total_vocab_with_oov={emb_est['compact_total_vocab_with_oov']}")
    print(f"emb_dim64_fp32_mb={emb_est['emb_dim_64_fp32_bytes'] / 1024 / 1024:.2f}")
    print(f"emb_dim64_fp16_mb={emb_est['emb_dim_64_fp16_bytes'] / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
