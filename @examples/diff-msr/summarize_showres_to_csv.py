#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_TARGETS: list[tuple[str, str, str, str]] = [
    ("weighted_auc", "domain_weighted", "auc", "auc"),
    ("macro_auc", "domain_macro", "auc", "auc"),
    ("overall_auc", "overall", "auc", "auc"),
    ("domain0_auc", "domain0", "auc", "auc"),
    ("domain1_auc", "domain1", "auc", "auc"),
    ("domain2_auc", "domain2", "auc", "auc"),
    ("weighted_logloss", "domain_weighted", "logloss", "logloss"),
]

STAGE_SPECS: list[tuple[str, tuple[str, ...]]] = [
    ("stage0_single", ("stage0", "final_test")),
    ("stage1_shared", ("stage1", "test")),
    ("stage4_base", ("stage4_final", "baseline_test")),
    ("stage4_aug", ("stage4_final", "augment_test")),
]


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent
    default_workspace = repo_root.parent.parent / "workspace"
    default_output = repo_root / "workspace_experiment_summary.csv"
    parser = argparse.ArgumentParser(
        description=(
            "Batch-export experiment results into a single CSV using the same "
            "metric slices as showres.py / parse_and_format_experiment_results."
        )
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=default_workspace,
        help="Workspace root to scan recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--exclude-dir-name",
        default="deplecate",
        help="Directory name to prune anywhere under the workspace.",
    )
    parser.add_argument(
        "--pattern",
        default="*experiment_record*.json",
        help="Filename glob for experiment record files.",
    )
    return parser


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def get_nested(data: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def format_number(value: Any, digits: int = 6) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return ""


def format_pct(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return ""


def compute_improvement(stage1_value: Any, stage4_aug_value: Any) -> tuple[str, str]:
    if not isinstance(stage1_value, (int, float)) or not isinstance(stage4_aug_value, (int, float)):
        return "", ""
    abs_improvement = stage4_aug_value - stage1_value
    rel_improvement = ""
    if stage1_value != 0:
        rel_improvement = format_pct((abs_improvement / stage1_value) * 100)
    return format_number(abs_improvement), rel_improvement


def extract_row(record_path: Path, workspace_root: Path) -> dict[str, str]:
    data = json.loads(record_path.read_text(encoding="utf-8"))
    hyperparameters = data.get("hyperparameters", {})
    work_dir = Path(str(data.get("work_dir", record_path.parent.as_posix())))
    domain_specs = data.get("domain_specs", [])
    domain_count_summary = data.get("domain_count_summary", {})

    row: dict[str, str] = {
        "record_path": record_path.resolve().as_posix(),
        "record_relpath": safe_relpath(record_path, workspace_root),
        "experiment_dir": record_path.parent.name,
        "work_dir": work_dir.as_posix(),
        "work_dir_name": work_dir.name,
        "dataset": str(data.get("dataset", "")),
        "custom": str(hyperparameters.get("custom", "")),
        "backbone_name": str(hyperparameters.get("backbone_name", "")),
        "seed": str(data.get("seed", "")),
        "domain_count": str(len(domain_specs)),
        "train_samples_total": str(sum(int(v) for v in domain_count_summary.get("train_samples", {}).values())),
        "train_ple_total": str(sum(int(v) for v in domain_count_summary.get("train_ple", {}).values())),
        "val_ple_total": str(sum(int(v) for v in domain_count_summary.get("val_ple", {}).values())),
        "test_ple_total": str(sum(int(v) for v in domain_count_summary.get("test_ple", {}).values())),
    }

    for stage_label, keys in STAGE_SPECS:
        metrics = get_nested(data, keys, default={})
        for metric_label, domain_key, metric_key, _metric_kind in METRIC_TARGETS:
            metric_value = get_nested(metrics, (domain_key, metric_key))
            row[f"{stage_label}__{metric_label}"] = format_number(metric_value)

    for metric_label, domain_key, metric_key, _metric_kind in METRIC_TARGETS:
        stage1_value = get_nested(data, ("stage1", "test", domain_key, metric_key))
        stage4_aug_value = get_nested(data, ("stage4_final", "augment_test", domain_key, metric_key))
        abs_improvement, rel_improvement = compute_improvement(stage1_value, stage4_aug_value)
        row[f"abs_improv_vs_stage1__{metric_label}"] = abs_improvement
        row[f"rel_improv_pct_vs_stage1__{metric_label}"] = rel_improvement

    return row


def discover_record_files(workspace_root: Path, pattern: str, exclude_dir_name: str) -> list[Path]:
    record_files: list[Path] = []
    for path in workspace_root.rglob(pattern):
        if exclude_dir_name in path.parts:
            continue
        if not path.is_file():
            continue
        record_files.append(path)
    return sorted(record_files)


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_fields = [
        "record_path",
        "record_relpath",
        "experiment_dir",
        "work_dir",
        "work_dir_name",
        "dataset",
        "custom",
        "backbone_name",
        "seed",
        "domain_count",
        "train_samples_total",
        "train_ple_total",
        "val_ple_total",
        "test_ple_total",
    ]
    stage_fields = [
        f"{stage_label}__{metric_label}"
        for stage_label, _keys in STAGE_SPECS
        for metric_label, _domain_key, _metric_key, _metric_kind in METRIC_TARGETS
    ]
    improvement_fields = [
        f"{prefix}__{metric_label}"
        for prefix in ("abs_improv_vs_stage1", "rel_improv_pct_vs_stage1")
        for metric_label, _domain_key, _metric_key, _metric_kind in METRIC_TARGETS
    ]
    discovered_fields = {key for row in rows for key in row.keys()}
    fieldnames = [
        *[field for field in metadata_fields if field in discovered_fields],
        *[field for field in stage_fields if field in discovered_fields],
        *[field for field in improvement_fields if field in discovered_fields],
        *sorted(discovered_fields - set(metadata_fields) - set(stage_fields) - set(improvement_fields)),
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = args.workspace.resolve()
    output_path = args.output.resolve()

    if not workspace_root.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace_root}")

    record_files = discover_record_files(
        workspace_root=workspace_root,
        pattern=args.pattern,
        exclude_dir_name=args.exclude_dir_name,
    )
    if not record_files:
        raise FileNotFoundError(
            f"No record files matched {args.pattern!r} under {workspace_root}"
        )

    rows = [extract_row(record_path, workspace_root) for record_path in record_files]
    rows.sort(
        key=lambda row: (
            row.get("dataset", ""),
            row.get("backbone_name", ""),
            row.get("custom", ""),
            row.get("work_dir_name", ""),
        )
    )
    write_csv(rows, output_path)

    print(f"Exported {len(rows)} experiments to {output_path}")
    for row in rows:
        print(
            f"- {row['work_dir_name']}: "
            f"stage4_aug weighted_auc={row.get('stage4_aug__weighted_auc', '')}, "
            f"improv_vs_stage1={row.get('abs_improv_vs_stage1__weighted_auc', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
