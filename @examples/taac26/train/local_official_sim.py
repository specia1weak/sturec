from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import polars as pl
import torch

from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb import SchemaManager
from betterbole.evaluate.evaluator import Evaluator, LogDecorator
from betterbole.evaluate.manager import EvaluatorManager
from betterbole.experiment import change_root_workdir

from config import TAACConfig, build_cfg
from dataset import TAAC2026Dataset
from feature_settings import build_sparse_settings
from model import SimpleTAACModel
from trainer import TAACTrainer
from betterbole.core.train.context import TrainerComponents, TrainerDataLoaders


change_root_workdir()


FEATURE_META_NAME = "feature_meta.json"
MODEL_STATE_NAME = "model.pt"
RUNTIME_META_NAME = "runtime_meta.json"


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def write_runtime_bundle(
    ckpt_dir: Path,
    model: torch.nn.Module,
    manager: SchemaManager,
    runtime_meta: Dict[str, Any],
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / MODEL_STATE_NAME)
    manager.save_schema()
    copy_if_exists(manager.meta_filepath, ckpt_dir / FEATURE_META_NAME)
    save_json(runtime_meta, ckpt_dir / RUNTIME_META_NAME)


def collect_present_raw_feature_fields(lf: pl.LazyFrame, manager: SchemaManager) -> None:
    schema_names = set(lf.collect_schema().names())
    feature_fields = [setting.field_name for setting in manager.settings]
    missing = [field for field in feature_fields if field not in schema_names]
    if missing:
        raise KeyError(
            "Missing raw columns required by current SchemaManager settings: "
            + ", ".join(missing)
        )


def build_runtime_meta(cfg: TAACConfig, manager: SchemaManager, train_path: str, valid_path: str, test_path: str) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "cfg": asdict(cfg),
        "time_field": manager.time_field,
        "label_fields": list(manager.label_fields),
        "domain_fields": list(manager.domain_fields),
        "train": {
            "train_path": train_path,
            "valid_path": valid_path,
            "test_path": test_path,
        },
    }


def inspect_mode(cfg: TAACConfig) -> None:
    input_path = os.environ.get("TRAIN_DATA_PATH") or os.environ.get("EVAL_DATA_PATH")
    if not input_path:
        raise RuntimeError("inspect mode requires TRAIN_DATA_PATH or EVAL_DATA_PATH")

    dataset = TAAC2026Dataset(parquet_path=input_path, label_shift=cfg.label_shift)
    if os.environ.get("TRAIN_LOG_PATH"):
        output_path = Path(required_env("TRAIN_LOG_PATH")) / "inspect_report.json"
    else:
        output_path = Path(required_env("EVAL_RESULT_PATH")) / "inspect_report.json"
    save_json(dataset.inspect(), output_path)
    print(f"[inspect] wrote report to {output_path}")


def train_mode(cfg: TAACConfig) -> None:
    train_data_path = required_env("TRAIN_DATA_PATH")
    ckpt_root = Path(required_env("TRAIN_CKPT_PATH"))
    log_dir = Path(required_env("TRAIN_LOG_PATH"))
    tf_events_dir = Path(required_env("TRAIN_TF_EVENTS_PATH"))
    manager_work_dir = ckpt_root / "schema_manager"

    ckpt_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tf_events_dir.mkdir(parents=True, exist_ok=True)
    manager_work_dir.mkdir(parents=True, exist_ok=True)

    dataset = TAAC2026Dataset(parquet_path=train_data_path, label_shift=cfg.label_shift)
    whole_lf = dataset.whole_lf

    settings = build_sparse_settings(cfg)
    manager = SchemaManager(settings, str(manager_work_dir), time_field=dataset.time_field, label_fields=dataset.label_field)
    collect_present_raw_feature_fields(whole_lf, manager)

    train_raw, valid_raw, test_raw = manager.split_dataset(
        whole_lf,
        strategy=cfg.split_strategy,
        train_ratio=cfg.train_ratio,
        valid_ratio=cfg.valid_ratio,
    )

    manager.fit(train_raw)
    train_lf = manager.transform(train_raw).sort(dataset.time_field)
    valid_lf = manager.transform(valid_raw).sort(dataset.time_field)
    test_lf = manager.transform(test_raw).sort(dataset.time_field)

    processed_dir = manager_work_dir / "processed"
    train_path, valid_path, test_path = manager.save_as_dataset(
        train_lf,
        valid_lf,
        test_lf,
        output_dir=str(processed_dir),
        redo=True,
    )

    train_ds = ParquetStreamDataset(
        train_path,
        manager.fields(),
        batch_size=cfg.batch_size,
        shuffle=True,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        drop_last=False,
    )
    valid_ds = ParquetStreamDataset(
        valid_path,
        manager.fields(),
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        drop_last=False,
    )

    evaluator_manager = EvaluatorManager()
    overall_evaluator = LogDecorator(
        Evaluator("auc"),
        save_path=log_dir / "train_eval.log",
        title=cfg.experiment_name,
    )
    evaluator_manager.register("overall_auc", overall_evaluator)

    model = SimpleTAACModel(manager, cfg.hidden_dims).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    trainer = TAACTrainer(
        model=model,
        optimizer=optimizer,
        manager=manager,
        loaders=TrainerDataLoaders(train=train_ds, valid=valid_ds),
        components=TrainerComponents(evaluator_manager=evaluator_manager),
        cfg=cfg,
    )
    trainer.run()

    ckpt_dir = ckpt_root / "model_output"
    runtime_meta = build_runtime_meta(cfg, manager, train_path, valid_path, test_path)
    write_runtime_bundle(ckpt_dir, model, manager, runtime_meta)
    print(f"[train] wrote checkpoint bundle to {ckpt_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local official-environment simulation for TAAC26.")
    parser.add_argument(
        "--mode",
        choices=["inspect", "train"],
        required=True,
        help="inspect = dump columns/head; train = fit schema + train model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_cfg()
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    if args.mode == "inspect":
        inspect_mode(cfg)
        return
    if args.mode == "train":
        train_mode(cfg)
        return


if __name__ == "__main__":
    main()
