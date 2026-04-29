from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import polars as pl
import torch
from torch import nn

from betterbole.core.enum_type import FeatureSource
from betterbole.core.train.context import TrainerComponents, TrainerDataLoaders
from betterbole.core.train.trainer import BaseTrainer
from betterbole.data.dataset import ParquetStreamDataset
from betterbole.datasets.taac2026 import TAAC2026Dataset
from betterbole.emb import SchemaManager
from betterbole.emb.schema import (
    MinMaxDenseSetting,
    QuantileEmbSetting,
    SparseEmbSetting,
    SparseSetEmbSetting,
)
from betterbole.evaluate.evaluator import Evaluator, LogDecorator
from betterbole.evaluate.manager import EvaluatorManager
from betterbole.experiment import change_root_workdir
from betterbole.models.base import BaseModel


change_root_workdir()


RAW_UID_FIELD = "__raw_user_id"
RAW_IID_FIELD = "__raw_item_id"
FEATURE_META_NAME = "feature_meta.json"
MODEL_STATE_NAME = "model.pt"
RUNTIME_META_NAME = "runtime_meta.json"


@dataclass
class SimConfig:
    mode: str
    batch_size: int
    max_epochs: int
    device: str
    id_emb_dim: int
    sparse_emb_dim: int
    min_freq: int
    hidden_dims: List[int]
    split_strategy: str
    train_ratio: float
    valid_ratio: float
    label_shift: int


class TAACTrainer(BaseTrainer):
    pass


class SimpleTAACModel(BaseModel):
    def __init__(self, manager: SchemaManager, hidden_dims: Sequence[int]) -> None:
        super().__init__(manager)
        self.hidden_dims = list(hidden_dims)
        input_dim = self.omni_embedding.whole.embedding_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev_dim, 1)
        self.label_field = manager.label_field

    def _logits(self, interaction) -> torch.Tensor:
        x = self.omni_embedding.whole(interaction)
        x = self.backbone(x)
        return self.head(x).squeeze(-1)

    def predict(self, interaction):
        return torch.sigmoid(self._logits(interaction))

    def calculate_loss(self, interaction):
        labels = interaction[self.label_field].float()
        logits = self._logits(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None and value != "" else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None and value != "" else default


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def parse_hidden_dims(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def resolve_parquet_files(path_str: str) -> List[Path]:
    path = Path(path_str)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {path}")
        return files
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path does not exist: {path}")


def scan_parquet_input(path_str: str) -> pl.LazyFrame:
    files = [str(path) for path in resolve_parquet_files(path_str)]
    if len(files) == 1:
        return pl.scan_parquet(files[0])
    return pl.scan_parquet(files)


def apply_label_shift(lf: pl.LazyFrame, label_field: str, shift: int) -> pl.LazyFrame:
    schema_names = set(lf.collect_schema().names())
    if shift == 0 or label_field not in schema_names:
        return lf
    return lf.with_columns((pl.col(label_field) - shift).alias(label_field))


def attach_raw_identity_columns(
    lf: pl.LazyFrame,
    uid_field: str = "user_id",
    iid_field: str = "item_id",
) -> pl.LazyFrame:
    schema_names = set(lf.collect_schema().names())
    exprs: List[pl.Expr] = []
    if uid_field in schema_names and RAW_UID_FIELD not in schema_names:
        exprs.append(pl.col(uid_field).alias(RAW_UID_FIELD))
    if iid_field in schema_names and RAW_IID_FIELD not in schema_names:
        exprs.append(pl.col(iid_field).alias(RAW_IID_FIELD))
    if not exprs:
        return lf
    return lf.with_columns(exprs)


def build_default_settings(
    id_emb_dim: int,
    sparse_emb_dim: int,
    min_freq: int,
) -> List[SparseEmbSetting]:
    settings: List[SparseEmbSetting] = [
        SparseEmbSetting("user_id", FeatureSource.USER_ID, id_emb_dim, min_freq=min_freq),
        SparseEmbSetting("item_id", FeatureSource.ITEM_ID, id_emb_dim, min_freq=min_freq),
    ]

    user_cols = TAAC2026Dataset.user_sparse_cols + TAAC2026Dataset.user_mid_sparse_cols
    item_cols = TAAC2026Dataset.item_sparse_cols + TAAC2026Dataset.item_mid_sparse_cols

    for col in user_cols:
        settings.append(SparseEmbSetting(col, FeatureSource.USER, sparse_emb_dim, min_freq=min_freq))
    for col in item_cols:
        settings.append(SparseEmbSetting(col, FeatureSource.ITEM, sparse_emb_dim, min_freq=min_freq))
    return settings


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def write_runtime_bundle(
    ckpt_dir: Path,
    model: nn.Module,
    manager: SchemaManager,
    runtime_meta: Dict[str, Any],
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / MODEL_STATE_NAME)
    manager.save_schema()
    copy_if_exists(manager.meta_filepath, ckpt_dir / FEATURE_META_NAME)
    save_json(runtime_meta, ckpt_dir / RUNTIME_META_NAME)


def load_settings_from_feature_meta(feature_meta_path: Path) -> List[Any]:
    meta_list = load_json(feature_meta_path)
    settings = []
    for item in meta_list:
        setting_type = item["type"]
        if setting_type == "SPARSE":
            settings.append(SparseEmbSetting.from_dict(item))
        elif setting_type == "SPARSE_SET":
            settings.append(SparseSetEmbSetting.from_dict(item))
        elif setting_type == "QUANTILE":
            settings.append(QuantileEmbSetting.from_dict(item))
        elif setting_type == "DENSE":
            settings.append(MinMaxDenseSetting.from_dict(item))
        else:
            raise NotImplementedError(
                f"Unsupported setting type in local sim loader: {setting_type}"
            )
    return settings


def build_manager_from_checkpoint(model_dir: Path) -> tuple[SchemaManager, Dict[str, Any]]:
    runtime_meta = load_json(model_dir / RUNTIME_META_NAME)
    settings = load_settings_from_feature_meta(model_dir / FEATURE_META_NAME)
    manager = SchemaManager(
        settings,
        work_dir=str(model_dir),
        time_field=runtime_meta["time_field"],
        label_fields=runtime_meta["label_fields"],
        domain_fields=runtime_meta["domain_fields"],
    )
    manager.load_schema()
    return manager, runtime_meta


def inspect_dataset(input_path: str, output_path: Path, head_rows: int = 5) -> None:
    lf = scan_parquet_input(input_path)
    schema = lf.collect_schema()
    head_df = lf.head(head_rows).collect()
    payload = {
        "input_path": input_path,
        "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in schema.items()],
        "head_rows": head_df.to_dicts(),
    }
    save_json(payload, output_path)
    print(f"[inspect] wrote report to {output_path}")


def collect_present_raw_feature_fields(lf: pl.LazyFrame, manager: SchemaManager) -> List[str]:
    schema_names = set(lf.collect_schema().names())
    feature_fields = [setting.field_name for setting in manager.settings]
    missing = [field for field in feature_fields if field not in schema_names]
    if missing:
        raise KeyError(
            "Missing raw columns required by current SchemaManager settings: "
            + ", ".join(missing)
        )
    return feature_fields


def collect_available_encoded_fields(parquet_path: str, manager: SchemaManager) -> List[str]:
    schema_names = set(pl.scan_parquet(parquet_path).collect_schema().names())
    desired = [field for field in manager.fields() if field in schema_names]
    for aux_field in (RAW_UID_FIELD, RAW_IID_FIELD):
        if aux_field in schema_names and aux_field not in desired:
            desired.append(aux_field)
    return desired


def train_mode(cfg: SimConfig) -> None:
    train_data_path = required_env("TRAIN_DATA_PATH")
    ckpt_root = Path(required_env("TRAIN_CKPT_PATH"))
    log_dir = Path(required_env("TRAIN_LOG_PATH"))
    tf_events_dir = Path(required_env("TRAIN_TF_EVENTS_PATH"))
    manager_work_dir = Path(env_str("SIM_WORKDIR", str(ckpt_root / "schema_manager")))

    ckpt_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tf_events_dir.mkdir(parents=True, exist_ok=True)
    manager_work_dir.mkdir(parents=True, exist_ok=True)

    set_seed()

    whole_lf = scan_parquet_input(train_data_path)
    whole_lf = apply_label_shift(whole_lf, "label_type", cfg.label_shift)
    whole_lf = attach_raw_identity_columns(whole_lf)

    settings = build_default_settings(cfg.id_emb_dim, cfg.sparse_emb_dim, cfg.min_freq)
    manager = SchemaManager(settings, str(manager_work_dir), time_field="timestamp", label_fields="label_type")

    collect_present_raw_feature_fields(whole_lf, manager)

    train_raw, valid_raw, test_raw = manager.split_dataset(
        whole_lf,
        strategy=cfg.split_strategy,
        train_ratio=cfg.train_ratio,
        valid_ratio=cfg.valid_ratio,
    )

    manager.fit(train_raw)
    train_lf = manager.transform(train_raw).sort("timestamp")
    valid_lf = manager.transform(valid_raw).sort("timestamp")
    test_lf = manager.transform(test_raw).sort("timestamp")

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
        shuffle_buffer_size=2_000_000,
        drop_last=False,
    )
    valid_ds = ParquetStreamDataset(
        valid_path,
        manager.fields(),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    evaluator_manager = EvaluatorManager()
    overall_evaluator = LogDecorator(
        Evaluator("auc"),
        save_path=str(log_dir / "train_eval.log"),
        title="taac_official_sim",
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
    runtime_meta = {
        "format_version": 1,
        "time_field": manager.time_field,
        "label_fields": list(manager.label_fields),
        "domain_fields": list(manager.domain_fields),
        "model": {
            "hidden_dims": cfg.hidden_dims,
        },
        "train": {
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.max_epochs,
            "train_path": train_path,
            "valid_path": valid_path,
            "test_path": test_path,
        },
        "selected_columns": {
            "user_sparse_cols": TAAC2026Dataset.user_sparse_cols,
            "user_mid_sparse_cols": TAAC2026Dataset.user_mid_sparse_cols,
            "item_sparse_cols": TAAC2026Dataset.item_sparse_cols,
            "item_mid_sparse_cols": TAAC2026Dataset.item_mid_sparse_cols,
        },
    }
    write_runtime_bundle(ckpt_dir, model, manager, runtime_meta)
    print(f"[train] wrote checkpoint bundle to {ckpt_dir}")


@torch.no_grad()
def eval_mode(cfg: SimConfig) -> None:
    model_dir = Path(required_env("MODEL_OUTPUT_PATH"))
    eval_data_path = required_env("EVAL_DATA_PATH")
    result_dir = Path(required_env("EVAL_RESULT_PATH"))
    result_dir.mkdir(parents=True, exist_ok=True)

    manager, runtime_meta = build_manager_from_checkpoint(model_dir)

    eval_lf = scan_parquet_input(eval_data_path)
    eval_lf = apply_label_shift(eval_lf, "label_type", cfg.label_shift)
    eval_lf = attach_raw_identity_columns(eval_lf)
    collect_present_raw_feature_fields(eval_lf, manager)

    encoded_eval_path = result_dir / "_eval_encoded.parquet"
    manager.transform(eval_lf).sink_parquet(encoded_eval_path)

    valid_fields = collect_available_encoded_fields(str(encoded_eval_path), manager)
    batch_size = int(runtime_meta["train"]["batch_size"])
    eval_ds = ParquetStreamDataset(
        str(encoded_eval_path),
        valid_col_names=valid_fields,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = SimpleTAACModel(manager, runtime_meta["model"]["hidden_dims"]).to(cfg.device)
    state_dict = torch.load(model_dir / MODEL_STATE_NAME, map_location=cfg.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    predictions: Dict[str, float] = {}
    for batch in eval_ds:
        raw_uid_tensor = batch[RAW_UID_FIELD] if RAW_UID_FIELD in batch.columns else batch[manager.uid_field]
        batch = batch.to(cfg.device)
        scores = model.predict(batch).detach().cpu().numpy().tolist()
        raw_uids = raw_uid_tensor.cpu().numpy().tolist()
        for uid, score in zip(raw_uids, scores):
            predictions[str(uid)] = float(score)

    save_json({"predictions": predictions}, result_dir / "predictions.json")
    print(f"[eval] wrote predictions to {result_dir / 'predictions.json'}")


def build_config(mode: str) -> SimConfig:
    return SimConfig(
        mode=mode,
        batch_size=env_int("SIM_BATCH_SIZE", 4096),
        max_epochs=env_int("SIM_MAX_EPOCHS", 3),
        device=env_str("SIM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        id_emb_dim=env_int("SIM_ID_EMB_DIM", 32),
        sparse_emb_dim=env_int("SIM_SPARSE_EMB_DIM", 16),
        min_freq=env_int("SIM_MIN_FREQ", 10),
        hidden_dims=parse_hidden_dims(env_str("SIM_HIDDEN_DIMS", "256,128")),
        split_strategy=env_str("SIM_SPLIT_STRATEGY", "sequential_ratio"),
        train_ratio=env_float("SIM_TRAIN_RATIO", 0.8),
        valid_ratio=env_float("SIM_VALID_RATIO", 0.1),
        label_shift=env_int("SIM_LABEL_SHIFT", 1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local official-environment simulation for TAAC.")
    parser.add_argument(
        "--mode",
        choices=["inspect", "train", "eval"],
        required=True,
        help="inspect = dump columns/head; train = fit schema + train model; eval = load checkpoint and emit predictions.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args.mode)
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    if args.mode == "inspect":
        inspect_input = os.environ.get("TRAIN_DATA_PATH") or os.environ.get("EVAL_DATA_PATH")
        if not inspect_input:
            raise RuntimeError("inspect mode requires TRAIN_DATA_PATH or EVAL_DATA_PATH")
        output_path = Path(env_str("SIM_INSPECT_OUTPUT", "workspace/taac_official_sim/inspect_report.json"))
        inspect_dataset(inspect_input, output_path)
        return

    if args.mode == "train":
        train_mode(cfg)
        return

    eval_mode(cfg)


if __name__ == "__main__":
    main()
