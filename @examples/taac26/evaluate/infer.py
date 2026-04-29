from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
import torch

from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb import SchemaManager
from betterbole.emb.schema import (
    MinMaxDenseSetting,
    QuantileEmbSetting,
    SparseEmbSetting,
    SparseSetEmbSetting,
)
from betterbole.experiment import change_root_workdir


CURRENT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = CURRENT_DIR.parent / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from config import build_cfg  # noqa: E402
from dataset import RAW_UID_FIELD, TAAC2026Dataset  # noqa: E402
from model import SimpleTAACModel  # noqa: E402


change_root_workdir()


FEATURE_META_NAME = "feature_meta.json"
MODEL_STATE_NAME = "model.pt"
RUNTIME_META_NAME = "runtime_meta.json"


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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


def collect_present_raw_feature_fields(lf: pl.LazyFrame, manager: SchemaManager) -> None:
    schema_names = set(lf.collect_schema().names())
    feature_fields = [setting.field_name for setting in manager.settings]
    missing = [field for field in feature_fields if field not in schema_names]
    if missing:
        raise KeyError(
            "Missing raw columns required by current SchemaManager settings: "
            + ", ".join(missing)
        )


def collect_available_encoded_fields(parquet_path: str, manager: SchemaManager) -> List[str]:
    schema_names = set(pl.scan_parquet(parquet_path).collect_schema().names())
    desired = [field for field in manager.fields() if field in schema_names]
    if RAW_UID_FIELD in schema_names and RAW_UID_FIELD not in desired:
        desired.append(RAW_UID_FIELD)
    return desired


def reduce_predictions(
    scored_rows: List[tuple[str, float]],
    strategy: str,
) -> Dict[str, float]:
    if strategy == "overwrite":
        predictions: Dict[str, float] = {}
        for uid_key, score in scored_rows:
            predictions[uid_key] = score
        return predictions

    if strategy == "strict":
        predictions: Dict[str, float] = {}
        for uid_key, score in scored_rows:
            if uid_key in predictions:
                raise ValueError(
                    f"Duplicate raw uid detected during infer: {uid_key}. "
                    "eval_uid_conflict_strategy='strict' forbids repeated user ids."
                )
            predictions[uid_key] = score
        return predictions

    grouped: Dict[str, List[float]] = {}
    for uid_key, score in scored_rows:
        grouped.setdefault(uid_key, []).append(score)

    if strategy == "agg_mean":
        return {uid_key: float(sum(scores) / len(scores)) for uid_key, scores in grouped.items()}
    if strategy == "agg_max":
        return {uid_key: float(max(scores)) for uid_key, scores in grouped.items()}
    if strategy == "agg_last":
        return {uid_key: float(scores[-1]) for uid_key, scores in grouped.items()}

    raise ValueError(f"Unknown eval_uid_conflict_strategy: {strategy}")


@torch.no_grad()
def main() -> None:
    cfg = build_cfg()
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    model_dir = Path(required_env("MODEL_OUTPUT_PATH"))
    eval_data_path = required_env("EVAL_DATA_PATH")
    result_dir = Path(required_env("EVAL_RESULT_PATH"))
    result_dir.mkdir(parents=True, exist_ok=True)

    manager, runtime_meta = build_manager_from_checkpoint(model_dir)

    label_shift = int(runtime_meta["cfg"]["label_shift"])
    dataset = TAAC2026Dataset(parquet_path=eval_data_path, label_shift=label_shift)
    eval_lf = dataset.whole_lf
    collect_present_raw_feature_fields(eval_lf, manager)

    encoded_eval_path = result_dir / "_eval_encoded.parquet"
    manager.transform(eval_lf).sink_parquet(encoded_eval_path)

    valid_fields = collect_available_encoded_fields(str(encoded_eval_path), manager)
    if RAW_UID_FIELD not in valid_fields:
        raise KeyError(
            f"Expected raw uid field {RAW_UID_FIELD!r} in encoded eval parquet, "
            "but it is missing. Refusing to fall back to encoded user_id."
        )
    batch_size = int(runtime_meta["cfg"]["batch_size"])
    eval_ds = ParquetStreamDataset(
        str(encoded_eval_path),
        valid_col_names=valid_fields,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = SimpleTAACModel(manager, runtime_meta["cfg"]["hidden_dims"]).to(cfg.device)
    state_dict = torch.load(model_dir / MODEL_STATE_NAME, map_location=cfg.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    scored_rows: List[tuple[str, float]] = []
    for batch in eval_ds:
        raw_uid_tensor = batch[RAW_UID_FIELD]
        batch = batch.to(cfg.device)
        scores = model.predict(batch).detach().cpu().numpy().tolist()
        raw_uids = raw_uid_tensor.cpu().numpy().tolist()
        for uid, score in zip(raw_uids, scores):
            scored_rows.append((str(uid), float(score)))

    predictions = reduce_predictions(scored_rows, cfg.eval_uid_conflict_strategy)

    output_path = result_dir / "predictions.json"
    save_json({"predictions": predictions}, output_path)
    print(
        f"[eval] wrote predictions to {output_path} "
        f"(rows={len(scored_rows)}, unique_uids={len(predictions)}, "
        f"uid_conflict_strategy={cfg.eval_uid_conflict_strategy})"
    )


if __name__ == "__main__":
    main()
