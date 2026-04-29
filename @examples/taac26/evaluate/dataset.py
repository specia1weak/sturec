from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import polars as pl


RAW_UID_FIELD = "__raw_user_id"
RAW_IID_FIELD = "__raw_item_id"


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


@dataclass
class TAAC2026Dataset:
    parquet_path: str
    label_field: str = "label_type"
    time_field: str = "timestamp"
    uid_field: str = "user_id"
    iid_field: str = "item_id"
    label_shift: int = 1

    @property
    def whole_lf(self) -> pl.LazyFrame:
        files = [str(path) for path in resolve_parquet_files(self.parquet_path)]
        lf = pl.scan_parquet(files if len(files) > 1 else files[0])
        schema_names = set(lf.collect_schema().names())

        exprs: List[pl.Expr] = []
        if self.label_field in schema_names and self.label_shift != 0:
            exprs.append((pl.col(self.label_field) - self.label_shift).alias(self.label_field))
        if self.uid_field in schema_names and RAW_UID_FIELD not in schema_names:
            exprs.append(pl.col(self.uid_field).alias(RAW_UID_FIELD))
        if self.iid_field in schema_names and RAW_IID_FIELD not in schema_names:
            exprs.append(pl.col(self.iid_field).alias(RAW_IID_FIELD))

        if exprs:
            lf = lf.with_columns(exprs)
        return lf

    def inspect(self, head_rows: int = 5) -> Dict[str, object]:
        lf = self.whole_lf
        schema = lf.collect_schema()
        head_df = lf.head(head_rows).collect()
        return {
            "input_path": self.parquet_path,
            "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in schema.items()],
            "head_rows": head_df.to_dicts(),
        }
