from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Union

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from betterbole.emb import SchemaManager
from .padding import ColumnFormatter, FeatureContext, TensorFormatter


ScannerSource = Union[str, Path, Sequence[Union[str, Path]], pl.LazyFrame]


def _resolve_parquet_paths(source: Union[str, Path, Sequence[Union[str, Path]]]) -> list[str]:
    if isinstance(source, (list, tuple)):
        paths = [str(Path(path)) for path in source]
    else:
        path_obj = Path(source)
        if path_obj.is_dir():
            paths = [str(path) for path in sorted(path_obj.glob("*.parquet"))]
        else:
            paths = [str(path_obj)]

    if not paths:
        raise FileNotFoundError(f"No parquet files found from input: {source}")
    return paths


def _fill_null_scalar(col_type: pa.DataType):
    if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
        return pa.scalar("", type=col_type)
    if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
        return pa.scalar([], type=col_type)
    if pa.types.is_boolean(col_type):
        return pa.scalar(False, type=col_type)
    return pa.scalar(0, type=col_type)


def arrow_table_to_batch_dict(
    table: pa.Table,
    selected_col_names: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    selected_set = set(selected_col_names) if selected_col_names is not None else None
    batch_dict: Dict[str, np.ndarray] = {}
    for col_name, col_data in zip(table.column_names, table.columns):
        if selected_set is not None and col_name not in selected_set:
            continue

        clean_col = pc.fill_null(col_data, _fill_null_scalar(col_data.type))
        batch_dict[col_name] = clean_col.to_numpy(zero_copy_only=False)
    return batch_dict


def _concat_tables(tables: Sequence[pa.Table]) -> Optional[pa.Table]:
    non_empty_tables = [table for table in tables if table is not None and table.num_rows > 0]
    if not non_empty_tables:
        return None
    if len(non_empty_tables) == 1:
        return non_empty_tables[0]
    return pa.concat_tables(non_empty_tables)


class DataScanner:
    def __init__(
        self,
        source: ScannerSource,
        read_cols: Optional[Iterable[str]] = None,
        filter_expr: Optional[pl.Expr] = None,
    ):
        self.source = source
        self.read_cols = list(dict.fromkeys(read_cols or [])) or None
        self.filter_expr = filter_expr
        self.parquet_paths = None if isinstance(source, pl.LazyFrame) else _resolve_parquet_paths(source)
        self.lazy_frame = self._build_lazy_frame(source)

    def _build_lazy_frame(self, source: ScannerSource) -> pl.LazyFrame:
        if isinstance(source, pl.LazyFrame):
            lf = source
        else:
            lf = pl.scan_parquet(self.parquet_paths)

        if self.filter_expr is not None:
            lf = lf.filter(self.filter_expr)
        if self.read_cols is not None:
            lf = lf.select(self.read_cols)
        return lf

    def _apply_worker_shard(self, lf: pl.LazyFrame, worker_id: int, num_workers: int) -> pl.LazyFrame:
        if num_workers <= 1:
            return lf
        worker_col = "__scanner_worker_row_id"
        return (
            lf.with_row_index(worker_col)
            .filter((pl.col(worker_col) % num_workers) == worker_id)
            .drop(worker_col)
        )

    def _iter_lazy_batches(self, batch_size: int, worker_id: int, num_workers: int) -> Iterator[pa.Table]:
        lf = self._apply_worker_shard(self.lazy_frame, worker_id=worker_id, num_workers=num_workers)
        for batch_df in lf.collect_batches(
            chunk_size=batch_size,
            maintain_order=True,
            engine="streaming",
        ):
            if batch_df.height == 0:
                continue
            yield batch_df.to_arrow()

    def _iter_parquet_batches(self, batch_size: int, worker_id: int, num_workers: int) -> Iterator[pa.Table]:
        for file_index, parquet_file in enumerate(self.parquet_paths or []):
            if num_workers > 1 and (file_index % num_workers) != worker_id:
                continue

            parquet_file_reader = pq.ParquetFile(parquet_file)
            for record_batch in parquet_file_reader.iter_batches(
                batch_size=batch_size,
                columns=self.read_cols,
            ):
                yield pa.Table.from_batches([record_batch])

    def iter_batches(
        self,
        batch_size: int,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> Iterator[pa.Table]:
        if self.parquet_paths is not None and self.filter_expr is None:
            yield from self._iter_parquet_batches(batch_size, worker_id, num_workers)
            return

        yield from self._iter_lazy_batches(batch_size, worker_id, num_workers)


class DataTransformer:
    def __init__(
        self,
        manager: SchemaManager,
        preprocess_exprs: Optional[Iterable[pl.Expr]] = None,
        filter_expr: Optional[pl.Expr] = None,
        output_col_names: Optional[Iterable[str]] = None,
    ):
        self.manager = manager
        self.preprocess_exprs = list(preprocess_exprs or [])
        self.filter_expr = filter_expr
        self.output_col_names = list(dict.fromkeys(output_col_names or manager.fields()))

    def __call__(self, raw_table: pa.Table) -> Optional[pa.Table]:
        if raw_table.num_rows == 0:
            return None

        raw_df = pl.from_arrow(raw_table)
        if self.preprocess_exprs:
            raw_df = raw_df.with_columns(self.preprocess_exprs)
        if self.filter_expr is not None:
            raw_df = raw_df.filter(self.filter_expr)
            if raw_df.height == 0:
                return None

        transformed_df = (
            self.manager.transform(raw_df.lazy())
            .select(self.output_col_names)
            .collect(engine="streaming")
        )
        if transformed_df.height == 0:
            return None
        return transformed_df.to_arrow()


class ShuffleBuffer:
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = True,
        selected_col_names: Optional[Sequence[str]] = None,
    ):
        self.capacity = max(batch_size, int(capacity))
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.selected_col_names = list(selected_col_names) if selected_col_names is not None else None
        self.reset()

    def reset(self):
        self.buffer: list[pa.Table] = []
        self.current_rows = 0

    def push(self, table: Optional[pa.Table]):
        if table is None or table.num_rows == 0:
            return
        self.buffer.append(table)
        self.current_rows += table.num_rows

    def _drain(self, force: bool) -> Iterator[Dict[str, np.ndarray]]:
        table = _concat_tables(self.buffer)
        if table is None or table.num_rows == 0:
            self.reset()
            return

        total_rows = table.num_rows
        batch_dict = arrow_table_to_batch_dict(table, selected_col_names=self.selected_col_names)
        indices = np.random.permutation(total_rows) if self.shuffle else np.arange(total_rows)
        num_complete_batches = total_rows // self.batch_size

        for batch_idx in range(num_complete_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]
            yield {col: arr[batch_indices] for col, arr in batch_dict.items()}

        residual_start = num_complete_batches * self.batch_size
        residual_indices = indices[residual_start:]
        residual_count = len(residual_indices)

        self.reset()
        if residual_count == 0:
            return

        if force:
            if not self.drop_last:
                yield {col: arr[residual_indices] for col, arr in batch_dict.items()}
            return

        residual_table = table.take(pa.array(residual_indices, type=pa.int64()))
        self.buffer = [residual_table]
        self.current_rows = residual_table.num_rows

    def yield_batches(self) -> Iterator[Dict[str, np.ndarray]]:
        if self.current_rows < self.capacity:
            return
        yield from self._drain(force=False)

    def flush(self) -> Iterator[Dict[str, np.ndarray]]:
        yield from self._drain(force=True)


class PipelineStreamDataset(IterableDataset):
    def __init__(
        self,
        scanner: DataScanner,
        formatter: TensorFormatter,
        transformer: Optional[DataTransformer] = None,
        buffer: Optional[ShuffleBuffer] = None,
        scan_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.scanner = scanner
        self.transformer = transformer
        self.buffer = buffer
        self.formatter = formatter
        self.scan_batch_size = scan_batch_size or (buffer.batch_size if buffer is not None else 4096)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if self.buffer is not None:
            self.buffer.reset()

        for raw_table in self.scanner.iter_batches(
            batch_size=self.scan_batch_size,
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            table = self.transformer(raw_table) if self.transformer else raw_table
            if table is None or table.num_rows == 0:
                continue

            if self.buffer is None:
                yield self.formatter.format(
                    arrow_table_to_batch_dict(table, selected_col_names=self.formatter.context.output_col_names)
                )
                continue

            self.buffer.push(table)
            for batch_dict in self.buffer.yield_batches():
                yield self.formatter.format(batch_dict)

        if self.buffer is not None:
            for batch_dict in self.buffer.flush():
                yield self.formatter.format(batch_dict)


class ParquetStreamDataset(PipelineStreamDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
    ):
        context = FeatureContext.from_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        scanner = DataScanner(parquet_path, read_cols=context.read_col_names)
        formatter = TensorFormatter(context)
        buffer = ShuffleBuffer(
            capacity=shuffle_buffer_size if shuffle else batch_size,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=context.output_col_names,
        )
        super().__init__(
            scanner=scanner,
            formatter=formatter,
            transformer=None,
            buffer=buffer,
            scan_batch_size=batch_size * 4 if shuffle else batch_size,
        )

        self.parquet_path = parquet_path
        self.parquet_paths = list(scanner.parquet_paths or [])
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.trigger_size = buffer.capacity
        self.context = context
        self.output_col_names = list(context.output_col_names)
        self.valid_col_names = list(context.output_col_names)
        self.read_col_names = list(context.read_col_names)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.col_formatters = formatter.col_formatters


class RawParquetStreamDataset(PipelineStreamDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
        transform_buffer_size: Optional[int] = None,
        raw_filter_expr: Optional[pl.Expr] = None,
        raw_preprocess_exprs: Optional[Iterable[pl.Expr]] = None,
    ):
        preprocess_exprs = list(raw_preprocess_exprs or [])
        scanner_filter_expr = raw_filter_expr if not preprocess_exprs else None
        transformer_filter_expr = None if scanner_filter_expr is not None else raw_filter_expr
        context = FeatureContext.from_raw_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        scanner = DataScanner(
            parquet_path,
            read_cols=context.read_col_names,
            filter_expr=scanner_filter_expr,
        )
        formatter = TensorFormatter(context)
        transformer = DataTransformer(
            manager=manager,
            preprocess_exprs=preprocess_exprs,
            filter_expr=transformer_filter_expr,
            output_col_names=context.output_col_names,
        )
        capacity = transform_buffer_size
        if capacity is None:
            capped_rows = batch_size * (8 if shuffle else 1)
            capacity = max(batch_size, min(shuffle_buffer_size, capped_rows))
        buffer = ShuffleBuffer(
            capacity=capacity,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=context.output_col_names,
        )
        super().__init__(
            scanner=scanner,
            formatter=formatter,
            transformer=transformer,
            buffer=buffer,
            scan_batch_size=batch_size * 2 if shuffle else batch_size,
        )

        self.parquet_path = parquet_path
        self.parquet_paths = list(scanner.parquet_paths or [])
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.trigger_size = buffer.capacity
        self.context = context
        self.output_col_names = list(context.output_col_names)
        self.valid_col_names = list(context.output_col_names)
        self.read_col_names = list(context.read_col_names)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.raw_filter_expr = raw_filter_expr
        self.raw_preprocess_exprs = preprocess_exprs
        self.transform_buffer_size = buffer.capacity
        self.col_formatters = formatter.col_formatters
