import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
from torch.utils.data import IterableDataset, DataLoader

from betterbole.core.interaction import Interaction
from betterbole.emb import SchemaManager
from .padding import DEFAULT_FALLBACK_FORMATTER, build_formatters_from_setting


class ParquetStreamDataset(IterableDataset):
    def __init__(self, parquet_path: str, manager: SchemaManager, batch_size: int = 4096,
                 shuffle: bool = True, drop_last: bool = True, shuffle_buffer_size: int = 2000000):
        """
        基于 PyArrow 的极速流式数据集
        :param manager: 传入 SchemaManager，将自动被“编译”为底层物理列处理规则，告别复杂的按列推断
        """
        super().__init__()
        self.parquet_path = parquet_path
        self.manager = manager
        self.batch_size = batch_size
        self.valid_col_names = manager.fields()
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.trigger_size = shuffle_buffer_size if self.shuffle else self.batch_size

        self.col_formatters = self._compile_formatters()

    def _compile_formatters(self):
        """
        将 schema 编译成列级 formatter，Dataset 只负责调度。
        """
        formatters = {}
        if not self.manager:
            return formatters

        for setting in self.manager.settings:
            formatters.update(build_formatters_from_setting(setting))

        # --- 补充环境/非特征字段 ---
        for ctx_col in self.manager.label_fields + self.manager.domain_fields + (self.manager.time_field,):
            if ctx_col and ctx_col not in formatters:
                formatters[ctx_col] = DEFAULT_FALLBACK_FORMATTER

        return formatters

    def _format_tensor_dict(self, batch_dict):
        """
        🚀 终极组装车间：O(1) 查表进行类型转换和定长 Padding，性能极高。
        """
        tensor_dict = {}
        for col, data in batch_dict.items():
            formatter = self.col_formatters.get(col, DEFAULT_FALLBACK_FORMATTER)

            try:
                tensor_dict[col] = formatter.format(data)
            except Exception as e:
                raise RuntimeError(
                    f"列 [{col}] 在转换为 Tensor 时发生错误，匹配到的 formatter 为: {formatter}。 错误信息: {e}"
                )

        return tensor_dict

    def _process_and_yield_buffer(self, arrow_batches_list, is_last=False):
        if not arrow_batches_list:
            return None

        table = pa.Table.from_batches(arrow_batches_list)
        total_rows = table.num_rows
        if total_rows == 0:
            return None

        # --- 阶段一：PyArrow 清洗与 NumPy 零拷贝视图获取 ---
        batch_dict = {}
        for col_name, col_data in zip(table.column_names, table.columns):
            if self.valid_col_names is not None and col_name not in self.valid_col_names:
                continue

            col_type = col_data.type
            if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
                clean_col = pc.fill_null(col_data, "")
            elif pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
                clean_col = pc.fill_null(col_data, pa.scalar([], type=col_type))
            else:
                clean_col = pc.fill_null(col_data, 0)

            batch_dict[col_name] = clean_col.to_numpy(zero_copy_only=False)

        # --- 阶段二：打乱索引 ---
        if self.shuffle:
            indices = np.random.permutation(total_rows)
        else:
            indices = np.arange(total_rows)

        num_complete_batches = total_rows // self.batch_size
        start_idx = 0

        # --- 阶段三：切片与组装 Interaction ---
        for _ in range(num_complete_batches):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]
            yield_dict = {col: arr[batch_indices] for col, arr in batch_dict.items()}
            tensor_dict = self._format_tensor_dict(yield_dict)
            yield Interaction(tensor_dict)

            start_idx = end_idx

        # --- 阶段四：处理余数 ---
        if start_idx < total_rows:
            if is_last:
                if not self.drop_last:
                    batch_indices = indices[start_idx:]
                    yield_dict = {col: arr[batch_indices] for col, arr in batch_dict.items()}
                    tensor_dict = self._format_tensor_dict(yield_dict)
                    yield Interaction(tensor_dict)
                return None
            else:
                return table.slice(start_idx, total_rows - start_idx)
        return None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        pf = pq.ParquetFile(self.parquet_path)
        num_row_groups = pf.metadata.num_row_groups

        if worker_info is None:
            iter_range = range(num_row_groups)
        else:
            per_worker = int(math.ceil(num_row_groups / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_range = range(worker_id * per_worker, min((worker_id + 1) * per_worker, num_row_groups))

        current_arrow_batches = []
        current_rows = 0

        for i in iter_range:
            read_chunk_size = self.batch_size * 4 if self.shuffle else self.batch_size
            row_group_iter = pf.iter_batches(batch_size=read_chunk_size, row_groups=[i], columns=self.valid_col_names)

            for arrow_batch in row_group_iter:
                current_arrow_batches.append(arrow_batch)
                current_rows += arrow_batch.num_rows

                if current_rows >= self.trigger_size:
                    gen = self._process_and_yield_buffer(current_arrow_batches, is_last=False)
                    residual_table = None
                    while True:
                        try:
                            item = next(gen)
                            if isinstance(item, pa.Table):
                                residual_table = item
                                break
                            yield item
                        except StopIteration:
                            break

                    current_arrow_batches = [residual_table] if residual_table is not None else []
                    current_rows = residual_table.num_rows if residual_table is not None else 0

        if current_rows > 0:
            gen = self._process_and_yield_buffer(current_arrow_batches, is_last=True)
            for item in gen:
                if not isinstance(item, pa.Table):
                    yield item
