import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
from torch.utils.data import IterableDataset, DataLoader

from betterbole.core.interaction import Interaction
from betterbole.emb.schema import EmbType
from betterbole.emb import SchemaManager


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

        self.col_rules = self._compile_rules()

    def _compile_rules(self):
        """
        基于 EmbType 的纯净编译器：配置驱动，解耦类依赖。
        """
        rules = {}
        if not self.manager:
            return rules

        for setting in self.manager.settings:
            pad_side = getattr(setting, "padding_side", "right")
            if setting.emb_type == EmbType.DENSE:
                rules[setting.field_name] = {"type": "DENSE"}

            # --- 2. 单独的变长 Set 序列 (如单独的 genres) ---
            elif setting.emb_type in (EmbType.SPARSE_SEQ, EmbType.SPARSE_SET):
                # 实际上这个类似不指定padding side也没事
                rules[setting.field_name] = {
                    "type": "2D_SEQ",
                    "max_len": getattr(setting, "max_len", 10),
                    "padding_side": pad_side
                }

            # --- 3. 共享词表的序列 (核心：根据 Target 类型降维打击) ---
            elif setting.emb_type == EmbType.SHARE_SEQ:
                if setting.target_setting.emb_type == EmbType.SPARSE_SET:
                    rules[setting.field_name] = {
                        "type": "3D_SEQ",
                        "max_seq_len": setting.max_len,
                        "max_tag_len": getattr(setting.target_setting, "max_len", 3),
                        "padding_side": pad_side # ✨ 新增
                    }
                else:
                    rules[setting.field_name] = {
                        "type": "2D_SEQ",
                        "max_len": setting.max_len,
                        "padding_side": pad_side # ✨ 新增
                    }

            # --- 4. 兼容处理老旧的 SEQ_GROUP (如果还有残留的话) ---
            elif setting.emb_type == EmbType.SEQ_GROUP:
                for seq_col, target_setting in setting.target_dict.items():
                    if target_setting.emb_type == EmbType.SPARSE_SET:
                        rules[seq_col] = {"type": "3D_SEQ", "max_seq_len": setting.max_len,
                                          "max_tag_len": getattr(target_setting, "max_len", 3)}
                    else:
                        rules[seq_col] = {"type": "2D_SEQ", "max_len": setting.max_len}

            # --- 5. 常规 1D 离散特征兜底 ---
            elif setting.emb_type in (EmbType.SPARSE, EmbType.QUANTILE, EmbType.ABS_RANGE):
                rules[setting.field_name] = {"type": "1D_INT"}

            else:
                rules[setting.field_name] = {"type": "1D_INT"}

        # --- 补充环境/非特征字段 ---
        for ctx_col in self.manager.label_fields + self.manager.domain_fields + (self.manager.time_field,):
            if ctx_col and ctx_col not in rules:
                rules[ctx_col] = {"type": "FALLBACK"}

        return rules

    def _format_tensor_dict(self, batch_dict):
        """
        🚀 终极组装车间：O(1) 查表进行类型转换和定长 Padding，性能极高。
        """
        tensor_dict = {}
        for col, data in batch_dict.items():
            rule = self.col_rules.get(col, {"type": "FALLBACK"})
            rule_type = rule["type"]

            try:
                # --- 1. 连续值特征 (Dense) ---
                if rule_type == "DENSE":
                    tensor_dict[col] = torch.tensor(data, dtype=torch.float32)

                # --- 2. 一维离散 ID ---
                elif rule_type == "1D_INT":
                    if data.dtype == np.uint32:
                        data = data.astype(np.int64)  # 终结 PyTorch uint32 报错
                    tensor_dict[col] = torch.tensor(data, dtype=torch.long)

                # --- 3. 2D 序列 (Id_seq / Tags) ---
                elif rule_type == "2D_SEQ":
                    arr = self._pad_list(
                        data,
                        max_len=rule["max_len"],
                        padding_side=rule.get("padding_side", "right") # ✨ 传入
                    )
                    tensor_dict[col] = torch.tensor(arr, dtype=torch.long)

                # --- 4. 3D 嵌套序列 (Tags_seq) ---
                elif rule_type == "3D_SEQ":
                    arr = self._pad_nested_list(
                        data,
                        max_seq_len=rule["max_seq_len"],
                        max_tag_len=rule["max_tag_len"],
                        padding_side=rule.get("padding_side", "right")
                    )
                    tensor_dict[col] = torch.tensor(arr, dtype=torch.long)

                # --- 5. 兜底处理 (Label / Time 等) ---
                elif rule_type == "FALLBACK":
                    if data.dtype in (np.float32, np.float64):
                        tensor_dict[col] = torch.tensor(data, dtype=torch.float32)
                    elif data.dtype == np.uint32:
                        tensor_dict[col] = torch.tensor(data.astype(np.int64), dtype=torch.long)
                    else:
                        tensor_dict[col] = torch.tensor(data)

            except Exception as e:
                raise RuntimeError(f"列 [{col}] 在转换为 Tensor 时发生错误，匹配到的规则为: {rule}。 错误信息: {e}")

        return tensor_dict

    def _pad_list(self, sequences, max_len, padding_side="right", pad_val=0):
        padded = np.full((len(sequences), max_len), pad_val, dtype=np.int64)
        for i, seq in enumerate(sequences):
            if seq is None or (isinstance(seq, float) and np.isnan(seq)) or len(seq) == 0:
                continue

            seq_list = list(seq)
            valid_len = min(len(seq_list), max_len)
            trunc_seq = seq_list[-valid_len:]
            if padding_side == "right":
                padded[i, :valid_len] = trunc_seq
            elif padding_side == "left":
                padded[i, max_len - valid_len:] = trunc_seq
            else:
                raise ValueError(f"Unknown padding_side: {padding_side}")

        return padded

    def _pad_nested_list(self, sequences, max_seq_len, max_tag_len, padding_side="right", pad_val=0):
        padded = np.full((len(sequences), max_seq_len, max_tag_len), pad_val, dtype=np.int64)
        for i, seq in enumerate(sequences):
            if seq is None or (isinstance(seq, float) and np.isnan(seq)) or len(seq) == 0:
                continue

            seq_list = list(seq)
            valid_seq_len = min(len(seq_list), max_seq_len)
            trunc_seq = seq_list[-valid_seq_len:]
            start_idx = 0 if padding_side == "right" else max_seq_len - valid_seq_len

            for j in range(valid_seq_len):
                tags = trunc_seq[j]
                if tags is None or len(tags) == 0:
                    continue

                tags_list = list(tags)
                valid_tag_len = min(len(tags_list), max_tag_len)
                trunc_tags = tags_list[:valid_tag_len]
                padded[i, start_idx + j, :valid_tag_len] = trunc_tags

        return padded

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