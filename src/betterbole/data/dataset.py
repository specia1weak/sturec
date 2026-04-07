from typing import Dict

import pyarrow.parquet as pq
import numpy as np
import torch
import pyarrow.compute as pc
import pyarrow as pa
from torch.utils.data import IterableDataset, DataLoader
from src.betterbole.interaction import Interaction
from src.utils.time import NamedTimer, timer
import torch.nn.utils.rnn as rnn_utils
def convert_to_tensor(data):
    """
    增强版转换函数：完美兼容 uint32 并自动对齐 PyTorch 常用类型。
    """
    # 1. 预处理：如果是含有 uint32 的 NumPy 数组，强制转为 int64 (Long)
    # 因为 PyTorch 不支持 uint32，且 ID 类数据转 int64 最保险（防止溢出）
    if isinstance(data, np.ndarray) and data.dtype == np.uint32:
        data = data.astype(np.int64)
    elif isinstance(data, pd.Series) and data.dtype == "uint32":
        data = data.astype(np.int64)

    elem = data[0]

    # 情况 A：单值/数值数组处理
    if isinstance(elem, (float, int, np.floating, np.integer)):
        new_data = torch.as_tensor(data)

    # 情况 B：序列处理 (例如 tags, history_videos)
    elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
        # 对每一个子序列进行递归或类型防御
        seq_data = []
        for d in data:
            # 防御子序列中的 uint32
            if isinstance(d, np.ndarray) and d.dtype == np.uint32:
                d = d.astype(np.int64)
            seq_data.append(torch.as_tensor(d))
        new_data = rnn_utils.pad_sequence(seq_data, batch_first=True)

    else:
        raise ValueError(f"[{type(elem)}] is not supported!")

    # 3. 类型后置对齐
    # float64 转 float32 是推荐系统的基操，节省一半显存
    if new_data.dtype == torch.float64:
        new_data = new_data.float()
    # uint8 虽然 PyTorch 支持，但在做 Embedding 索引时通常需要 Long
    elif new_data.dtype == torch.uint8:
        new_data = new_data.long()

    return new_data

class ParquetStreamDataset(IterableDataset):
    """
    流式 Parquet 读取器 (极致优化版)
    解决: OOM、行错位、频繁内存深拷贝、强转类型丢失精度等问题
    """
    def __init__(self, parquet_path: str, valid_col_names=None, batch_size: int = 4096, shuffle_and_drop_last=True, shuffle_buffer_size: int = 100000):
        super().__init__()
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.valid_col_names = valid_col_names

        self.shuffle_and_drop_last = shuffle_and_drop_last
        self.shuffle_buffer_size = shuffle_buffer_size
        self.buffer = {}
        self.buffer_len = 0

    def clear_buffer(self):
        self.buffer = {}
        self.buffer_len = 0

    def append_buffer(self, batch_dict):
        chunk_len = len(next(iter(batch_dict.values())))
        for col_name, arr in batch_dict.items():
            if col_name not in self.buffer:
                self.buffer[col_name] = [arr]
            else:
                self.buffer[col_name].append(arr)
        self.buffer_len += chunk_len

    def overwrite_buffer(self, batch_dict):
        self.clear_buffer()
        self.append_buffer(batch_dict)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        pf = pq.ParquetFile(self.parquet_path)
        num_row_groups = pf.metadata.num_row_groups

        if worker_info is None:
            iter_range = range(num_row_groups)
        else:
            per_worker = int(np.ceil(num_row_groups / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_range = range(worker_id * per_worker, min((worker_id + 1) * per_worker, num_row_groups))

        # 使用 list 暂存 array，杜绝在循环中频繁调用 np.concatenate
        self.clear_buffer()

        for i in iter_range:
            row_group = pf.read_row_group(i, columns=self.valid_col_names)
            for batch in row_group.to_batches(max_chunksize=self.batch_size):
                # 小口进水，避免内存峰值暴涨
                batch_dict_tmp = {}
                is_batch_valid = True

                # 1. 原子化解析：如果一列坏了，整个 batch 丢弃，保证严格的行对齐

                for col_name, col_data in zip(batch.schema.names, batch.columns):
                    if self.valid_col_names is not None and col_name not in self.valid_col_names:
                        continue
                    try:
                        # 动态处理空值填充，避免给 string 列填入 0 导致报错
                        col_type = col_data.type
                        if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
                            clean_col = pc.fill_null(col_data, "")
                        elif pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
                            clean_col = pc.fill_null(col_data, pa.scalar([], type=col_type))
                        else:
                            clean_col = pc.fill_null(col_data, 0)

                        # 核心优化：去掉 .astype(np.int64)，让 numpy 自动保留原始类型 (int/float/str)
                        arr = clean_col.to_numpy(zero_copy_only=False)
                        batch_dict_tmp[col_name] = arr
                    except Exception as e:
                        print(col_name)
                        print(e)
                        is_batch_valid = False
                        break

                if not is_batch_valid:
                    continue
                # 2. 存入缓存池
                self.append_buffer(batch_dict_tmp)
                # 3. 水池满后，一次性 Shuffle 并留下没处理完的
                if self.shuffle_and_drop_last and self.buffer_len >= self.shuffle_buffer_size:
                    yield from self._shuffle_and_yield()
                if not self.shuffle_and_drop_last and self.buffer_len >= self.batch_size:
                    yield from self._yield_sequential_batches(is_last=False)

        # 处理文件末尾最后一点残留数据
        if self.shuffle_and_drop_last and self.buffer_len >= self.batch_size:
            yield from self._shuffle_and_yield()
        if not self.shuffle_and_drop_last and self.buffer_len > 0:
            yield from self._yield_sequential_batches(is_last=True)

    def _yield_sequential_batches(self, is_last=False):
        """
        顺序切分并抛出 batch。
        is_last=False: 抛出完整的 batch，不足 batch_size 的部分放回 buffer。
        is_last=True: 处于文件末尾，强制将 buffer 中所有剩余数据作为一个 batch 抛出。
        """
        if self.buffer_len == 0:
            return

        # 1. 仅在此处做一次全量的内存拼接
        merged_buffer = {col: np.concatenate(arr_list) for col, arr_list in self.buffer.items()}

        num_complete_batches = self.buffer_len // self.batch_size
        start_idx = 0
        end_idx = self.batch_size

        # 2. 依次抛出完整的 batch
        for i in range(num_complete_batches):
            batch_dict = {
                col: arr[start_idx:end_idx]
                for col, arr in merged_buffer.items()
            }
            yield Interaction(batch_dict)
            start_idx += self.batch_size
            end_idx += self.batch_size

        # 3. 处理不足一个 batch 的残留数据
        if start_idx < self.buffer_len:
            if is_last:
                # 文件末尾，强制吐出最后的残余部分
                left_batch_dict = {
                    col: arr[start_idx:]
                    for col, arr in merged_buffer.items()
                }
                yield Interaction(left_batch_dict)
                self.clear_buffer()
            else:
                # 尚未结束，存回 buffer 等待下一次拼接
                left_batch_dict = {
                    col: arr[start_idx:].copy()
                    for col, arr in merged_buffer.items()
                }
                self.overwrite_buffer(left_batch_dict)
        else:
            self.clear_buffer()

    def _shuffle_and_yield(self):
        """
        内部方法：将积累的列表拼接、打乱，并以 O(N) 内存开销切分成 batch 吐出
        """
        # 1. 仅在此处做一次全量的内存拼接
        merged_buffer = {col: np.concatenate(arr_list) for col, arr_list in self.buffer.items()}
        indices = np.random.permutation(self.buffer_len)
        num_complete_batches = self.buffer_len // self.batch_size
        # 只遍历完整的 batch，剩下的 index 直接无视
        start_idx = 0
        end_idx = self.batch_size
        for i in range(num_complete_batches):
            batch_indices = indices[start_idx:end_idx]
            batch_dict = {
                col: arr[batch_indices]
                for col, arr in merged_buffer.items()
            }
            inter = Interaction(batch_dict)
            yield inter
            start_idx += self.batch_size
            end_idx += self.batch_size


        batch_indices = indices[start_idx:]
        left_batch_dict = {
            col: arr[batch_indices].copy()
            for col, arr in merged_buffer.items()
        }
        self.overwrite_buffer(left_batch_dict)





# class ParquetStreamDataset(IterableDataset):
#     def __init__(self, parquet_path: str, valid_col_names=None, batch_size: int = 4096):
#         super().__init__()
#         self.parquet_path = parquet_path
#         self.batch_size = batch_size
#         self.valid_col_names = valid_col_names
#
#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         pf = pq.ParquetFile(self.parquet_path)
#         num_row_groups = pf.metadata.num_row_groups
#
#         if worker_info is None:
#             iter_range = range(num_row_groups)
#         else:
#             per_worker = int(np.ceil(num_row_groups / float(worker_info.num_workers)))
#             worker_id = worker_info.id
#             iter_range = range(worker_id * per_worker, min((worker_id + 1) * per_worker, num_row_groups))
#
#         # 遍历每一个天然的 Row Group
#         for i in iter_range:
#             # 1. 极速按需读取，丢弃无用列
#             if self.valid_col_names:
#                 row_group_table = pf.read_row_group(i, columns=self.valid_col_names)
#             else:
#                 row_group_table = pf.read_row_group(i)
#
#             total_len = row_group_table.num_rows
#             if total_len == 0:
#                 continue
#
#             col_arrays = {}
#             is_valid = True
#
#             # 2. 对这一整个 Row Group 只做一次填充和转换，绝不切碎！
#             for col_name in row_group_table.column_names:
#                 try:
#                     col_data = row_group_table.column(col_name)
#                     col_type = col_data.type
#
#                     if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
#                         clean_col = pc.fill_null(col_data, "")
#                     elif pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
#                         clean_col = pc.fill_null(col_data, pa.scalar([], type=col_type))
#                     else:
#                         clean_col = pc.fill_null(col_data, 0)
#
#                     # 转为整块 Numpy 数组
#                     col_arrays[col_name] = clean_col.to_numpy(zero_copy_only=False)
#                 except Exception as e:
#                     print(f"列 {col_name} 解析失败: {e}")
#                     is_valid = False
#                     break
#
#             if not is_valid:
#                 continue
#
#             # 3. 直接在当前 Row Group 内部打乱
#             indices = np.random.permutation(total_len)
#             num_complete_batches = total_len // self.batch_size
#
#             # 4. 丝滑吐出 Batch，绝不等待
#             for b in range(num_complete_batches):
#                 start_idx = b * self.batch_size
#                 end_idx = start_idx + self.batch_size
#                 batch_indices = indices[start_idx:end_idx]
#
#                 batch_dict = {}
#                 for col_name, arr in col_arrays.items():
#                     sliced_arr = arr[batch_indices]
#                     # 直接转为纯净字典 yield 给主进程，在主进程再去套 Interaction 壳子
#                     batch_dict[col_name] = sliced_arr
#
#                 for k in batch_dict:
#                     batch_dict[k] = convert_to_tensor(batch_dict[k])
#                 yield batch_dict

if __name__ == '__main__':
    import pandas as pd
    import os
    def create_dummy_parquet(file_path="dummy_data.parquet", num_rows=250000):
        print(f"正在生成测试文件: {file_path} ({num_rows} 行)...")
        df = pd.DataFrame({
            "user_id": np.random.randint(1, 10000, size=num_rows),
            "item_id": np.random.randint(1, 50000, size=num_rows),
            "label": np.random.randint(0, 2, size=num_rows)
        })
        table = pa.Table.from_pandas(df)
        # 强制分块写入，模拟真实的 Row Group
        pq.write_table(table, file_path, row_group_size=50000)
        print("测试文件生成完毕！\n" + "-" * 40)


    # 1. 准备数据
    test_file = "dummy_data.parquet"
    if not os.path.exists(test_file):
        create_dummy_parquet(test_file, num_rows=250000)  # 生成 25 万条数据

    # 2. 实例化 Dataset
    # 设定 batch_size 为 4000，乱序池为 50000
    dataset = ParquetStreamDataset(
        parquet_path=test_file,
        batch_size=4000,
        shuffle_buffer_size=50000
    )

    # 3. 配置 DataLoader（参数非常关键）
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # 必须为 None，让 Dataset 自己划分 Batch
        num_workers=2,  # 开启 2 个进程并行读取
        prefetch_factor=2,  # 每个进程提前预读 2 个 Batch
        pin_memory=False  # 测试时不占用显存，如果是真实 GPU 训练设为 True
    )

    # 4. 模拟训练循环
    print("开始流式读取数据：")
    step = 0
    total_samples = 0

    for batch in dataloader:
        step += 1
        # batch 就是你 yield 出来的字典（或 Interaction）
        # 因为 batch_size=None，DataLoader 不会给你增加额外的维度
        user_ids = batch['user_id']
        items = batch['item_id']
        labels = batch['label']

        current_batch_size = len(user_ids)
        total_samples += current_batch_size

        print(f"Step {step:02d} | 读取到 Batch 大小: {current_batch_size} | "
              f"User_ID 形状: {user_ids.shape} | 累计读取: {total_samples}")

        # 模拟训练过程中的耗时
        # loss = model(batch)
        # loss.backward()

        if step >= 10:
            print("... (只打印前 10 步验证效果)")
            break

    print("-" * 40)
    print("测试成功！多进程 DataLoader 与流式 Dataset 工作正常。")