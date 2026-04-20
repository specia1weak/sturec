import pyarrow.parquet as pq
import numpy as np
import pyarrow.compute as pc
import pyarrow as pa
from torch.utils.data import IterableDataset
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

import torch
from torch.utils.data import IterableDataset, DataLoader
from betterbole.core.interaction import Interaction
import math


class ParquetStreamDataset(IterableDataset):
    def __init__(self, parquet_path: str, valid_col_names=None, batch_size: int = 4096,
                 shuffle: bool = True, drop_last: bool = True, shuffle_buffer_size: int = 2000000):
        """
        :param parquet_path:
        :param valid_col_names:
        :param batch_size:
        :param shuffle:
        :param drop_last:
        :param shuffle_buffer_size: 它占用的是cpu内存
        """
        super().__init__()
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.valid_col_names = valid_col_names

        # 将配置彻底解耦
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 核心逻辑：如果不 shuffle，就不需要大缓冲池，只要攒够一个 batch_size 就可以触发计算
        self.trigger_size = shuffle_buffer_size if self.shuffle else self.batch_size

    def _process_and_yield_buffer(self, arrow_batches_list, is_last=False):
        if not arrow_batches_list:
            return None

        table = pa.Table.from_batches(arrow_batches_list)
        total_rows = table.num_rows
        if total_rows == 0:
            return None

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

        # 根据独立开关决定是否打乱索引
        if self.shuffle:
            indices = np.random.permutation(total_rows)
        else:
            indices = np.arange(total_rows)

        num_complete_batches = total_rows // self.batch_size
        start_idx = 0

        # 吐出所有完整的 Batch
        for _ in range(num_complete_batches):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]

            yield_dict = {col: arr[batch_indices] for col, arr in batch_dict.items()}
            yield Interaction(yield_dict)

            start_idx = end_idx

        # 独立处理残留数据
        if start_idx < total_rows:
            if is_last:
                # 文件彻底结束：根据 drop_last 决定是抛弃还是强行吐出
                if not self.drop_last:
                    batch_indices = indices[start_idx:]
                    yield_dict = {col: arr[batch_indices] for col, arr in batch_dict.items()}
                    yield Interaction(yield_dict)
                return None
            else:
                # 文件未结束：保持 Arrow 零拷贝视图，留给下一个池子
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
            # 根据情况动态调整底层读取步长
            read_chunk_size = self.batch_size * 4 if self.shuffle else self.batch_size
            row_group_iter = pf.iter_batches(batch_size=read_chunk_size,
                                             row_groups=[i],
                                             columns=self.valid_col_names)

            for arrow_batch in row_group_iter:
                current_arrow_batches.append(arrow_batch)
                current_rows += arrow_batch.num_rows

                # 使用动态 trigger_size 判断是否开始处理
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