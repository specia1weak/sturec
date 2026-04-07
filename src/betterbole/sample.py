from collections import Counter
from typing import Literal

import torch
import numpy as np


class AbstractSampler(object):
    """:class:`AbstractSampler` is a abstract class, all sampler should inherit from it. This sampler supports returning
    a certain number of random value_ids according to the input key_id, and it also supports to prohibit
    certain key-value pairs by setting used_ids.

    Args:
        distribution (str): The string of distribution, which is used for subclass.

    Attributes:
        used_ids (numpy.ndarray): The result of :meth:`get_used_ids`.
    """

    def __init__(self, distribution, alpha):
        self.distribution = ""
        self.alpha = alpha
        self.set_distribution(distribution)
        self.used_ids = self.get_used_ids()

    def set_distribution(self, distribution):
        """Set the distribution of sampler.

        Args:
            distribution (str): Distribution of the negative items.
        """
        self.distribution = distribution
        if distribution == "popularity":
            self._build_alias_table()

    def _uni_sampling(self, sample_num):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError("Method [_uni_sampling] should be implemented")

    def _get_candidates_list(self):
        """Get sample candidates list for _pop_sampling()

        Returns:
            candidates_list (list): a list of candidates id.
        """
        raise NotImplementedError("Method [_get_candidates_list] should be implemented")

    def _build_alias_table(self):
        """Build alias table for popularity_biased sampling."""
        candidates_list = self._get_candidates_list()
        self.prob = dict(Counter(candidates_list))
        self.alias = self.prob.copy()
        large_q = []
        small_q = []
        for i in self.prob:
            self.alias[i] = -1
            self.prob[i] = self.prob[i] / len(candidates_list)
            self.prob[i] = pow(self.prob[i], self.alpha)
        normalize_count = sum(self.prob.values())
        for i in self.prob:
            self.prob[i] = self.prob[i] / normalize_count * len(self.prob)
            if self.prob[i] > 1:
                large_q.append(i)
            elif self.prob[i] < 1:
                small_q.append(i)
        while len(large_q) != 0 and len(small_q) != 0:
            l = large_q.pop(0)
            s = small_q.pop(0)
            self.alias[s] = l
            self.prob[l] = self.prob[l] - (1 - self.prob[s])
            if self.prob[l] < 1:
                small_q.append(l)
            elif self.prob[l] > 1:
                large_q.append(l)

    def _pop_sampling(self, sample_num):
        """Sample [sample_num] items in the popularity-biased distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """

        keys = list(self.prob.keys())
        random_index_list = np.random.randint(0, len(keys), sample_num)
        random_prob_list = np.random.random(sample_num)
        final_random_list = []

        for idx, prob in zip(random_index_list, random_prob_list):
            if self.prob[keys[idx]] > prob:
                final_random_list.append(keys[idx])
            else:
                final_random_list.append(self.alias[keys[idx]])

        return np.array(final_random_list)

    def sampling(self, sample_num):
        """Sampling [sample_num] item_ids.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        if self.distribution == "uniform":
            return self._uni_sampling(sample_num)
        elif self.distribution == "popularity":
            return self._pop_sampling(sample_num)
        else:
            raise NotImplementedError(
                f"The sampling distribution [{self.distribution}] is not implemented."
            )

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used ids. Index is key_id, and element is a set of value_ids.
        """
        raise NotImplementedError("Method [get_used_ids] should be implemented")

    def sample_by_key_ids(self, key_ids, num):
        """Sampling by key_ids.

        Args:
            key_ids (numpy.ndarray or list): Input key_ids.
            num (int): Number of sampled value_ids for each key_id.

        Returns:
            torch.tensor: Sampled value_ids.
            value_ids[0], value_ids[len(key_ids)], value_ids[len(key_ids) * 2], ..., value_id[len(key_ids) * (num - 1)]
            is sampled for key_ids[0];
            value_ids[1], value_ids[len(key_ids) + 1], value_ids[len(key_ids) * 2 + 1], ...,
            value_id[len(key_ids) * (num - 1) + 1] is sampled for key_ids[1]; ...; and so on.
        """
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num
        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids = self.sampling(total_num)
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value_ids[check_list] = value = self.sampling(len(check_list))
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list] = self.sampling(len(check_list))
                check_list = np.array(
                    [
                        i
                        for i, used, v in zip(
                            check_list,
                            self.used_ids[key_ids[check_list]],
                            value_ids[check_list],
                        )
                        if v in used
                    ]
                )
        return torch.tensor(value_ids, dtype=torch.long)

import polars as pl
class PolarsUISampler(AbstractSampler):
    def __init__(self, num_items: int, user_id_lf: pl.LazyFrame, item_id_lf: pl.LazyFrame, distribution="uniform",
                 alpha=1.0):
        """
        Args:
            num_items (int): 物品的总数
            user_id_lf (pl.LazyFrame): 单列 LazyFrame，包含 user_id
            item_id_lf (pl.LazyFrame): 单列 LazyFrame，包含 item_id
            distribution (str): 'uniform' 或 'popularity'
            alpha (float): 流行度参数
        """
        self.num_items = num_items

        # 1. 规范化列名：无视外部传入的具体列名，强行重命名为标准名，防止报错
        u_col = user_id_lf.collect_schema().names()[0]
        i_col = item_id_lf.collect_schema().names()[0]
        u_lf = user_id_lf.rename({u_col: "user_id"})
        i_lf = item_id_lf.rename({i_col: "item_id"})

        # 2. Lazy 水平拼接：此时不发生真实计算，只在 Polars 内部构建计算图
        self.ui_lf = pl.concat([u_lf, i_lf], how="horizontal")

        # 调用父类初始化（会隐式触发下方我们重写的 get_used_ids 和 _get_candidates_list）
        super().__init__(distribution=distribution, alpha=alpha)

    def _uni_sampling(self, sample_num):
        """均匀分布采样：这部分保持 numpy 即可，因为只是生成随机张量"""
        return np.random.randint(0, self.num_items, size=sample_num)

    def _get_candidates_list(self):
        """
        获取流行度候选列表：
        触发一次 collect，提取全量 item 序列。
        """
        return self.ui_lf.select("item_id").collect().to_series().to_list()

    def get_used_ids(self):
        """
        核心优化点：全量使用 Polars 计算每个用户的历史交互 set。
        彻底抛弃 Python 层面的逐行遍历，利用向量化 Join 和 Aggregation 解决问题。
        """
        # 1. 计算全局最大 user_id (触发一次极小的执行)
        max_u_res = self.ui_lf.select(pl.col("user_id").max()).collect().item()
        if max_u_res is None:  # 防御性编程：空数据情况
            return np.empty(0, dtype=object)

        max_u = int(max_u_res)

        # 2. 构建连续的用户骨架 (0 到 max_u)
        # 这一步是为了防止某些中间的 user_id 完全没有交互数据，导致数组索引错位
        all_users = pl.LazyFrame({"user_id": np.arange(max_u + 1)})

        # 3. 核心计算图：分组 -> 聚合去重 -> Left Join -> 填充 Null -> 排序
        final_df = (
            all_users
            .join(
                self.ui_lf.group_by("user_id").agg(pl.col("item_id").unique().alias("items")),
                on="user_id",
                how="left"
            )
            .with_columns(
                pl.col("items").fill_null([])  # 无交互的用户填充为空列表
            )
            .sort("user_id")  # 严格升序，使其与最终生成的 numpy 数组索引天然对应
            .collect()  # 一次性触发底层 Rust 并发执行引擎
        )

        # 4. 转为父类刚需的 Numpy Set 数组
        # Polars 的 items 列现在是一个 list of lists，转成 set 赋值
        used_ids = np.empty(max_u + 1, dtype=object)
        used_ids[:] = [set(lst) for lst in final_df["items"].to_list()]

        return used_ids

    def sample_by_key_ids(self, key_ids, num,  format: Literal['flat', 'listwise']="flat"):
        """
        ...
        Args:
            format (str): 输出格式。
                          'flat' 保持一维张量交错输出 [n0_1, n1_1, n0_2, n1_2]；
                          'listwise' 输出形状为 (batch_size, num) 的矩阵。
        """
        # 1. 调取原生的扁平化结果
        raw_values = super().sample_by_key_ids(key_ids, num)

        # 2. 根据要求进行后处理
        if format == "flat":
            return raw_values
        elif format == "listwise":
            batch_size = len(key_ids)
            # 替下游把最容易踩坑的 view 顺序和转置彻底封装掉
            return raw_values.view(num, batch_size).t().contiguous()
        else:
            raise ValueError(f"不支持的输出格式: {format}")

import time

# 测试代码
if __name__ == "__main__":
    # 1. 模拟推荐系统交互日志
    num_items = 20  # 物品池大小：0 ~ 19

    # 模拟交互序列：
    # User 0: 看了 1, 2
    # User 1: 看了 5, 5, 6 (重复点击)
    # User 2: 完全没有记录 (测试索引断层)
    # User 3: 看了 10
    # User 4: 看了 15, 16, 17
    mock_uids = [0, 0, 1, 1, 1, 3, 4, 4, 4]
    mock_iids = [1, 2, 5, 5, 6, 10, 15, 16, 17]

    # 构建 LazyFrame
    user_lf = pl.LazyFrame({"user_id": mock_uids})
    item_lf = pl.LazyFrame({"item_id": mock_iids})

    print("🚀 初始化 PolarsUISampler...")
    start_time = time.time()

    # 初始化会自动触发 get_used_ids 的底层 Polars 计算图
    sampler = PolarsUISampler(
        num_items=num_items,
        user_id_lf=user_lf,
        item_id_lf=item_lf,
        distribution="uniform"
    )
    print(f"✅ 初始化完成，耗时: {time.time() - start_time:.4f} 秒\n")

    # 测试 1: 检查 used_ids 结构是否完美对齐
    print("--- 测试 1: 检查底层解析的 used_ids ---")
    for uid, items in enumerate(sampler.used_ids):
        print(f"User {uid:2d} 历史交互: {items}")

    print("\n--- 测试 2: 触发负采样 ---")
    # 假设构建 Batch，要为 user_id=0 和 user_id=1 各采样 4 个负样本
    target_users = [0, 1]
    num_neg_per_user = 4

    print(f"目标请求: 为用户 {target_users} 各采样 {num_neg_per_user} 个样本")

    # 调用继承自父类的方法，它内部会循环检查并避开 used_ids
    sampled_negatives = sampler.sample_by_key_ids(key_ids=target_users, num=num_neg_per_user)

    # 父类返回的 tensor 是交错平铺的：
    # [u0_n1, u1_n1, u0_n2, u1_n2, ...]
    neg_array = sampled_negatives.numpy()

    print(f"\n原始输出 Tensor 形状: {sampled_negatives.shape}")
    print(f"原始输出 Tensor 数据: {neg_array}")

    # 还原一下视角，看看每个用户到底分到了什么：
    u0_samples = neg_array[0::2]  # 从索引 0 开始，步长 2
    u1_samples = neg_array[1::2]  # 从索引 1 开始，步长 2

    print(f"\n检查 User 0 采到的样本: {u0_samples} (绝对不会包含 {sampler.used_ids[0]})")
    print(f"检查 User 1 采到的样本: {u1_samples} (绝对不会包含 {sampler.used_ids[1]})")