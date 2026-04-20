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
    def __init__(
            self,
            whole_lf: pl.LazyFrame,
            user_col: str,
            item_col: str,
            distribution="uniform",
            alpha=1.0
    ):
        # ==========================================
        # 核心优化 1：只执行一次 collect() 进行物化！
        # 将这两列拉入内存，变成 eager DataFrame，后续全部零拷贝复用
        # ==========================================
        self.ui_df = (
            whole_lf.select(
                pl.col(user_col).alias("user_id"),
                pl.col(item_col).alias("item_id")
            )
            .collect()  # <--- 注意：这里变成了 collect()
        )

        # ==========================================
        # 核心优化 2：所有的操作全部基于内存中的 self.ui_df，快如闪电
        # ==========================================

        # 1. 构建候选池：因为数据已经在内存里，.unique().to_numpy() 是毫秒级操作
        self.item_pool = (
            self.ui_df.select("item_id")
            .unique()
            .drop_nulls()
            .filter(pl.col("item_id") > 0)  # 剔除 Padding
            .to_series()
            .to_numpy()
        )

        self.pool_size = len(self.item_pool)
        if self.pool_size == 0:
            raise ValueError("Item pool is empty! Please check your dataset.")

        # 调用父类初始化
        super().__init__(distribution=distribution, alpha=alpha)

    def _uni_sampling(self, sample_num):
        """非连续集合的均匀分布采样"""
        return np.random.choice(self.item_pool, size=sample_num, replace=True)

    def _get_candidates_list(self):
        """获取流行度候选列表"""
        # 直接复用内存里的列，无需 collect
        return (
            self.ui_df.select("item_id")
            .filter(pl.col("item_id") > 0)
            .to_series()
            .to_list()
        )

    def get_used_ids(self):
        """
        获取用户历史交互集合：
        通过 .lazy() 重新进入惰性模式以利用 Polars 的多线程引擎加速 Join
        """
        # 1. 极速获取 max_user_id
        max_u_res = self.ui_df.select(pl.col("user_id").max()).item()
        if max_u_res is None:
            return np.empty(0, dtype=object)

        max_u = int(max_u_res)
        all_users = pl.LazyFrame({"user_id": np.arange(max_u + 1)})

        # 2. 从 eager 重新转为 lazy 执行图，利用向量化引擎极速 join 和 agg
        # 这里的 join 是完全在内存中发生的，不涉及磁盘 IO
        final_df = (
            all_users
            .join(
                self.ui_df.lazy().group_by("user_id").agg(pl.col("item_id").unique().alias("items")),
                on="user_id",
                how="left"
            )
            .with_columns(pl.col("items").fill_null([]))
            .sort("user_id")
            .collect()
        )

        used_ids = np.empty(max_u + 1, dtype=object)
        used_ids[:] = [set(lst) for lst in final_df["items"].to_list()]
        return used_ids
    def sample_by_key_ids(self, key_ids, num, format: Literal['flat', 'listwise']="flat"):
        """
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
    whole_lf = pl.concat([user_lf, item_lf], how="horizontal")

    print("🚀 初始化 PolarsUISampler...")
    start_time = time.time()

    # 初始化会自动触发 get_used_ids 的底层 Polars 计算图
    sampler = PolarsUISampler(
        whole_lf=whole_lf,
        user_col="user_id",
        item_col="item_id",
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