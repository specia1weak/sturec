import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union


def plot_bias_distributions(
        lf: pl.LazyFrame,
        uid_col: str = "user_id",
        iid_col: str = "item_id",
        label_col: str = "label",
        save_path: Union[str, Path] = "bias_distribution.png",
        min_item_interactions: int = 10
):
    """
    计算并绘制用户偏置（User Bias）和物品偏置（Item Bias）的分布直方图。

    参数:
        lf: 包含交互数据的 Polars LazyFrame。
        uid_col: 用户 ID 列名。
        iid_col: 物品 ID 列名。
        label_col: 标签列名（应为 0 或 1 的二分类目标）。
        save_path: 图片保存路径。
        min_item_interactions: 物品的最少交互次数阈值，用于在绘图时过滤长尾冷门物品以避免视觉噪音。
    """
    print(f"开始计算 {uid_col} 和 {iid_col} 的偏置分布...")

    # 1. 计算 User Bias: 每个用户的正样本率
    user_bias_df = lf.group_by(uid_col).agg(
        pl.len().alias("total_inter"),
        pl.col(label_col).sum().alias("pos_inter")
    ).with_columns(
        (pl.col("pos_inter") / pl.col("total_inter")).alias("user_bias_ratio")
    ).collect()

    # 2. 计算 Item Bias: 每个物品的正样本率
    item_bias_df = lf.group_by(iid_col).agg(
        pl.len().alias("total_inter"),
        pl.col(label_col).sum().alias("pos_inter")
    ).with_columns(
        (pl.col("pos_inter") / pl.col("total_inter")).alias("item_bias_ratio")
    ).collect()

    # 打印基础统计信息
    print(f"\n--- User Bias ({uid_col}) 统计 ---")
    print(user_bias_df.select("user_bias_ratio").describe())

    print(f"\n--- Item Bias ({iid_col}) 统计 ---")
    print(item_bias_df.select("item_bias_ratio").describe())

    # 3. 绘制分布直方图
    plt.figure(figsize=(12, 5))

    # User Bias 子图
    plt.subplot(1, 2, 1)
    plt.hist(user_bias_df["user_bias_ratio"].to_numpy(), bins=50, color='skyblue', edgecolor='black')
    plt.title(f'User Bias Distribution\n(Ratio of {label_col}=1 per user)')
    plt.xlabel('Positive Ratio')
    plt.ylabel('Number of Users')

    # Item Bias 子图
    plt.subplot(1, 2, 2)
    item_bias_filtered = item_bias_df.filter(pl.col("total_inter") > min_item_interactions)
    plt.hist(item_bias_filtered["item_bias_ratio"].to_numpy(), bins=50, color='salmon', edgecolor='black')
    plt.title(f'Item Bias Distribution\n(Ratio of {label_col}=1 per item)')
    plt.xlabel('Positive Ratio')
    plt.ylabel(f'Number of Items (Interactions > {min_item_interactions})')

    plt.tight_layout()

    # 确保保存目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # 释放内存

    print(f"偏置分布直方图已保存至: {save_path}")



def plot_sparsity_distributions(
        lf: pl.LazyFrame,
        uid_col: str = "user_id",
        iid_col: str = "item_id",
        save_path: Union[str, Path] = "sparsity_distribution.png",
        log_y: bool = True
):
    """
    计算并绘制用户和物品的交互次数分布，以评估数据的稀疏程度和长尾效应。

    参数:
        lf: 包含交互数据的 Polars LazyFrame。
        uid_col: 用户 ID 列名。
        iid_col: 物品 ID 列名。
        save_path: 图片保存路径。
        log_y: 是否对 Y 轴（频数）使用对数刻度。强烈建议在推荐系统数据中开启，以应对长尾分布。
    """
    print(f"开始计算 {uid_col} 和 {iid_col} 的交互稀疏度分布...")

    # 1. 计算每个用户的交互次数 (User Activity)
    user_inter_df = lf.group_by(uid_col).agg(
        pl.len().alias("interaction_count")
    ).collect()

    # 2. 计算每个物品的交互次数 (Item Popularity)
    item_inter_df = lf.group_by(iid_col).agg(
        pl.len().alias("interaction_count")
    ).collect()

    # 打印基础统计信息（分位数对于观察稀疏度非常有用）
    print(f"\n--- User Interactions ({uid_col}) 统计 ---")
    print(user_inter_df.select("interaction_count").describe())

    print(f"\n--- Item Interactions ({iid_col}) 统计 ---")
    print(item_inter_df.select("interaction_count").describe())

    # 3. 绘制分布直方图
    plt.figure(figsize=(12, 5))

    # User 交互次数分布子图
    plt.subplot(1, 2, 1)
    plt.hist(user_inter_df["interaction_count"].to_numpy(), bins=50, color='mediumpurple', edgecolor='black', log=log_y)
    plt.title('User Interaction Distribution\n(Activity per user)')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users' + (' (Log Scale)' if log_y else ''))

    # Item 交互次数分布子图
    plt.subplot(1, 2, 2)
    plt.hist(item_inter_df["interaction_count"].to_numpy(), bins=50, color='lightgreen', edgecolor='black', log=log_y)
    plt.title('Item Interaction Distribution\n(Popularity per item)')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Items' + (' (Log Scale)' if log_y else ''))

    plt.tight_layout()

    # 确保保存目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # 释放内存

    print(f"稀疏度分布直方图已保存至: {save_path}")


def plot_sparsity_ecdf(
        lf: pl.LazyFrame,
        uid_col: str = "user_id",
        iid_col: str = "item_id",
        save_path: Union[str, Path] = "sparsity_ecdf.png"
):
    """
    计算并绘制用户和物品交互次数的经验累积分布函数 (ECDF) 图。
    该图表专门用于直观读取 50%、75%、90% 等分位数指标，非常适合极端长尾的数据集。

    参数:
        lf: 包含交互数据的 Polars LazyFrame。
        uid_col: 用户 ID 列名。
        iid_col: 物品 ID 列名。
        save_path: 图片保存路径。
    """
    print(f"开始计算 {uid_col} 和 {iid_col} 的累积分布(ECDF)...")

    # 1. 统计每个用户/物品的交互次数并转为 NumPy 数组
    user_counts = lf.group_by(uid_col).agg(pl.len().alias("count")).collect()["count"].to_numpy()
    item_counts = lf.group_by(iid_col).agg(pl.len().alias("count")).collect()["count"].to_numpy()

    # 2. 绘图设置
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def plot_ecdf(ax, data, title):
        # 排序数据以计算 ECDF
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)

        # 提取关键分位数的精确值
        p50 = np.percentile(x, 50)
        p75 = np.percentile(x, 75)
        p90 = np.percentile(x, 90)

        # 绘制 ECDF 阶梯曲线
        ax.step(x, y, where='post', color='dodgerblue', linewidth=2)

        # 绘制水平参考线 (Y轴累积比例)
        ax.axhline(0.50, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.75, color='gray', linestyle='--', alpha=0.5)

        # 绘制垂直参考线及图例 (X轴具体的交互次数)
        ax.axvline(p50, color='red', linestyle=':', linewidth=2, label=f'50% (Median) = {p50:.0f}')
        ax.axvline(p75, color='green', linestyle=':', linewidth=2, label=f'75% = {p75:.0f}')
        ax.axvline(p90, color='darkorange', linestyle=':', linewidth=2, label=f'90% = {p90:.0f}')

        # 核心设置：X轴使用对数刻度以展开高度集中的长尾数据
        ax.set_xscale('log')

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Number of Interactions (Log Scale)', fontsize=10)
        ax.set_ylabel('Cumulative Proportion', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, which="both") # 开启网格线以便更精准对齐

    # 分别绘制用户和物品的 ECDF
    plot_ecdf(axes[0], user_counts, f'User ECDF ({uid_col})')
    plot_ecdf(axes[1], item_counts, f'Item ECDF ({iid_col})')

    plt.tight_layout()

    # 确保保存目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"ECDF累积分布图已保存至: {save_path}")


from matplotlib.ticker import PercentFormatter
def plot_power2_sparsity(
        lf: pl.LazyFrame,
        uid_col: str = "user_id",
        iid_col: str = "item_id",
        save_path: Union[str, Path] = "sparsity_power2.png"
):
    """
    计算并绘制按 2^x 区间分桶的稀疏度分布图。
    横坐标为交互次数的 2^x 对数区间，纵坐标为落入该区间的用户/物品百分比。
    """
    print(f"开始计算 {uid_col} 和 {iid_col} 的 2^x 区间百分比分布...")

    # 1. 统计交互次数
    user_counts = lf.group_by(uid_col).agg(pl.len().alias("count")).collect()["count"].to_numpy()
    item_counts = lf.group_by(iid_col).agg(pl.len().alias("count")).collect()["count"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    def plot_subplot(ax, data, title, color):
        # 交互次数转为 log2 以实现 2^x 分桶
        log2_data = np.log2(data)

        # 确定最大需要的指数（确保最大值能被包含在最后一个桶的左边界）
        max_power = int(np.floor(np.log2(data.max()))) + 1
        bins = np.arange(0, max_power + 1)

        # 计算权重以显示百分比（每个样本的权重为 1/总数 * 100）
        weights = np.ones_like(log2_data) / len(log2_data) * 100

        # 绘制直方图，rwidth 留出柱子间距让区间感更强
        counts, edges, patches = ax.hist(
            log2_data,
            bins=bins,
            weights=weights,
            color=color,
            edgecolor='black',
            rwidth=0.85
        )

        # 设置 X 轴的刻度与标签（显示为 2^0, 2^1, 2^2...）
        ax.set_xticks(bins)
        ax.set_xticklabels([f'$2^{{{i}}}$' for i in bins])

        # 设置 Y 轴为百分比格式
        ax.yaxis.set_major_formatter(PercentFormatter())

        # 在柱子上标注具体的百分比数值（过滤掉太小的，避免遮挡）
        for i in range(len(counts)):
            if counts[i] > 0.5:  # 大于 0.5% 才标注数值，保持图面整洁
                # 横坐标取柱子中心，纵坐标取柱子高度加一点点
                ax.text(edges[i] + 0.5, counts[i] + 0.5, f'{counts[i]:.1f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(f'Interaction Count Range $[2^x, 2^{{x+1}})$', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)

        # 添加水平参考网格
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # 移除顶部和右侧的边框线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 2. 分别绘制 User 和 Item
    plot_subplot(axes[0], user_counts, f'User Distribution ({uid_col})', 'mediumpurple')
    plot_subplot(axes[1], item_counts, f'Item Distribution ({iid_col})', 'lightgreen')

    plt.tight_layout()

    # 3. 保存图片
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"2^x分布图已保存至: {save_path}")