from pathlib import Path
from src.dataset.base import DatasetBase
import polars as pl

sparse_columns = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '216', '508', '509', '702', '853', '301', '109_14', '110_14', '127_14', '150_14']
dense_columns = ['109_14', '110_14', '127_14', '150_14', '508', '509', '702', '853']
multi_sparse_columns = ['210']

common_columns = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '109_14', '110_14', '127_14', '150_14']
skeleton_columns = ['205', '206', '207', '210', '216', '301', '508', '509', '702', '853']

common_sparse_columns = [col for col in sparse_columns if col in common_columns]
common_dense_columns = [col for col in dense_columns if col in common_columns]

skeleton_sparse_columns = [col for col in sparse_columns if col in skeleton_columns]
skeleton_dense_columns = [col for col in dense_columns if col in skeleton_columns]
skeleton_multi_sparse_columns = ['210']

rename_dict = {
    "click": "click",
    "purchase": "purchase",
    # === User Features (用户特征) ===
    "101": "user_id",
    "D109_14": "user_hist_category_count",
    "D110_14": "user_hist_shop_count",
    "D127_14": "user_hist_brand_count",
    "D150_14": "user_hist_intention_count",
    "121": "user_profile_id",           # 用户画像分类id
    "122": "user_profile_group_id",
    "124": "user_gender_id",
    "125": "user_age_id",
    "126": "user_consumption_level_1",
    "127": "user_consumption_level_2",
    "128": "user_is_working",             # 职业：是否工作
    "129": "user_geography_info",

    # === Item Features (商品/物品特征) ===
    "205": "item_id",
    "206": "item_category_id",
    "207": "item_shop_id",
    "210": "item_intention_node_id",
    "216": "item_brand_id",

    # === Combination Features (交叉/组合特征) ===
    # 命名逻辑: combo_特征A_特征B
    "D508": "combo_hist_category_cross_item_category",   # 109_14 & 206
    "D509": "combo_hist_shop_cross_item_shop",           # 110_14 & 207
    "D702": "combo_hist_brand_cross_item_brand",         # 127_14 & 216
    "D853": "combo_hist_intention_cross_item_intention", # 150_14 & 210

    # === Context Features (上下文特征) ===
    "301": "domain_id"
}

class _ALIDS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "Ali-CCP"
    TRAIN_DIR = BASE_DIR / "sample_train"
    TEST_DIR = BASE_DIR / "sample_test"

    TRAIN_USER_CTX_FEATURES = TRAIN_DIR / "common_features_train.csv"
    TRAIN_ITEM_INTERACTIONS = TRAIN_DIR / "sample_skeleton_train.csv"

    TEST_USER_CTX_FEATURES = TEST_DIR / "common_features_test.csv"
    TEST_ITEM_INTERACTIONS = TEST_DIR / "sample_skeleton_test.csv"

    TRAIN_WHOLE_INTERACTIONS = TRAIN_DIR / "train_whole_inter.parquet"
    TEST_WHOLE_INTERACTIONS = TEST_DIR / "test_whole_inter.parquet"

    @staticmethod
    def _parse_common_features(file_path) -> pl.LazyFrame:
        """
        使用 Polars LazyFrame 处理 Ali-CCP 的公共特征文件。
        全程无 Python 级别的 for 循环遍历，底层使用 Rust 多线程和 SIMD 正则匹配。
        """
        extract_exprs = []
        for col in common_sparse_columns:
            # 正则含义: 匹配开头或\x01，紧跟 field_id 和 \x02，捕获中间的特征值直到遇见 \x03
            pattern = rf"(?:^|\x01){col}\x02([^\x03]+)\x03"
            extract_exprs.append(
                pl.col("raw_features")
                .str.extract(pattern, 1)  # 1 表示提取第一个括号内的捕获组
                .fill_null("0")  # 你原代码中的 feat_dict.get(k, '0') 逻辑
                .alias(col)
            )

        for col in common_dense_columns:
            # 正则含义: 匹配 field_id 和 \x02，跳过特征值和 \x03，捕获后面的权重值直到遇见 \x01 或结尾
            pattern = rf"(?:^|\x01){col}\x02[^\x03]+\x03([^\x01]+)"
            extract_exprs.append(
                pl.col("raw_features")
                .str.extract(pattern, 1)
                .fill_null("0")
                .cast(pl.Float32)  # Dense 特征直接转换为浮点数，省去了后续的转换
                .alias(f"D{col}")  # 加上 D 前缀
            )
        lazy_df = (
            pl.scan_csv(
                file_path,
                has_header=False,
                new_columns=["common_id", "feature_num", "raw_features"]
            )
            .with_columns(extract_exprs)  # 一次性并发执行所有正则提取
            .drop(["feature_num", "raw_features"])  # 丢弃无用的原始列
        )
        return lazy_df

    @staticmethod
    def _parse_skeleton(file_path) -> pl.LazyFrame:
        """
        使用 Polars LazyFrame 解析样本骨架表，支持多值特征提取为 List
        """
        extract_exprs = []
        # 1. 提取单值稀疏特征 (Single Sparse) -> str
        for col in skeleton_sparse_columns:
            pattern = rf"(?:^|\x01){col}\x02([^\x03]+)\x03"
            extract_exprs.append(
                pl.col("raw_item_features")
                .str.extract(pattern, 1)
                .fill_null("0")  # 如果没找到该特征，填充默认值 "0"
                .alias(col)
            )
        # 2. 提取连续特征 (Dense) -> float32
        for col in skeleton_dense_columns:
            pattern = rf"(?:^|\x01){col}\x02[^\x03]+\x03([^\x01]+)"
            extract_exprs.append(
                pl.col("raw_item_features")
                .str.extract(pattern, 1)
                .fill_null("0")
                .cast(pl.Float32)  # 在内存分配前直接转换为小内存的 float32
                .alias(f"{col}")  # 加上 D 前缀
            )
        # 3. 提取多值稀疏特征 (Multi Sparse) -> List[str]
        for col in skeleton_multi_sparse_columns:
            extract_exprs.append(
                pl.col("raw_item_features")
                .str.split("\x01")  # 先炸开成数组
                .list.eval(
                    pl.element()
                    .filter(pl.element().str.starts_with(f"{col}\x02"))  # 只过滤出包含该 Field 的项
                    .str.extract(r"\x02([^\x03]+)", 1)  # 提取特征 ID
                )
                .alias(col)  # 这里的最终产物是一个真正的 List!
            )
        # 4. 构建 Lazy 计算图
        lf = (
            pl.scan_csv(
                file_path,
                has_header=False,
                new_columns=["sample_id", "click", "purchase", "common_id", "feature_num", "raw_item_features"],
                truncate_ragged_lines=True
            )
            .filter(
                ~((pl.col("click") == 0) & (pl.col("purchase") == 1))
            )
            .with_columns(extract_exprs)
            .drop(["sample_id", "feature_num", "raw_item_features"]) # 没必要的col
        )
        return lf

    @classmethod
    def _merge_table(cls, mode="train") -> pl.LazyFrame:
        """内部方法：负责 Join 宽表"""
        assert mode in ["train", "test"]
        if mode == "train":
            common_path = cls.TRAIN_USER_CTX_FEATURES
            skeleton_path = cls.TRAIN_ITEM_INTERACTIONS
        else:
            common_path = cls.TEST_USER_CTX_FEATURES
            skeleton_path = cls.TEST_ITEM_INTERACTIONS

        lf_common = cls._parse_common_features(common_path)
        lf_skeleton = cls._parse_skeleton(skeleton_path)

        # Join 并丢弃关联主键
        return lf_skeleton.join(
            lf_common, on="common_id", how="left"
        ).drop("common_id")

    @property
    def TRAIN_INTER_LF(self)-> pl.LazyFrame:
        if not self.TRAIN_WHOLE_INTERACTIONS.exists():
            self._merge_table("train").sink_parquet(self.TRAIN_WHOLE_INTERACTIONS)
        return pl.scan_parquet(self.TRAIN_WHOLE_INTERACTIONS).rename(rename_dict)
    @property
    def TEST_INTER_LF(self)-> pl.LazyFrame:
        if not self.TEST_WHOLE_INTERACTIONS.exists():
            self._merge_table("test").sink_parquet(self.TEST_WHOLE_INTERACTIONS)
        return pl.scan_parquet(self.TEST_WHOLE_INTERACTIONS).rename(rename_dict)

AliCCPDataset = _ALIDS()

# ================= 测试运行 =================
if __name__ == "__main__":
    lf_train = AliCCPDataset.TEST_INTER_LF

    print("准备打印前 5 行数据：")
    df_preview = lf_train.head(5).collect()
    print(df_preview)
    print(df_preview.columns)
    from src.utils.visualize import plot_power2_sparsity, plot_sparsity_ecdf, plot_bias_distributions
    plot_power2_sparsity(lf_train,"user_id", "item_id")
    plot_sparsity_ecdf(lf_train, "user_id", "item_id")
    plot_bias_distributions(lf_train, "user_id", "item_id", "click")