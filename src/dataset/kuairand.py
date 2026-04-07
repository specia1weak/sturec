from pathlib import Path

from src.betterbole.data.dataset import ParquetStreamDataset
from src.betterbole.emb.schema import SchemaManager, SparseEmbSetting, QuantileEmbSetting, \
    SparseSetEmbSetting, IdSeqEmbSetting
import polars as pl
from src.betterbole.enum_type import FeatureSource
from src.betterbole.plutils import extract_history_sequences
from src.dataset.base import DatasetBase
import pprint

class KuaiRandDataset(DatasetBase):
    BASE_DIR = Path("D:/pyprojects/recommend-study/Datasets/KuaiRand")
    DATA_KUAI_1K = BASE_DIR / "data" / "KuaiRand-1K" / "data"
    STD_LOG_FORMER_DATA = DATA_KUAI_1K / "log_standard_4_08_to_4_21_1k.csv"
    STD_LOG_FORMER_DATA_P1 = STD_LOG_FORMER_DATA
    STD_LOG_FORMER_DATA_P2 = DATA_KUAI_1K / "log_standard_4_22_to_5_08_1k.csv"
    RAND_LOG_FORMER_DATA = DATA_KUAI_1K / "log_random_4_22_to_5_08_1k.csv"
    USER_FEATURES = DATA_KUAI_1K / "user_features_1k.csv"
    VIDEO_FEATURES = DATA_KUAI_1K / "video_features_basic_1k.csv"

# ==========================================
# 执行与测试示例
# ==========================================
if __name__ == "__main__":
    # 为了防止路径不存在报错，这里用 try-except 包裹文件读取逻辑
    # 请确保你的实际路径中存在这些 CSV 文件
    from src.dataset.overview import get_general_info, get_head_info, get_group_stats
    try:
        # 读取三个文件 (这里假设存在且格式标准)
        df_log = pl.read_csv(KuaiRandDataset.STD_LOG_FORMER_DATA_P2)
        df_user = pl.read_csv(KuaiRandDataset.USER_FEATURES)
        df_video = pl.read_csv(KuaiRandDataset.VIDEO_FEATURES)

        datasets = {
            "Log Standard": df_log,
            "User Features": df_user,
            "Video Features": df_video
        }

        # 1. 测试通用信息打印
        print("=== 一、通用信息统计 ===")
        for name, df in datasets.items():
            info = get_general_info(df)
            print(f"\n[{name}] 数据集统计:")
            # 使用 pprint 打印字典，结构清晰且不会省略数据
            pprint.pprint(info, depth=4, compact=False)

        # 2. 测试分组统计函数 (以 df_log 中的某个假设列为例，例如 'is_click')
        print("\n=== 二、分组统计 ===")
        # 这里假设 df_log 有一个 'is_click' 列，你可以替换为你实际想看的列名
        test_col = "tab"
        group_stats = get_group_stats(df_log, test_col)
        print(f"对 '{test_col}' 的分组统计结果:")
        pprint.pprint(group_stats)

        # 3. 测试 Head(3) 视图
        print("\n=== 三、Head(3) 完整数据及类型视图 ===")
        head_view = get_head_info(df_user)  # 以 User Features 为例
        print("[User Features] 前三行完整数据:")
        pprint.pprint(head_view)

    except FileNotFoundError as e:
        print(f"文件未找到，请检查路径: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

