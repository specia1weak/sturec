from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd


class KuaiRand:
    BASE_DIR = Path("D:/pyprojects/recommend-study/Datasets/KuaiRand")
    DATA_KUAI_1K = BASE_DIR / "data" / "KuaiRand-1K" / "data"
    STD_LOG_FORMER_DATA = DATA_KUAI_1K / "log_standard_4_08_to_4_21_1k.csv"
    USER_FEATURES = DATA_KUAI_1K / "user_features_1k.csv"
    VIDEO_FEATURES = DATA_KUAI_1K / "video_features_basic_1k.csv"

TARGET_COLUMN = "is_click"
LOG_FEATURES = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "duration_ms",
    "tab",
    "is_rand",
]


def load_interactions(
    log_features: Sequence = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the log, user, and video feature tables."""
    features = log_features if log_features is not None else LOG_FEATURES
    interaction_cols = list(features) + [TARGET_COLUMN]
    log_df = pd.read_csv(KuaiRand.STD_LOG_FORMER_DATA, usecols=interaction_cols)
    user_df = pd.read_csv(KuaiRand.USER_FEATURES)
    video_df = pd.read_csv(KuaiRand.VIDEO_FEATURES)
    return log_df, user_df, video_df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expand coarse time stamps into more useful model features."""
    enriched = df.copy()
    enriched["date"] = pd.to_datetime(
        enriched["date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    enriched["day_of_week"] = enriched["date"].dt.dayofweek.fillna(-1).astype("int8")
    enriched["is_weekend"] = enriched["day_of_week"].isin([5, 6]).astype("int8")
    enriched = enriched.drop(columns=["date"])

    hourmin = enriched.pop("hourmin")
    hour = (hourmin // 100).clip(lower=0, upper=23)
    minute = (hourmin % 100).clip(lower=0, upper=59)
    enriched["hour"] = hour.astype("int8")
    enriched["minute"] = minute.astype("int8")
    return enriched


def build_feature_table() -> pd.DataFrame:
    """Create the modeling table after joining all available features."""
    log_df, user_df, video_df = load_interactions()
    merged = log_df.merge(user_df, on="user_id", how="left")
    merged = merged.merge(video_df, on="video_id", how="left")
    merged = add_time_features(merged)
    merged = merged.dropna(subset=[TARGET_COLUMN])
    return merged


import os
import pandas as pd


def convert_to_bole(dataset_name='KuaiRand-1k', output_dir='dataset/KuaiRand-1k'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading raw data...")
    log_df, user_df, video_df = load_interactions()

    # ==========================================
    # 1. User 特征映射 (根据你提供的列名)
    # ==========================================
    print("Processing Users...")

    # 技巧：Raw数值(如fans_num)和分桶(fans_num_range)同时存在时，
    # 建议保存两者，但在 Config 中优先加载 range 作为 token。

    user_rename_map = {
        'user_id': 'user_id:token',

        # --- 离散/类别特征 (:token) ---
        'user_active_degree': 'user_active_degree:token',
        'is_live_streamer': 'is_live_streamer:token',
        'is_video_author': 'is_video_author:token',
        # 分桶字段 (Range) 强烈建议作为 Token
        'follow_user_num_range': 'follow_user_num_range:token',
        'fans_user_num_range': 'fans_user_num_range:token',
        'friend_user_num_range': 'friend_user_num_range:token',
        'register_days_range': 'register_days_range:token',

        # --- 连续数值特征 (:float) ---
        # 如果你想让模型理解数值的大小关系，或者做多任务回归，保留它们
        'follow_user_num': 'follow_user_num:float',
        'fans_user_num': 'fans_user_num:float',
        'friend_user_num': 'friend_user_num:float',
        'register_days': 'register_days:float',
    }

    # 处理 Onehot 特征 (feat0 - feat17) -> 全部视为 :token
    # 虽然它们是0/1，但在 RecBole 里作为 Token 嵌入 Embedding 效果通常更稳
    for i in range(18):
        col_name = f'onehot_feat{i}'
        if col_name in user_df.columns:
            user_rename_map[col_name] = f'{col_name}:token'

    # 执行重命名并保存
    # 只选择 map 中存在的列，防止报错
    valid_user_cols = [c for c in user_rename_map.keys() if c in user_df.columns]
    user_final = user_df[valid_user_cols].rename(columns=user_rename_map)
    user_final.to_csv(os.path.join(output_dir, f'{dataset_name}.user'), index=False, sep='\t')
    print(f"Saved User data with {len(valid_user_cols)} features.")

    # ==========================================
    # 2. Item (Video) 特征映射
    # ==========================================
    print("Processing Items...")

    video_rename_map = {
        'video_id': 'video_id:token',

        # --- 离散/类别特征 (:token) ---
        'video_type': 'video_type:token',
        'upload_type': 'upload_type:token',
        'visible_status': 'visible_status:token',
        'music_type': 'music_type:token',
        'music_id': 'music_id:token',  # ID 类特征一定是 Token
        'author_id': 'author_id:token',  # 作者也是一种 ID 特征

        # --- 连续数值特征 (:float) ---
        # 视频时长、分辨率宽高等原始物理属性
        'video_duration': 'video_duration:float',
        'server_width': 'server_width:float',
        'server_height': 'server_height:float',

        # --- 特殊处理 ---
        # upload_dt (日期) 建议转为时间戳 float 或者去掉，直接当 token 也可以但很多唯一值
        # 这里暂时作为 token (离散的时间点)
        'upload_dt': 'upload_dt:token',
        'tag': 'tag:token_seq'  # 如果 tag 是 "1,2,3" 这种逗号分隔字符串，可以用 :token_seq
    }

    valid_item_cols = [c for c in video_rename_map.keys() if c in video_df.columns]
    video_final = video_df[valid_item_cols].rename(columns=video_rename_map)
    video_final.to_csv(os.path.join(output_dir, f'{dataset_name}.item'), index=False, sep='\t')
    print(f"Saved Item data with {len(valid_item_cols)} features.")

    # ==========================================
    # 3. Interaction 特征映射 (同上)
    # ==========================================
    print("Processing Interactions...")
    inter_df = add_time_features(log_df)

    inter_rename_map = {
        'user_id': 'user_id:token',
        'video_id': 'video_id:token',
        'is_click': 'label:float',
        'duration_ms': 'duration_ms:float',
        'tab': 'tab:token',
        'is_rand': 'is_rand:token',
        'day_of_week': 'day_of_week:token',  # 时间上下文是典型的 Token
        'is_weekend': 'is_weekend:token',
        'hour': 'hour:token',
        # minute 通常粒度太细，可以不放，或者分桶。这里先放着
        'minute': 'minute:token'
    }

    valid_inter_cols = [c for c in inter_rename_map.keys() if c in inter_df.columns]
    inter_final = inter_df[valid_inter_cols].rename(columns=inter_rename_map)
    inter_final.to_csv(os.path.join(output_dir, f'{dataset_name}.inter'), index=False, sep='\t')
    print("Conversion Complete.")


# === 配置路径 ===
# 请确保这个路径指向你的 dataset/KuaiRand-1k 文件夹
data_dir = r'D:\pyprojects\recommend-study\studybole\dataset\KuaiRand-1k'
inter_path = os.path.join(data_dir, 'KuaiRand-1k.inter')
item_path = os.path.join(data_dir, 'KuaiRand-1k.item')


def clean_data():
    print("🚀 开始清洗数据...")

    # 1. 读取 Interaction 数据
    print(f"正在读取 {inter_path} ...")
    df_inter = pd.read_csv(inter_path, sep='\t')
    raw_inter_count = len(df_inter)
    print(f"原始交互记录数: {raw_inter_count}")

    # 2. 统计每个视频的出现次数
    # 注意：列名是 video_id:token
    video_col = 'video_id:token'
    video_counts = df_inter[video_col].value_counts()

    # === 核心参数：只保留出现次数 >= 20 的视频 ===
    # 这里的 20 可以根据你想保留多少数据调整
    valid_videos = video_counts[video_counts >= 20].index

    print(f"原始视频总数: {len(video_counts)}")
    print(f"保留视频总数 (>=20次): {len(valid_videos)}")

    # 3. 过滤 .inter 文件
    df_inter_clean = df_inter[df_inter[video_col].isin(valid_videos)]
    print(f"清洗后交互记录数: {len(df_inter_clean)}")

    # 4. 覆盖保存 .inter
    df_inter_clean.to_csv(inter_path, sep='\t', index=False)
    print("✅ .inter 文件已更新")

    # 5. 同步过滤 .item 文件 (关键步骤！)
    # 必须把 .item 文件里那些冷门视频也删掉，否则 RecBole 还是会加载它们
    if os.path.exists(item_path):
        print(f"正在读取 {item_path} ...")
        df_item = pd.read_csv(item_path, sep='\t')

        # 只保留在 valid_videos 里的视频
        df_item_clean = df_item[df_item[video_col].isin(valid_videos)]

        df_item_clean.to_csv(item_path, sep='\t', index=False)
        print(f"✅ .item 文件已更新，剩余行数: {len(df_item_clean)}")
    else:
        print("⚠️ 未找到 .item 文件，跳过。")

    print("-" * 30)
    print("🎉 数据清洗完成！")
    print("现在请重新运行 run-mmoe.py，参数量应该会降到几百万级别。")