import pandas as pd
import os
from src.dataset.kuairand import build_feature_table

# 1. 获取数据
df = build_feature_table()

# --- 数据预处理建议 ---
# MMoE 这种深度模型对数值特征归一化比较敏感，建议对 duration 等数值进行归一化或分桶
# 这里为了演示流程，暂时保持原样，但在 Config 中定义为 float

# 2. 定义保存路径
dataset_name = 'kuairand_mmoe'
data_path = f'dataset/{dataset_name}'
if not os.path.exists(data_path):
    os.makedirs(data_path)

# 3. 拆分并重命名列以符合 RecBole 规范 (ColumnName:Type)
# RecBole 核心字段：user_id:token, item_id:token, label:float/token

# === 生成 .inter 文件 (交互数据 + 上下文特征) ===
# 选取交互相关的列
inter_cols = ['user_id', 'video_id', 'is_click', 'duration_ms', 'day_of_week', 'is_weekend', 'hour']
inter_df = df[inter_cols].copy()

# 重命名列头，指定类型
inter_rename_map = {
    'user_id': 'user_id:token',
    'video_id': 'video_id:token',
    'is_click': 'label:float',       # MMoE 做 CTR 预估时，标签通常设为 float
    'duration_ms': 'duration_ms:float',
    'day_of_week': 'day_of_week:token',
    'is_weekend': 'is_weekend:token',
    'hour': 'hour:token'
}
inter_df.rename(columns=inter_rename_map, inplace=True)
inter_df.to_csv(f'{data_path}/{dataset_name}.inter', index=False, sep='\t')
print(f"Generated {dataset_name}.inter")

# === 生成 .user 文件 (用户特征) ===
# 注意：RecBole 要求 user 文件中 user_id 必须唯一
user_cols = ['user_id', 'user_active_degree', 'is_live_streamer', 'is_video_author',
             'follow_user_num_range', 'fans_user_num_range', 'register_days_range']
             # 添加你的 onehot_feat... 如果需要
user_df = df[user_cols].drop_duplicates('user_id').copy()

user_rename_map = {
    'user_id': 'user_id:token',
    'user_active_degree': 'user_active_degree:token',
    'is_live_streamer': 'is_live_streamer:token',
    'is_video_author': 'is_video_author:token',
    'follow_user_num_range': 'follow_user_num_range:token',
    'fans_user_num_range': 'fans_user_num_range:token',
    'register_days_range': 'register_days_range:token'
}
user_df.rename(columns=user_rename_map, inplace=True)
user_df.to_csv(f'{data_path}/{dataset_name}.user', index=False, sep='\t')
print(f"Generated {dataset_name}.user")

# === 生成 .item 文件 (视频特征) ===
item_cols = ['video_id', 'video_type', 'upload_type', 'visible_status', 'music_type']
item_df = df[item_cols].drop_duplicates('video_id').copy()

item_rename_map = {
    'video_id': 'video_id:token',
    'video_type': 'video_type:token',
    'upload_type': 'upload_type:token',
    'visible_status': 'visible_status:token',
    'music_type': 'music_type:token'
}
item_df.rename(columns=item_rename_map, inplace=True)
item_df.to_csv(f'{data_path}/{dataset_name}.item', index=False, sep='\t')
print(f"Generated {dataset_name}.item")