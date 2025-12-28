"""
将 Amazon 数据集转换为 RecBole 格式
- 输入: raw/amazon_5_core/amazon_time.csv
- 输出: dataset/amazon/amazon.inter, amazon.user, amazon.item, amazon.yaml
"""
def convert_amazon_to_recbole():
    import pandas as pd
    import os
    from collections import Counter

    # 读取原始数据
    print("正在读取 Amazon 数据...")
    df = pd.read_csv('raw/amazon_5_core/amazon_time.csv')
    print(f"数据行数: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    print("\n前5行数据:")
    print(df.head())

    # 创建输出目录
    output_dir = 'dataset/amazon'
    os.makedirs(output_dir, exist_ok=True)

    # ===== 1. 创建 .inter 文件 (交互数据) =====
    print("\n创建 .inter 文件...")
    inter_df = df[['user', 'item', 'label', 'time', 'domain_indicator']].copy()
    inter_df.columns = ['user_id:token', 'item_id:token', 'label:float', 'timestamp:float', 'domain:token']

    # 保存为 tab 分隔的文件
    inter_path = os.path.join(output_dir, 'amazon.inter')
    inter_df.to_csv(inter_path, sep='\t', index=False)
    print(f"[OK] 保存到 {inter_path}")
    print(f"  - 交互数量: {len(inter_df)}")
    print(f"  - 用户数量: {inter_df['user_id:token'].nunique()}")
    print(f"  - 物品数量: {inter_df['item_id:token'].nunique()}")

    # ===== 2. 创建 .user 文件 (用户特征) =====
    print("\n创建 .user 文件...")
    # 使用 groupby 优化性能
    user_stats = df.groupby('user').agg({
        'domain_indicator': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # 主要domain
        'label': ['count', 'mean']  # 交互次数和平均评分
    }).reset_index()

    user_stats.columns = ['user_id:token', 'main_domain:token', 'inter_count:float', 'avg_rating:float']
    user_df = user_stats
    user_path = os.path.join(output_dir, 'amazon.user')
    user_df.to_csv(user_path, sep='\t', index=False)
    print(f"[OK] 保存到 {user_path}")
    print(f"  - 用户数量: {len(user_df)}")

    # ===== 3. 创建 .item 文件 (物品特征) =====
    print("\n创建 .item 文件...")
    # 使用 groupby 优化性能
    item_stats = df.groupby('item').agg({
        'domain_indicator': 'first',  # 物品的domain（应该是唯一的）
        'label': ['count', 'mean', 'sum']  # 交互次数、平均评分、正向评价总数
    }).reset_index()

    item_stats.columns = ['item_id:token', 'domain:token', 'inter_count:float', 'avg_rating:float', 'positive_sum']
    item_stats['positive_rate:float'] = item_stats['positive_sum'] / item_stats['inter_count:float']
    item_df = item_stats[['item_id:token', 'domain:token', 'inter_count:float', 'avg_rating:float', 'positive_rate:float']]
    item_path = os.path.join(output_dir, 'amazon.item')
    item_df.to_csv(item_path, sep='\t', index=False)
    print(f"[OK] 保存到 {item_path}")
    print(f"  - 物品数量: {len(item_df)}")

    # ===== 4. 创建 .yaml 配置文件 =====
    print("\n创建 .yaml 配置文件...")
    yaml_content = """# =========================================================
    #  Amazon Dataset Configuration for RecBole
    # =========================================================

    # 1. 基础路径配置
    # ---------------------------------------------------------
    data_path: dataset/
    dataset: amazon

    # 2. 文件格式配置
    # ---------------------------------------------------------
    field_separator: "\\t"
    seq_separator: " "

    # 3. 核心字段映射
    # ---------------------------------------------------------
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    LABEL_FIELD: label
    TIME_FIELD: timestamp

    # 4. 加载列配置
    # ---------------------------------------------------------
    load_col:
        inter:
            - user_id
            - item_id
            - label
            - timestamp
            - domain
        user:
            - user_id
            - main_domain
            - inter_count
            - avg_rating
        item:
            - item_id
            - domain
            - inter_count
            - avg_rating
            - positive_rate

    # 5. 特征预处理
    # ---------------------------------------------------------
    # 归一化数值型特征
    normalize_field:
        - inter_count
        - avg_rating
        - positive_rate
        - timestamp

    # 6. 过滤与划分设置
    # ---------------------------------------------------------
    # 过滤掉交互少于 5 次的用户/物品
    min_user_inter_num: 5
    min_item_inter_num: 5

    # 数据集划分方式
    eval_args:
        split: {'RS': [0.8, 0.1, 0.1]}  # 训练:验证:测试 = 8:1:1
        group_by: user  # 按用户分组（保证用户不会同时出现在训练和测试集）
        order: TO  # Time Ordering - 按时间排序
        mode: labeled  # 标签模式（适用于CTR预测）

    # 7. 模型参数（以 DeepFM 为例）
    # ---------------------------------------------------------
    embedding_size: 16
    mlp_hidden_size: [64, 64, 64]
    dropout_prob: 0.2

    # 8. 训练参数
    # ---------------------------------------------------------
    epochs: 50
    train_batch_size: 2048
    learner: adam
    learning_rate: 0.001
    eval_step: 1
    stopping_step: 10
    """

    yaml_path = os.path.join(output_dir, 'amazon.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"[OK] 保存到 {yaml_path}")

    # ===== 5. 数据统计信息 =====
    print("\n" + "="*60)
    print("数据集统计信息:")
    print("="*60)
    print(f"总交互数: {len(inter_df):,}")
    print(f"用户数: {len(user_df):,}")
    print(f"物品数: {len(item_df):,}")
    print(f"稀疏度: {len(inter_df) / (len(user_df) * len(item_df)) * 100:.4f}%")
    print(f"\nDomain 分布:")
    domain_names = {0: 'Beauty', 1: 'Clothing', 2: 'Health'}
    for domain, count in sorted(df['domain_indicator'].value_counts().items()):
        print(f"  {domain_names[domain]} (domain {domain}): {count:,} ({count/len(df)*100:.2f}%)")
    print(f"\nLabel 分布:")
    for label, count in sorted(df['label'].value_counts().items()):
        print(f"  Label {label}: {count:,} ({count/len(df)*100:.2f}%)")

    print("\n[OK] 转换完成！现在可以使用 RecBole 加载这个数据集了。")
    print(f"\n使用示例:")
    print(f"  from recbole.quick_start import run_recbole")
    print(f"  run_recbole(model='DeepFM', dataset='amazon', config_file_list=['dataset/amazon/amazon.yaml'])")

