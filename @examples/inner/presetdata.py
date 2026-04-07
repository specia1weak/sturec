def make_LS_dataset():
    from src.utils import change_root_workdir, ignore_future_warning
    change_root_workdir()
    ignore_future_warning()
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    # 1. 初始化配置
    # 注意：传 GRU4Rec 或 SASRec，Config 就会去加载序列模型的预处理逻辑
    cfg_override = {
        "gpu_id": "",
        "MAX_ITEM_LIST_LENGTH": 10,
        "eval_args": {
            "split": {'LS': 'valid_and_test'},  # 留一法 LS
            "order": 'TO',
            "mode": 'full'
        },
        "val_interval": {
            "rating": '[4.0, 5.0]'
        },
        # ===== 新增这一行 =====
        "train_neg_sample_args": None  # 明确告诉框架：CE 损失不需要训练负采样
    }

    config = Config(
        model='GRU4Rec',
        dataset='ml-1m',
        config_file_list=['dataset/ml-1m/m1-1m.yaml'],
        config_dict=cfg_override
    )

    # 2. 创建并加载数据集
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return train_data, valid_data, test_data



if __name__ == '__main__':
    train_data, valid_data, test_data = make_LS_dataset()
    for batch_idx, batch_data in enumerate(train_data):
        print(f"Batch {batch_idx} 的完整对象:\n", batch_data)
        print("-" * 50)
        # 核心：提取序列特征
        user_tensor = batch_data['user_id']  # [B]
        item_seq = batch_data['item_id_list']  # [B, MAX_ITEM_LIST_LENGTH]
        item_seq_len = batch_data['item_length']  # [B]
        target_item = batch_data['item_id']  # [B]

        print(f"当前 Batch Size: {len(user_tensor)}")

        print("\n--- 前 3 个样本的深入观察 ---")
        print(f"【用户 ID】 (user_id):\n{user_tensor[:3]}")

        # 最关键的一步：观察 2D 序列张量
        print(f"\n【历史序列】 (item_id_list, shape: {item_seq.shape}):")
        print(item_seq[:3])
        print("👆 提示：你可以看到右侧填充了大量的 0 (Padding)，左侧则是用户最近的历史交互物品 ID。")

        print(f"\n【真实长度】 (item_length):\n{item_seq_len[:3]}")
        print("👆 提示：它记录了上面每个序列里，非 0 元素的真实个数。写 Attention Mask 时就是拿它来遮蔽 padding 位的。")

        print(f"\n【预测目标】 (item_id):\n{target_item[:3]}")
        print("👆 提示：这就是该序列紧接着的 Next Item，也就是我们模型的预测 Ground Truth。")

        break