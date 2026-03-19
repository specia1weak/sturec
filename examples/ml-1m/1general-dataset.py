if __name__ == '__main__':
    from src.boleutils import change_root_workdir, ignore_future_warning
    change_root_workdir()
    ignore_future_warning()

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    # 1. 初始化配置
    # 注意：这里我们随便填一个模型名字（比如 'BPR'），
    # 哪怕不训练，Config 也需要根据模型类型来决定如何构造 DataLoader。
    cfg_override = {
        "gpu_id": "",
    }
    config = Config(model='BPR', dataset='ml-1m', config_file_list=['dataset/ml-1m/m1-1m.yaml'], config_dict=cfg_override)
    # 2. 创建并加载数据集
    # 这一步会读取原子文件，进行 ID 映射 (Remapping)，并打印数据集的统计信息
    dataset = create_dataset(config)
    print("========== 数据集基本信息 ==========")
    print(dataset)
    print("===================================\n")
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print("========== 偷窥第一个 Batch 的数据 ==========")
    for batch_idx, batch_data in enumerate(train_data):
        print(f"Batch {batch_idx} 的完整对象:\n", batch_data)
        print("-" * 40)

        # 你可以像使用字典一样，通过字段名获取具体的 Tensor
        user_tensor = batch_data['user_id']
        item_tensor = batch_data['item_id']
        print(f"当前 Batch 的大小 (Batch Size): {len(user_tensor)}")
        print(f"前 5 个用户的内部 ID: {user_tensor[:5]}")
        print(f"前 5 个物品的内部 ID: {item_tensor[:5]}")
        # 如果你用的是 BPR 这种需要负采样的模型，你会发现 RecBole 自动帮你生成了负样本
        if 'neg_item_id' in batch_data:
            print(f"前 5 个被采样的负物品 ID: {batch_data['neg_item_id'][:5]}")

        # 看完第一个 Batch 就直接退出循环，不看整个 epoch 了
        break


    """
    小总结：你可以在dataset/ml-1m/m1-1m.yaml查看我们配置的详细信息
    但是我们回到python代码里面，你会发现batch_data更像是最简单的dict类型，k就是col名，v就是值，只不过一个batch会让v的值有个B维度。
    batch_data并不会变成Embedding，但是他确确实实帮你做了一些映射。例如他会给token类型的col，维护一个从0-n-1的编码，float类型就会留下。label类型也会帮你算好条件。
    """