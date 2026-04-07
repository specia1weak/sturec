from recbole.data.dataset import Dataset
from recbole.utils import FeatureSource

if __name__ == '__main__':
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
            "split": {'LS': 'valid_and_test'}, # 留一法 LS
            "order": 'TO',
            "mode": 'full'
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
    dataset: Dataset = create_dataset(config)
    print("========== 序列数据集基本信息 ==========")
    print(dataset)
    print("===================================\n")

    """
    load_col:
        inter: [user_id, item_id, rating, timestamp]
        user: [user_id, age, gender, occupation]
    """
    print(dataset.inter_feat) # 保留4列
    print(dataset.user_feat) # 保留4列
    print(dataset.item_feat) # 未指定，则为None

    """
    if self.user_feat is None:
        self._check_field('uid_field')
        return Interaction({self.uid_field: torch.arange(self.user_num)})
    else:
        return self.user_feat
    """
    print(dataset.get_item_feature()) # if None 兜底机制，保证起码有一个iid col, 实际上就是带兜底的user_feat
    print(dataset.get_user_feature())


    print(dataset.fields())
    print(dataset.fields(source=[FeatureSource.INTERACTION])) # 时间戳、label等属于Interaction 的field不会被加入被Embedding的行列
    print(dataset.field2type)
    print(dataset.field2source)
    print(dataset.field2seqlen)
    print(dataset.field2bucketnum)
    print(dataset.field2id_token) # name: 文件原始内容
    print(dataset.time_field) # 留一法必须保证这个field有效
    print(dataset.uid_field)
    print(dataset.iid_field)
    print(dataset.label_field)