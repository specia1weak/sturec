import os
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from src.boleutils import ignore_future_warning
ignore_future_warning()
# 1. 生成测试数据 (包含 5 个交互，满足序列划分的最低要求)
os.makedirs('test', exist_ok=True)
with open('test/test.inter', 'w') as f:
    f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
    f.write("u1\ti1\t5.0\t1001\n")
    f.write("u1\ti2\t2.0\t1002\n")  # <- 关键的低分交互
    f.write("u1\ti3\t4.0\t1003\n")
    f.write("u1\ti4\t5.0\t1004\n")
    f.write("u1\ti5\t5.0\t1005\n")


def verify_sequence(mode):
    config_dict = {
        'data_path': './',
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'LABEL_FIELD': 'rating',

        # 关闭默认的频率过滤，防止测试数据被清空
        'user_inter_num_interval': '[0,inf)',
        'item_inter_num_interval': '[0,inf)',

        # 序列推荐必需的字段
        'ITEM_LIST_LENGTH_FIELD': 'item_length',
        'LIST_SUFFIX': '_list',
        'MAX_ITEM_LIST_LENGTH': 10,
        'train_neg_sample_args': None,
        # 数据集切分方式：Leave-one-out
        'eval_args': {'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'mode': 'full'},
    }

    if mode == 'threshold':
        config_dict['threshold'] = {'rating': 4.0}
    else:
        config_dict['val_interval'] = {'rating': '[4.0, 5.0]'}

    config = Config(model='SASRec', dataset='test', config_dict=config_dict)
    dataset = create_dataset(config)

    # 触发序列构建与切分
    train_data, valid_data, test_data = data_preparation(config, dataset)

    print(f"\n=== 测试配置: {mode} ===")

    # 获取 token 到内部 ID 的映射字典，方便我们把内部数字还原成 i1, i2 看得更清楚
    id2token = dataset.field2id_token['item_id']

    # 从训练集的 DataLoader 中抽取第一个 batch
    for batch in train_data:
        item_seq = batch['item_id_list'].tolist()
        target_item = batch['item_id'].tolist()

        print("喂给模型的实际序列:")
        for seq, target in zip(item_seq, target_item):
            # 过滤掉内部 ID 为 0 的 Padding 占位符
            seq_tokens = [id2token[i] for i in seq if i != 0]
            target_token = id2token[target]
            print(f"历史序列: {seq_tokens} -> 预测目标: {target_token}")

        break  # 抓取一个 batch 就够了


verify_sequence('threshold')
verify_sequence('val_interval')