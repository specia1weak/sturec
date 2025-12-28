from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from torch.utils.data import Dataset
# 1. 准备配置
# 可以在这里通过 props 字典直接覆盖 yaml 里的配置
# model='DeepFM' 只是占位，读取数据阶段不强依赖特定模型，但在做 data_preparation 时可能会用到模型的特定要求
config = Config(model='DeepFM', dataset='test', config_file_list=["dataset/test.yaml"], config_dict={
    'data_path': './dataset/', # 指定数据根目录，默认就是当前目录下的 dataset
    'field_separator': ',',    # 如果你的 inter 文件是逗号分隔
    'load_col': {              # 再次确认要加载的列
        'inter': ['user_id', 'item_id', 'rating', 'timestamp', 'scene_id'] # 假设你有多场景ID
    },
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating'
})

# 2. 核心步骤：读取数据
# 这一步会自动完成：加载文件 -> ID映射(Remap) -> 过滤稀疏数据
dataset = create_dataset(config)

# 3. 验证数据是否读进去了
print(f"数据读取成功！")
print(f"用户数: {dataset.user_num}")
print(f"物品数: {dataset.item_num}")
print(f"交互总数: {len(dataset)}")

# 4. 查看具体数据 (比如看前 5 行)
print(dataset.inter_feat[:5])
# 5. 生成 DataLoader (切分训练/验证/测试集)
# 这一步会根据 config 里的 eval_args 切分数据
train_data, valid_data, test_data = data_preparation(config, dataset)

# 接下来就可以把 train_data 喂给模型了
for batch in train_data:
    print(batch)