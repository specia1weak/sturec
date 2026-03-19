import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class AmazonMTLDataset(Dataset):
    def __init__(self, df):
        """
        :param df: 已经包含 user_idx, item_idx, domain_idx, label 列的 DataFrame
        """
        self.df = df.reset_index(drop=True)

        # 直接读取已经编码好的列 (无需 LabelEncoder)
        # 注意：我们在外部已经做好了 +1 (padding) 的处理
        self.user_tensor = torch.tensor(self.df['user_idx'].values, dtype=torch.long)
        self.item_tensor = torch.tensor(self.df['item_idx'].values, dtype=torch.long)
        self.domain_tensor = torch.tensor(self.df['domain_idx'].values, dtype=torch.long)
        self.label_tensor = torch.tensor(self.df['label'].values, dtype=torch.float32)

        # 统计维度 (只需统计最大ID即可)
        self.n_users = self.df['user_idx'].max() + 1
        self.n_items = self.df['item_idx'].max() + 1
        self.n_domains = self.df['domain_idx'].max() + 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'user': self.user_tensor[idx],
            'item': self.item_tensor[idx],
            'domain': self.domain_tensor[idx],
            'label': self.label_tensor[idx]
        }


def load_and_split_data(inter_path):
    print(f"Loading raw data from {inter_path}...")
    df = pd.read_csv(inter_path, sep='\t')

    # --- Step 1: 全局编码 (Global Encoding) ---
    print("Encoding IDs globally...")

    # User ID
    user_le = LabelEncoder()
    # fit_transform 全量数据
    df['user_idx'] = user_le.fit_transform(df['user_id:token']) + 1

    # Item ID
    item_le = LabelEncoder()
    df['item_idx'] = item_le.fit_transform(df['item_id:token']) + 1

    # Domain
    domain_le = LabelEncoder()
    df['domain_idx'] = domain_le.fit_transform(df['domain:token'])

    # Label
    df['label'] = df['label:float']

    # 打印全局统计
    print(f"Global Stats: Users={len(user_le.classes_)}, Items={len(item_le.classes_)}")

    # --- Step 2: 时序切分 (Leave-One-Out) ---
    print("Sorting by Timestamp...")
    # 按用户分组，按时间排序
    df = df.sort_values(by=['user_idx', 'timestamp:float'])

    print("Splitting (Leave-One-Out)...")
    # 生成倒序排名: 1=Test, 2=Valid, >2=Train
    df['time_rank'] = df.groupby('user_idx')['timestamp:float'] \
        .rank(method='first', ascending=False)

    # 切分 DataFrame
    test_df = df[df['time_rank'] == 1].copy()
    valid_df = df[df['time_rank'] == 2].copy()
    train_df = df[df['time_rank'] > 2].copy()

    print(f"Split stats: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

    # --- Step 3: 构建 Dataset ---
    # 现在 Dataset 只需要“傻瓜式”地把 DataFrame 包装起来
    train_ds = AmazonMTLDataset(train_df)
    valid_ds = AmazonMTLDataset(valid_df)
    test_ds = AmazonMTLDataset(test_df)



    # 为了让外部能获取 n_users / n_items，我们需要手动修正 train_ds 的统计信息
    # 因为 train_df 可能缺少某些 ID，直接取 max() 可能会小
    # 我们应该用全局的 max() 覆盖它
    global_n_users = len(user_le.classes_) + 1
    global_n_items = len(item_le.classes_) + 1

    for ds in [train_ds, valid_ds, test_ds]:
        ds.n_users = global_n_users
        ds.n_items = global_n_items

    # 构建 Domain -> Item 映射 (用于域内负采样)
    # domain_idx -> unique item_idx array
    print("Building Domain-Item Map...")
    domain_item_map = df.groupby('domain_idx')['item_idx'].unique().to_dict()
    # Convert to Tensor for efficiency
    domain_item_dict = {
        k: torch.tensor(v, dtype=torch.long)
        for k, v in domain_item_map.items()
    }

    return train_ds, valid_ds, test_ds, domain_item_dict