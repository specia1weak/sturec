"""
数据集划分脚本 - 混合模式 + PLE支持
逻辑：
1. 全局切分：80% Train / 10% Val / 10% Test (按时间排序，包含正负样本)。
2. train_samples.pkl: 仅提取 Train 中的 Label=1，生成序列样本 (History + Target)。
3. train_ple.pkl (NEW): 保存 Train 中的原始数据 (Label 0/1)，用于CTR/PLE训练。
4. val.pkl / test.pkl: 保存 Val/Test 中的原始数据 (Label 0/1)。
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from itertools import combinations

class DatasetSplitter:
    def __init__(self, 
                 interaction_file,
                 output_dir='./data/processed_hybrid',
                 domain_names=['domain1', 'domain2', 'domain3'],
                 history_length=20,
                 min_interaction_len=5,
                 random_seed=42):
        
        self.interaction_file = interaction_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.domain_names = domain_names
        self.history_length = history_length
        self.min_interaction_len = min_interaction_len
        
        np.random.seed(random_seed)
        
        self.domain_to_id = {name: i for i, name in enumerate(domain_names)}
        self.id_to_domain = {i: name for i, name in enumerate(domain_names)}
        
        # 数据容器
        self.interactions = None
        self.user_interactions = defaultdict(list)
        
        # 用户统计
        self.overlapping_users = set()
        
        # === 最终输出容器 ===
        self.train_samples = []   # 序列格式 (仅 Label=1)
        self.train_ple_data = []  # 简单格式 (Label 0/1) -> NEW
        self.val_data = []        # 简单格式 (Label 0/1)
        self.test_data = []       # 简单格式 (Label 0/1)
        
        # 集合跟踪
        self.users_in_train = set()
        self.users_in_val = set()
        self.users_in_test = set()
        
    def load_data(self):
        print("Loading interaction data...")
        self.interactions = pd.read_csv(self.interaction_file)
        
        if 'label' not in self.interactions.columns:
            print("⚠️ Warning: 'label' column not found. Creating default label=1.")
            self.interactions['label'] = 1

        if self.interactions['domain_indicator'].dtype == 'object':
            self.interactions['domain_indicator'] = self.interactions['domain_indicator'].map(
                self.domain_to_id
            )
        
        # 按 User 和 Time 全局排序
        self.interactions = self.interactions.sort_values(['user', 'time']).reset_index(drop=True)
        print(f"Loaded {len(self.interactions)} interactions (Positive + Negative)")
        
    def build_user_sequences(self):
        print("\nBuilding raw user sequences...")
        for _, row in tqdm(self.interactions.iterrows(), total=len(self.interactions)):
            user_id = row['user']
            self.user_interactions[user_id].append({
                'item': row['item'],
                'domain': row['domain_indicator'],
                'time': row['time'],
                'label': int(row['label'])
            })
        print(f"Processed sequences for {len(self.user_interactions)} users")

    def get_history_sequence(self, positive_interactions, current_idx):
        history_inters = positive_interactions[:current_idx]
        history = history_inters[-self.history_length:] if len(history_inters) > 0 else []
        global_history = [(inter['item'], inter['domain']) for inter in history]
        while len(global_history) < self.history_length:
            global_history.insert(0, (0, -1))
        return global_history

    def generate_splits(self):
        print("\n" + "="*60)
        print("GENERATING HYBRID SPLITS (Train/PLE/Val/Test)")
        print("="*60)
        
        skipped_users = 0
        
        for user_id, full_seq in tqdm(self.user_interactions.items()):
            n_total = len(full_seq)
            
            # 长度过滤
            if n_total < self.min_interaction_len:
                skipped_users += 1
                continue
                
            # 计算切分点
            train_end = int(n_total * 0.8)
            val_end = int(n_total * 0.9)
            
            # 切分原始序列
            raw_train = full_seq[:train_end]
            raw_val = full_seq[train_end:val_end]
            raw_test = full_seq[val_end:]
            
            # === 1. 处理 TRAIN (Sequence & PLE) ===
            if len(raw_train) > 0:
                self.users_in_train.add(user_id)
                
                # A. 构建 train_ple_data (简单格式，含 Pos & Neg)
                for item in raw_train:
                    self.train_ple_data.append({
                        'user': user_id,
                        'item': item['item'],
                        'domain': item['domain'],
                        'label': item['label']
                    })
                
                # B. 构建 train_samples (序列格式，仅 Label=1)
                pos_train = [x for x in raw_train if x['label'] == 1]
                
                # 统计重叠 (基于正样本)
                domains_in_train = set([x['domain'] for x in pos_train])
                is_overlapping = len(domains_in_train) >= 2
                if is_overlapping:
                    self.overlapping_users.add(user_id)
                
                if len(pos_train) > 0:
                    for i in range(3,len(pos_train)):
                        target = pos_train[i]
                        history = self.get_history_sequence(pos_train, i)
                        
                        self.train_samples.append({
                            'user_id': user_id,
                            'global_history': history,
                            'target_item': target['item'],
                            'target_domain': target['domain'],
                            'is_overlapping': is_overlapping,
                            'time': target['time']
                        })
            
            # === 2. 处理 VAL ===
            if len(raw_val) > 0:
                self.users_in_val.add(user_id)
                for item in raw_val:
                    self.val_data.append({
                        'user': user_id,
                        'item': item['item'],
                        'domain': item['domain'],
                        'label': item['label']
                    })
            
            # === 3. 处理 TEST ===
            if len(raw_test) > 0:
                self.users_in_test.add(user_id)
                for item in raw_test:
                    self.test_data.append({
                        'user': user_id,
                        'item': item['item'],
                        'domain': item['domain'],
                        'label': item['label']
                    })

        print(f"Skipped {skipped_users} users")
        print(f"train_samples (Pos only, Seq): {len(self.train_samples)}")
        print(f"train_ple     (Pos + Neg):     {len(self.train_ple_data)}")
        print(f"val_data      (Pos + Neg):     {len(self.val_data)}")
        print(f"test_data     (Pos + Neg):     {len(self.test_data)}")

    def analyze_train_overlaps(self):
        """统计训练集中(正样本)的重叠情况"""
        print("\n" + "="*60)
        print("TRAIN SET OVERLAP ANALYSIS (Based on Positive Interactions)")
        print("="*60)
        
        train_user_domains = defaultdict(set)
        for sample in self.train_samples:
            train_user_domains[sample['user_id']].add(sample['target_domain'])
            
        stats = {
            'total_train_users': len(train_user_domains),
            '3_domain_overlap': 0,
            'pair_overlap': defaultdict(int)
        }
        
        domain_ids = sorted(self.domain_to_id.values())
        pairs = list(combinations(domain_ids, 2))
        
        for uid, domains in train_user_domains.items():
            if len(domains) >= 3:
                stats['3_domain_overlap'] += 1
            for d1, d2 in pairs:
                if d1 in domains and d2 in domains:
                    stats['pair_overlap'][(d1, d2)] += 1
                    
        print(f"Total Users in Train: {stats['total_train_users']}")
        print(f"3-Domain Overlap: {stats['3_domain_overlap']}")
        for k, v in stats['pair_overlap'].items():
            d1_name = self.id_to_domain[k[0]]
            d2_name = self.id_to_domain[k[1]]
            print(f"{d1_name} & {d2_name}: {v}")
            
        return stats

    def save_results(self, overlap_stats):
        print("\n" + "="*60)
        print("SAVING FILES")
        print("="*60)
        
        # 1. train_samples.pkl (Sequence List)
        with open(self.output_dir / 'train_samples.pkl', 'wb') as f:
            pickle.dump(self.train_samples, f)
        print(f"✓ Saved train_samples.pkl")
        
        # 2. train_ple.pkl (DataFrame) - NEW
        df_train_ple = pd.DataFrame(self.train_ple_data)
        if not df_train_ple.empty:
            df_train_ple = df_train_ple[['user', 'item', 'domain', 'label']]
        with open(self.output_dir / 'train_ple.pkl', 'wb') as f:
            pickle.dump(df_train_ple, f)
        print(f"✓ Saved train_ple.pkl (Shape: {df_train_ple.shape})")
        
        # 3. val.pkl (DataFrame)
        df_val = pd.DataFrame(self.val_data)
        if not df_val.empty:
            df_val = df_val[['user', 'item', 'domain', 'label']]
        with open(self.output_dir / 'val_ple.pkl', 'wb') as f:
            pickle.dump(df_val, f)
        print(f"✓ Saved val_ple.pkl (Shape: {df_val.shape})")
        
        # 4. test.pkl (DataFrame)
        df_test = pd.DataFrame(self.test_data)
        if not df_test.empty:
            df_test = df_test[['user', 'item', 'domain', 'label']]
        with open(self.output_dir / 'test_ple.pkl', 'wb') as f:
            pickle.dump(df_test, f)
        print(f"✓ Saved test_ple.pkl (Shape: {df_test.shape})")
        
        # 5. user_splits.pkl
        user_splits = {
            'train_users': list(self.users_in_train),
            'val_users': list(self.users_in_val),
            'test_users': list(self.users_in_test),
            'overlapping_users': list(self.overlapping_users),
            'split_strategy': 'Hybrid_80_10_10_WithPLE',
            'train_overlap_stats': {
                '3_domain_count': overlap_stats['3_domain_overlap'],
                'pair_counts': {f"{d1}-{d2}": c for (d1, d2), c in overlap_stats['pair_overlap'].items()}
            }
        }
        with open(self.output_dir / 'user_splits.pkl', 'wb') as f:
            pickle.dump(user_splits, f)
        print("✓ Saved user_splits.pkl")
        
        # 6. JSON Stats
        stats = {
            'counts': {
                'train_samples_seq': len(self.train_samples),
                'train_rows_ple': len(self.train_ple_data),
                'val_rows': len(self.val_data),
                'test_rows': len(self.test_data)
            },
            'train_overlap_stats': user_splits['train_overlap_stats']
        }
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print("✓ Saved dataset_stats.json")

    def run(self):
        self.load_data()
        self.build_user_sequences()
        self.generate_splits()
        overlap_stats = self.analyze_train_overlaps()
        self.save_results(overlap_stats)
        print("\n✅ Hybrid PLE Split Completed!")

# ==================== 使用示例 ====================
if __name__ == '__main__':
    splitter = DatasetSplitter(
        interaction_file='amazon_time.csv', # 确保包含 user, item, domain, time, label
        output_dir='processed_time2',
        domain_names=['0', '1', '2'],
        history_length=20,
        min_interaction_len=5
    )
    splitter.run()
    '''
Loading interaction data...
Loaded 112093 interactions (Positive + Negative)

Building raw user sequences...
100%|███████████████████████████████████████████████████████████████████████████████████████| 112093/112093 [00:03<00:00, 31386.20it/s]
Processed sequences for 11978 users

============================================================
GENERATING HYBRID SPLITS (Train/PLE/Val/Test)
============================================================
100%|█████████████████████████████████████████████████████████████████████████████████████████| 11978/11978 [00:00<00:00, 51660.61it/s]
Skipped 0 users
train_samples (Pos only, Seq): 35514
train_ple     (Pos + Neg):     85536
val_data      (Pos + Neg):     10448
test_data     (Pos + Neg):     16109

============================================================
TRAIN SET OVERLAP ANALYSIS (Based on Positive Interactions)
============================================================
Total Users in Train: 8493
3-Domain Overlap: 3
0 & 2: 9
0 & 1: 83
1 & 2: 22

============================================================
SAVING FILES
============================================================
✓ Saved train_samples.pkl
✓ Saved train_ple.pkl (Shape: (85536, 4))
✓ Saved val_ple.pkl (Shape: (10448, 4))
✓ Saved test_ple.pkl (Shape: (16109, 4))
✓ Saved user_splits.pkl
✓ Saved dataset_stats.json

✅ Hybrid PLE Split Completed!
    '''