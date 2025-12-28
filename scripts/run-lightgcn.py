import warnings
warnings.filterwarnings("ignore")   # 全局忽略所有警告
import torch
import numpy as np
import os
import pickle
from pathlib import Path
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN
from recbole.trainer import Trainer
from recbole.utils import init_seed, get_model
from src.dataset.amazon import BOLE_DATASET_BASE_DIR

def train_and_extract(domain_path, domain_name, output_dir, embedding_size=64):
    """
    在指定域上训练 LightGCN 并导出带有原始 ID 索引的 Embedding
    """
    print(f"\n[>>>] Start Pre-training on {domain_name} ...")

    # 1. 构造 RecBole 配置
    # 我们使用最纯粹的配置，不做复杂的增强，只为了拿 Embedding
    config_dict = {
        'data_path': str(domain_path.parent),  # 父目录
        'dataset': domain_path.name,  # 数据集名 (文件夹名)
        'model': 'LightGCN',
        'epochs': 20,  # 预训练不用太久，20-50 轮收敛即可
        'train_batch_size': 4096,
        'learning_rate': 0.001,
        'embedding_size': embedding_size,
        'n_layers': 2,  # LightGCN 层数，2层比较稳
        'eval_step': 1,
        'stopping_step': 5,
        'use_gpu': True,
        'show_progress': True,
        'save_dataset': False,  # 不保存 RecBole 内部 dataset
        'save_dataloaders': False,
        'log_wandb': False,

        # 关键：必须要留出验证集和测试集，否则 RecBole 会报错
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'RO'
        },
        'metrics': ['Recall', 'NDCG'],
        'topk': [20],
        'valid_metric': 'NDCG@20',  # <---【关键】显式指定使用 NDCG@20 作为验证指标
    }

    # 2. 初始化 RecBole 环境
    config = Config(model='LightGCN', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # 3. 加载数据
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 4. 初始化模型
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])

    # 5. 训练
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    print(f"[{domain_name}] Best Valid Score: {best_valid_score}")

    # 6. 提取 Embedding 和 ID 映射
    # 加载最佳模型状态
    checkpoint_path = trainer.saved_model_file
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 获取 Embedding (Num_Users x Dim)
    # LightGCN 的 embedding 是 user_embedding.weight 和 item_embedding.weight
    # 注意：LightGCN 前向传播会做图卷积，但作为静态特征，我们通常直接取 0 层的 Embedding，
    # 或者取经过卷积后的最终表示。这里建议取**最终表示 (Final Embedding)**，因为它包含了高阶邻居信息。
    with torch.no_grad():
        user_all_embeddings, item_all_embeddings = model.forward()

    user_emb_np = user_all_embeddings.detach().cpu().numpy()
    item_emb_np = item_all_embeddings.detach().cpu().numpy()

    # 7. 构建 {Raw_ID: Embedding} 字典
    # RecBole 的 token (原始ID) -> id (内部ID) 映射在 dataset.field2token_id 中
    # 但 dataset API 提供了更直接的方法：id2token

    # 处理 User
    user_id_field = config['USER_ID_FIELD']
    # dataset.id2token(field, ids) 返回原始 token
    # 注意：RecBole 内部 ID 从 1 开始，0 是 padding
    # 我们只保存有效用户的 Embedding (索引 1 到 N)
    valid_user_indices = np.arange(1, dataset.user_num)
    raw_user_ids = dataset.id2token(user_id_field, valid_user_indices)
    valid_user_embs = user_emb_np[valid_user_indices]

    user_emb_dict = dict(zip(raw_user_ids, valid_user_embs))

    # 处理 Item
    item_id_field = config['ITEM_ID_FIELD']
    valid_item_indices = np.arange(1, dataset.item_num)
    raw_item_ids = dataset.id2token(item_id_field, valid_item_indices)
    valid_item_embs = item_emb_np[valid_item_indices]

    item_emb_dict = dict(zip(raw_item_ids, valid_item_embs))

    # 8. 保存
    os.makedirs(output_dir, exist_ok=True)

    user_save_path = os.path.join(output_dir, f"{domain_name}_user_emb.pkl")
    item_save_path = os.path.join(output_dir, f"{domain_name}_item_emb.pkl")

    with open(user_save_path, 'wb') as f:
        pickle.dump(user_emb_dict, f)

    with open(item_save_path, 'wb') as f:
        pickle.dump(item_emb_dict, f)

    print(f"[Success] Saved {len(user_emb_dict)} users and {len(item_emb_dict)} items to {output_dir}")
    return user_save_path, item_save_path


def run_pretraining():
    # 数据集路径 (基于你上一步 convert_data.py 的输出)
    # 请确认这里的路径与你实际生成的文件夹一致
    PROCESSED_DIR = BOLE_DATASET_BASE_DIR / "Amazon_Processed_10Core"
    SOURCE_PATH = PROCESSED_DIR / "Source_Books"
    TARGET_PATH = PROCESSED_DIR / "Target_Movies"

    # 输出路径
    EMB_OUTPUT_DIR = PROCESSED_DIR / "Pretrained_Embeddings"

    # 1. 训练 Source Domain (Books)
    # 我们主要需要 Books 的 User Embedding
    train_and_extract(SOURCE_PATH, "Source_Books", EMB_OUTPUT_DIR)

    # 2. 训练 Target Domain (Movies)
    # 我们主要需要 Movies 的 Item Embedding (作为 Flow Matching 的目标)
    train_and_extract(TARGET_PATH, "Target_Movies", EMB_OUTPUT_DIR)

    print("\n[ALL DONE] Embeddings are ready for Flow Matching!")


if __name__ == "__main__":
    os.chdir("..")
    run_pretraining()