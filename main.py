import torch
from torch import optim

from torch.utils.data import DataLoader

from betterbole.datasets.amazon import AmazonMTLDataset, load_and_split_data
from betterbole.models.ple import PLEFramework, evaluate_model, train_one_epoch

import torch.optim as optim


def main():
    INTER_FILE = "dataset/Amazon-5core/amazon.inter"
    BATCH_SIZE = 1024
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 1. 使用时序切分加载数据
        train_ds, valid_ds, test_ds, domain_item_dict = load_and_split_data(INTER_FILE)
        
        # 将 Candidate Set 搬运到 GPU
        domain_item_dict = {k: v.to(DEVICE) for k, v in domain_item_dict.items()}

        # 2. DataLoader
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        # 验证/测试集不需要 shuffle
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 3. 初始化模型 (使用 Train 集的统计数据)
        model = PLEFramework(
            n_users=train_ds.n_users,
            n_items=train_ds.n_items
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 4. 训练循环
        print(f"Start training on {DEVICE}...")
        for epoch in range(5):
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, domain_item_dict)

            # Evaluate on Valid (用验证集调优)
            val_metrics = evaluate_model(model, valid_loader, valid_ds.n_items, DEVICE, domain_item_dict)

            print(f"-" * 40)
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.5f}")
            print(f"   [Valid] Book  AUC: {val_metrics['Book']['AUC']:.4f} | HR@10: {val_metrics['Book']['HR@10']:.4f} | NDCG@10: {val_metrics['Book']['NDCG@10']:.4f}")
            print(f"   [Valid] Movie AUC: {val_metrics['Movie']['AUC']:.4f} | HR@10: {val_metrics['Movie']['HR@10']:.4f} | NDCG@10: {val_metrics['Movie']['NDCG@10']:.4f}")

        # 5. 最后测试 (Evaluate on Test)
        print("=" * 40)
        print("Final Testing...")
        test_metrics = evaluate_model(model, test_loader, valid_ds.n_items, DEVICE, domain_item_dict)
        print(f"   [Test ] Book  AUC: {test_metrics['Book']['AUC']:.4f} | HR@10: {test_metrics['Book']['HR@10']:.4f} | NDCG@10: {test_metrics['Book']['NDCG@10']:.4f}")
        print(f"   [Test ] Movie AUC: {test_metrics['Movie']['AUC']:.4f} | HR@10: {test_metrics['Movie']['HR@10']:.4f} | NDCG@10: {test_metrics['Movie']['NDCG@10']:.4f}")

    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()