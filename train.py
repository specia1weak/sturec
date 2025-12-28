# ================= 训练函数 =================
import os

from torch import optim
from torch.utils.data import DataLoader
import torch
from src.model.fm import CrossDomainItemDataset, Config, FlowMatchingNet


def train_distribution_matching():
    Config.MODEL_SAVE_DIR.mkdir(exist_ok=True)
    dataset = CrossDomainItemDataset(Config)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    model = FlowMatchingNet(Config.EMBEDDING_SIZE, Config.EMBEDDING_SIZE, Config.HIDDEN_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    print("[-] Start Training Item Distribution Matching...")

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0

        for src_user, tgt_items in dataloader:
            # src_user: [B, D]
            # tgt_items: [B, K, D] (K = SAMPLES_PER_USER)

            src_user = src_user.to(Config.DEVICE)
            tgt_items = tgt_items.to(Config.DEVICE)

            B, K, D = tgt_items.shape

            # === Flatten 操作 ===
            # 我们把 User 的 K 个采样看作 B*K 个独立样本
            # Source 需要复制扩展
            src_flat = src_user.unsqueeze(1).repeat(1, K, 1).view(-1, D)  # [B*K, D]
            tgt_flat = tgt_items.view(-1, D)  # [B*K, D]

            # === Flow Matching 构建 ===
            # 1. 采样噪声 x0
            x0 = torch.randn_like(tgt_flat).to(Config.DEVICE)

            # 2. 采样时间 t
            t = torch.rand(B * K).to(Config.DEVICE)

            # 3. 插值
            # x_t = (1-t) * x0 + t * x1
            t_expand = t.unsqueeze(-1)
            x_t = (1 - t_expand) * x0 + t_expand * tgt_flat

            # 4. 目标速度
            v_target = tgt_flat - x0

            # === 预测 ===
            v_pred = model(x_t, t, src_flat)

            # === Loss ===
            # 这里直接用 MSE！因为 x0 是随机的，Target 是随机采样的
            # 模型自然会学习到：对于不同的 x0，应该去匹配哪个概率高的区域
            loss = torch.mean((v_pred - v_target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | Loss: {total_loss / len(dataloader):.6f}")
            torch.save(model.state_dict(), Config.MODEL_SAVE_DIR / "flow_item_dist.pth")


if __name__ == "__main__":
    train_distribution_matching()