from src.model.generative.diffusion.base import PredX0Model, PredVModel
from src.model.generative.diffusion.schedulers import BaseScheduler, LinearDDPMScheduler
import torch.nn as nn
import torch

import math


class SinusoidalPosEmb(nn.Module):
    """标准的正弦波时间位置编码，将离散的 t 映射为高维连续向量"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # t[:, None] 变成 (B, 1)，embeddings[None, :] 变成 (1, half_dim)
        embeddings = t.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock1D(nn.Module):
    """带 FiLM (Feature-wise Linear Modulation) 注入的 1D 残差块"""
    def __init__(self, hidden_dim, time_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.SiLU(),  # 放弃 ReLU，使用 SiLU 防止扩散后期死区
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 2)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, t_emb):
        # 1. 预处理数据
        h = self.mlp(self.norm1(x))
        time_hidden = self.time_mlp(t_emb)
        scale, shift = time_hidden.chunk(2, dim=-1)
        h = h * (scale + 1.0) + shift
        h = self.proj_out(self.norm2(h))
        return x + h


class RecSysEmbeddingModel(PredVModel):
    def __init__(self, scheduler, data_dim=64, hidden_dim=512, time_dim=256, num_blocks=3):
        super().__init__(scheduler)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        self.proj_in = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, time_dim) for _ in range(num_blocks)
        ])
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def _raw_predict(self, x, t, y=None):
        """
        核心推断逻辑：实现父类要求的抽象方法
        """
        # x shape: (B, data_dim)
        # t shape: (B,)
        t_emb = self.time_mlp(t)
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h, t_emb)
        out = self.proj_out(h)
        return out


if __name__ == '__main__':
    def get_toy_2d_data(batch_size):
        cluster_idx = torch.randint(0, 2, (batch_size,))
        noise = torch.randn(batch_size, 2)
        centers = torch.tensor([[5.0, 5.0], [-5.0, -5.0]])
        x_0 = noise + centers[cluster_idx]
        return x_0


    # 1. 初始化组件 (完美解耦)
    scheduler = LinearDDPMScheduler(num_timesteps=1000)
    model = RecSysEmbeddingModel(scheduler, data_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 128
    epochs = 1000

    print("开始训练...")
    model.train()

    for epoch in range(epochs):
        x_0 = get_toy_2d_data(batch_size)
        loss = model.compute_loss(x_0)
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # 测试推理代码保持不变
    x_T = torch.randn(10, 2)  # 注意：生成起始往往是标准正态噪声 randn，而不是 zeros
    x_0 = model.reconstruct(x_T)
    print(x_0)