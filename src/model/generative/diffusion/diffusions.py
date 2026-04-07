import math
import torch
from torch import nn
from src.model.generative.diffusion.base import PredVModel


class SinusoidalPosEmb(nn.Module):
    """标准的正弦波时间位置编码，将离散的 t 映射为高维连续向量"""
    def __init__(self, dim):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb 要求 dim 必须是偶数，但收到了 {dim}")
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
    # 修改 __init__，增加 cond_dim 参数和 cond_proj 层
    def __init__(self, scheduler, data_dim, cond_dim, hidden_dim=512, time_dim=256, num_blocks=3):
        super().__init__(scheduler)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # 新增：条件向量的投影层，将其映射到与时间特征相同的维度
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
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
        t_emb = self.time_mlp(t)

        # 新增：如果传入了条件 y，则将其特征加到时间特征上
        if y is not None:
            cond_emb = self.cond_proj(y)
            t_emb = t_emb + cond_emb

        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h, t_emb)
        out = self.proj_out(h)
        return out