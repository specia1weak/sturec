import math

import torch
from torch import nn

from betterbole.models.generative.diffusion.base import PredVModel, PredX0Model
from betterbole.models.generative.diffusion.schedulers import PredType


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb expects even dim, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.linear1(self.act(self.norm1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        while scale.ndim < h.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        h = h * (scale + 1.0) + shift
        h = self.linear2(self.act(self.norm2(h)))
        return x + h


class RecSysEmbeddingModel(PredVModel):
    def __init__(self, scheduler, data_dim, cond_dim, hidden_dim=512, time_dim=256, num_blocks=3):
        super().__init__(scheduler)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.proj_in = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(hidden_dim, time_dim) for _ in range(num_blocks)
        ])
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def _raw_predict(self, x, t, y=None):
        cond = self.time_mlp(t)
        if y is not None:
            cond = cond + self.cond_proj(y)
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.proj_out(h)


class CDCDRMlpDiffusion(PredX0Model):
    def __init__(
            self,
            scheduler,
            data_dim: int,
            uncon_p: float = 0.1,
            hidden_dim: int = None,
            time_dim: int = None,
            num_blocks: int = 3,
            num_fields: int = None,
            objective: str = "pred_x0",
    ):
        super().__init__(scheduler)
        objective_to_pred_type = {
            "pred_x0": PredType.X0,
            "pred_v": PredType.V,
            "pred_noise": PredType.NOISE,
        }
        if objective not in objective_to_pred_type:
            raise ValueError(f"objective must be one of {tuple(objective_to_pred_type)}, got {objective}")
        self.pred_type = objective_to_pred_type[objective]
        self.objective = objective
        self.data_dim = data_dim
        self.num_fields = num_fields
        self.input_shape = (num_fields, data_dim) if num_fields is not None else (data_dim,)
        self.embedding_size = data_dim
        self.flat_embedding_size = data_dim * num_fields if num_fields is not None else data_dim
        self.uncon_p = uncon_p
        self.hidden_dim = hidden_dim or data_dim * 2
        self.time_dim = time_dim or data_dim

        self.none_embedding = nn.Parameter(torch.zeros(1, self.flat_embedding_size))
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.GELU(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(self.flat_embedding_size, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.input_proj = nn.Linear(data_dim, self.hidden_dim)
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(self.hidden_dim, self.time_dim) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, data_dim),
        )

    def _null_condition(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.none_embedding.to(device).expand(batch_size, -1)

    def _apply_cfg_dropout(self, y: torch.Tensor) -> torch.Tensor:
        y = y.view(y.size(0), -1)
        if (not self.training) or self.uncon_p <= 0.0:
            return y
        keep_mask = (torch.rand(y.size(0), 1, device=y.device) > self.uncon_p).float()
        null_y = self._null_condition(y.size(0), y.device)
        return y * keep_mask + null_y * (1.0 - keep_mask)

    def _raw_predict(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if y is None:
            y = self._null_condition(x.size(0), x.device)
        else:
            y = self._apply_cfg_dropout(y)

        cond = self.time_mlp(t) + self.cond_proj(y)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)

    def get_null_condition(self, batch_size, device):
        return self._null_condition(batch_size, device)
