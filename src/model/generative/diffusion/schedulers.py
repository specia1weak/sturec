import math
from typing import Tuple
import torch
from abc import ABC, abstractmethod
from enum import Enum

class PredType(Enum):
    X0 = "x0"
    NOISE = "noise"
    V = "velocity"

# ==========================================
# 顶层：纯粹的行为契约 (不包含任何物理常数)
# ==========================================
class BaseScheduler(ABC):
    def __init__(self, num_timesteps: int):
        self.num_timesteps = num_timesteps
        self.timesteps = torch.arange(num_timesteps - 1, -1, -1)

    @abstractmethod
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_target(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, pred_type: PredType) -> torch.Tensor:
        pass

    @abstractmethod
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor,
             pred_type: PredType=PredType.X0) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

# ==========================================
# 中间层：DDPM 物理引擎 (包揽所有数学公式)
# ==========================================
class BaseDDPMScheduler(BaseScheduler):
    def __init__(self, num_timesteps: int, betas: torch.Tensor):
        super().__init__(num_timesteps)
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def _convert_to_x0(self, model_output: torch.Tensor, sample: torch.Tensor,
                       alpha_bar_t: torch.Tensor, pred_type: PredType) -> torch.Tensor:
        if pred_type == PredType.X0:
            return model_output
        elif pred_type == PredType.NOISE:
            return (sample - torch.sqrt(1.0 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
        elif pred_type == PredType.V:
            return torch.sqrt(alpha_bar_t) * sample - torch.sqrt(1.0 - alpha_bar_t) * model_output
        else:
            raise ValueError(f"未知的 prediction_type: {pred_type}")

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x_0)
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_0.shape)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return x_t, noise

    def get_target(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, pred_type: PredType) -> torch.Tensor:
        if pred_type == PredType.NOISE:
            return noise
        elif pred_type == PredType.X0:
            return x_0
        elif pred_type == PredType.V:
            alpha_bar_t = self._extract(self.alphas_cumprod, t, x_0.shape)
            return torch.sqrt(alpha_bar_t) * noise - torch.sqrt(1.0 - alpha_bar_t) * x_0
        else:
            raise ValueError(f"未知的 pred_type: {pred_type}")

    def step(self, model_output, timestep, sample, pred_type: PredType=PredType.X0, clip_range=4.0):
        t = timestep
        device = sample.device
        alpha_t = self.alphas[t].to(device)
        alpha_bar_t = self.alphas_cumprod[t].to(device)
        alpha_bar_t_minus_1 = self.alphas_cumprod[t - 1].to(device) if t > 0 else torch.tensor(1.0, device=device)
        beta_t = self.betas[t].to(device)

        pred_original_sample = self._convert_to_x0(model_output, sample, alpha_bar_t, pred_type)
        if clip_range is not None:
            pred_original_sample = torch.clamp(pred_original_sample, -clip_range, clip_range)
        if t == 0:
            return pred_original_sample, None

        coef1 = (torch.sqrt(alpha_bar_t_minus_1) * beta_t) / (1.0 - alpha_bar_t)
        coef2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_t_minus_1)) / (1.0 - alpha_bar_t)
        mean = coef1 * pred_original_sample + coef2 * sample

        noise = torch.randn_like(sample)
        prev_sample = mean + torch.sqrt(beta_t) * noise

        return pred_original_sample, prev_sample

# ==========================================
# 底层：策略实现 (只需提供 Beta 序列)
# ==========================================
class LinearDDPMScheduler(BaseDDPMScheduler):
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        super().__init__(num_timesteps, betas)

class CosineDDPMScheduler(BaseDDPMScheduler):
    def __init__(self, num_timesteps=1000, s=0.008, max_beta=0.999):
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps)
        f_t = torch.cos(((t / num_timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, min=0.0, max=max_beta)
        super().__init__(num_timesteps, betas)