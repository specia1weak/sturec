import math
from typing import Tuple
import torch
from abc import ABC, abstractmethod
from enum import Enum

import math
import numpy as np
import torch
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional

class PredType(Enum):
    X0 = "x0"
    NOISE = "noise"
    V = "velocity"

# ==========================================
# 通用数学工具 (解决复用问题)
# ==========================================
def get_beta_schedule(schedule_type: str, num_timesteps: int, beta_start=1e-4, beta_end=0.02, s=0.008) -> torch.Tensor:
    """纯函数：根据配置生成 beta 序列"""
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule_type == "other":
        warmup_steps = min(50, num_timesteps)
        warmup_betas = torch.linspace(beta_start, beta_start, warmup_steps)
        if warmup_steps == num_timesteps:
            return warmup_betas
        decay_betas = torch.linspace(beta_start, 0.05, num_timesteps - warmup_steps)
        return torch.cat((warmup_betas, decay_betas), dim=0)
    elif schedule_type == "cosine":
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps)
        f_t = torch.cos(((t / num_timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, min=0.0, max=0.999)
    else:
        raise ValueError(f"不支持的调度类型: {schedule_type}")

def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """辅助函数：从一维数组 a 中按索引 t 提取值，并 reshape 匹配 x_shape"""
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# ==========================================
# 顶层契约 (极其轻薄的接口)
# ==========================================
class BaseScheduler(ABC):
    def __init__(self, num_train_timesteps):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
    @abstractmethod
    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """配置推理步数，生成时间步序列"""
        pass

    @abstractmethod
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """前向加噪过程"""
        pass

    @abstractmethod
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor,
             pred_type: PredType = PredType.X0) -> Tuple[torch.Tensor, torch.Tensor]:
        """反向去噪过程，返回 (pred_original_sample, prev_sample)"""
        pass

    @abstractmethod
    def get_target(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor,
                   pred_type: PredType) -> torch.Tensor:
        pass

class DDPMScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps=1000, schedule_type="linear", **kwargs):
        super().__init__(num_train_timesteps)
        self.betas = get_beta_schedule(schedule_type, num_train_timesteps, **kwargs)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """DDPM 理论上必须走完全程，这里加入保护限制"""
        if num_inference_steps != self.num_train_timesteps:
            print("警告: 标准 DDPM 通常需要走完全部时间步。")
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1, device=device)

    def add_noise(self, original_samples, noise, timesteps):
        alpha_bar_t = extract_into_tensor(self.alphas_cumprod, timesteps, original_samples.shape)
        return torch.sqrt(alpha_bar_t) * original_samples + torch.sqrt(1.0 - alpha_bar_t) * noise

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

    def step(self, model_output, timestep, sample, pred_type=PredType.X0):
        t = timestep
        device = sample.device
        alpha_t = self.alphas[t].to(device)
        alpha_bar_t = self.alphas_cumprod[t].to(device)
        alpha_bar_t_minus_1 = self.alphas_cumprod[t - 1].to(device) if t > 0 else torch.tensor(1.0, device=device)
        beta_t = self.betas[t].to(device)

        pred_original_sample = self._convert_to_x0(model_output, sample, alpha_bar_t, pred_type)

        if t == 0:
            return pred_original_sample, None
        coef1 = (torch.sqrt(alpha_bar_t_minus_1) * beta_t) / (1.0 - alpha_bar_t)
        coef2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_t_minus_1)) / (1.0 - alpha_bar_t)
        mean = coef1 * pred_original_sample + coef2 * sample
        noise = torch.randn_like(sample)
        prev_sample = mean + torch.sqrt(beta_t) * noise
        return pred_original_sample, prev_sample

    def get_target(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor,
                   pred_type: PredType) -> torch.Tensor:
        """
        根据不同的预测目标，计算对应的 ground truth (用于 MSE Loss)。
        """
        if pred_type == PredType.NOISE:
            return noise
        elif pred_type == PredType.X0:
            return x_0
        elif pred_type == PredType.V:
            alpha_bar_t = extract_into_tensor(self.alphas_cumprod, timesteps, x_0.shape)
            return torch.sqrt(alpha_bar_t) * noise - torch.sqrt(1.0 - alpha_bar_t) * x_0
        else:
            raise ValueError(f"不支持的 pred_type: {pred_type}")

"""
实现机制，通过self.timesteps的设立，你传入当前时间步，在这个数组里面查表能找到下一个跨步的delta
"""
class DDIMScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps=1000, schedule_type="linear", **kwargs):
        super().__init__(num_train_timesteps)
        self.betas = get_beta_schedule(schedule_type, num_train_timesteps, **kwargs)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """核心逻辑：根据推理步数，生成跳跃的时间步数组"""
        c = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
        timesteps = np.round(c).astype(np.int64)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def add_noise(self, original_samples, noise, timesteps):
        alpha_bar_t = extract_into_tensor(self.alphas_cumprod, timesteps, original_samples.shape)
        return torch.sqrt(alpha_bar_t) * original_samples + torch.sqrt(1.0 - alpha_bar_t) * noise

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

    def step(self, model_output, timestep, sample, pred_type=PredType.X0, eta=0.0):
        """
        不再需要外部传入 prev_timestep。调度器自己知道目前处在跳步序列的哪个位置。
        """
        device = sample.device
        step_index = (self.timesteps == timestep).nonzero().item()
        prev_t = self.timesteps[step_index + 1].item() if step_index + 1 < len(self.timesteps) else -1

        alpha_bar_t = self.alphas_cumprod[timestep].to(device)
        alpha_bar_prev = self.alphas_cumprod[prev_t].to(device) if prev_t >= 0 else torch.tensor(1.0, device=device)

        pred_original_sample = self._convert_to_x0(model_output, sample, alpha_bar_t, pred_type)
        pred_epsilon = (sample - torch.sqrt(alpha_bar_t) * pred_original_sample) / torch.sqrt(1.0 - alpha_bar_t)

        # 3. 计算方差
        variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
        std_dev_t = eta * torch.sqrt(variance)

        # 4. 指向 x_{prev} 的向量计算
        pred_dir_xt = torch.sqrt(1.0 - alpha_bar_prev - std_dev_t ** 2) * pred_epsilon
        prev_sample = torch.sqrt(alpha_bar_prev) * pred_original_sample + pred_dir_xt

        # 5. 注入随机噪声 (当 eta > 0)
        if eta > 0.0 and prev_t >= 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + std_dev_t * noise

        return pred_original_sample, prev_sample

    def get_target(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor,
                   pred_type: PredType) -> torch.Tensor:
        """
        根据不同的预测目标，计算对应的 ground truth (用于 MSE Loss)。
        """
        if pred_type == PredType.NOISE:
            return noise
        elif pred_type == PredType.X0:
            return x_0
        elif pred_type == PredType.V:
            alpha_bar_t = extract_into_tensor(self.alphas_cumprod, timesteps, x_0.shape)
            return torch.sqrt(alpha_bar_t) * noise - torch.sqrt(1.0 - alpha_bar_t) * x_0
        else:
            raise ValueError(f"不支持的 pred_type: {pred_type}")
