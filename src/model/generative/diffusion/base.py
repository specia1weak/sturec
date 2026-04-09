from abc import abstractmethod
import torch
import torch.nn as nn
from src.model.generative.diffusion.schedulers import BaseScheduler, PredType
import torch.nn.functional as F

class BaseDiffusionModel(nn.Module):
    """
    所有扩散模型的终极基类。
    它负责所有的去噪循环和 CFG 逻辑，具体的模型只需要实现 _raw_predict。
    """
    # 默认预测目标，子类通过覆盖此属性来改变行为
    pred_type = PredType.X0
    def __init__(self, scheduler: BaseScheduler):
        super().__init__()
        self.scheduler = scheduler
        self.num_timesteps = scheduler.num_timesteps

    @abstractmethod
    def _raw_predict(self, x, t, y=None):
        """
        子类只需实现这个原始输出：
        如果是 DenoiseModel，返回 noise；
        如果是 PredX0Model，返回 x0。
        """
        pass

    def forward(self, x, t, y=None):
        return self._raw_predict(x, t, y)

    @torch.no_grad()
    def denoise(self, x, t_start, y=None, null_y=None, guidance_scale=1.0, steps=1):
        """
        万能去噪函数：支持有条件/无条件、CFG、以及任意预测目标。
        """
        x_0, x_t = None, x
        # 预处理空条件
        if guidance_scale > 1.0 and y is not None and null_y is None:
            null_y = torch.zeros_like(y)

        for i in range(steps):
            t_curr = t_start - i
            if t_curr < 0: break

            t_tensor = torch.full((x_t.shape[0],), t_curr, dtype=torch.long, device=x_t.device)

            # 1. 神经网络前向推断 (处理 CFG 逻辑)
            if guidance_scale > 1.0 and y is not None:
                x_t_double = torch.cat([x_t, x_t], dim=0)
                t_tensor_double = torch.cat([t_tensor, t_tensor], dim=0)
                y_double = torch.cat([y, null_y], dim=0)

                output_double = self._raw_predict(x_t_double, t_tensor_double, y_double)
                output_cond, output_uncond = output_double.chunk(2)
                model_output = output_uncond + guidance_scale * (output_cond - output_uncond)
            else:
                model_output = self._raw_predict(x_t, t_tensor, y)

            # 2. 调用调度器，传入当前类定义的 pred_type
            x_0, x_t = self.scheduler.step(model_output, t_curr, x_t, pred_type=self.pred_type)

        return x_0, x_t # x_0, x_{t-steps}

    @torch.no_grad()
    def reconstruct(self, x_T, y=None, guidance_scale=1.0):
        self.eval()
        return self.denoise(
            x_T,
            t_start=self.num_timesteps - 1,
            y=y,
            guidance_scale=guidance_scale,
            steps=self.num_timesteps
        )[0]

    def compute_loss(self, x_0: torch.Tensor, y: torch.Tensor = None, null_y: torch.Tensor = None,
                     cond_drop_prob: float = 0.1) -> torch.Tensor:
        """
        - cond_drop_prob: CFG 训练时的条件丢弃概率，通常设为 0.1 到 0.2。
        - null_y: 当丢弃条件时使用的替代值。如果未提供，默认使用全零张量。
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=x_0.device)
        x_t, true_noise = self.scheduler.add_noise(x_0, t)
        # CFG Condition Dropout 逻辑
        y_input = y
        if y is not None and cond_drop_prob > 0.0:
            if null_y is None:
                null_y = torch.zeros_like(y)

            drop_mask = torch.rand(batch_size, device=y.device) < cond_drop_prob
            reshape_args = [-1] + [1] * (y.dim() - 1)
            drop_mask = drop_mask.view(*reshape_args)
            y_input = torch.where(drop_mask, null_y, y)

        # 前向推断 (输入可能被 drop 掉的条件)
        predictions = self(x_t, t, y_input)
        target = self.scheduler.get_target(x_0, true_noise, t, self.pred_type)
        return F.mse_loss(predictions, target)

    @torch.no_grad()
    def refine(self, x_0: torch.Tensor, strength: float = 0.5, y=None, guidance_scale=1.0):
        """
        图生图 (SDEdit 范式)
        - x_0: 干净的初始图片 (比如用户的草图、或者需要被增强的低频特征图)
        - strength: 重绘幅度 [0.0, 1.0]。
                    1.0 代表完全破坏并重绘 (等价于 reconstruct)；
                    0.0 代表完全不改变原图。
        """
        self.eval()

        # 1. 约束并映射 strength 到具体的时间步 t_start
        t_start = int(strength * self.num_timesteps) - 1
        t_start = max(0, min(t_start, self.num_timesteps - 1))

        # 如果 strength 为 0，说明完全不需要重绘，直接返回原图
        if t_start < 0:
            return x_0

        # 2. 前向加噪：把干净的 x_0 "弄脏" 到 t_start 阶段
        batch_size = x_0.shape[0]
        t_tensor = torch.full((batch_size,), t_start, dtype=torch.long, device=x_0.device)
        x_t_start, _ = self.scheduler.add_noise(x_0, t_tensor)
        # 3. 反向去噪：调用你写好的万能 denoise 函数，从 t_start 开始往回走
        # 需要走的步数是 t_start + 1 (因为包含了 t=0 这一步)
        return self.denoise(
            x_t_start,
            t_start=t_start,
            y=y,
            guidance_scale=guidance_scale,
            steps=t_start + 1
        )[0]

class DenoiseModel(BaseDiffusionModel):
    pred_type = PredType.NOISE  # 覆盖属性，语义清晰
    @abstractmethod
    def _raw_predict(self, x, t, y=None):
        pass

class PredX0Model(BaseDiffusionModel):
    pred_type = PredType.X0
    @abstractmethod
    def _raw_predict(self, x, t, y=None):
        pass

class PredVModel(BaseDiffusionModel):
    pred_type = PredType.V
    @abstractmethod
    def _raw_predict(self, x, t, y=None):
        pass