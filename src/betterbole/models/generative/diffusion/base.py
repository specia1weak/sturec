from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from betterbole.models.generative.diffusion.schedulers import BaseScheduler, PredType

class BaseDiffusionModel(nn.Module):
    pred_type = PredType.X0

    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        # 仅在训练时使用，推理时应依赖 scheduler.timesteps
        self.num_train_timesteps = scheduler.num_train_timesteps

    @abstractmethod
    def _raw_predict(self, x, t, y=None):
        pass

    def forward(self, x, t, y=None):
        return self._raw_predict(x, t, y)

    def calculate_loss(self, x_0: torch.Tensor, y: torch.Tensor = None, loss_fn=F.mse_loss):
        """
        通用的扩散模型训练损失计算引擎。
        自动完成：随机时间步采样 -> 加噪 -> 模型预测 -> 获取 Target -> 计算 Loss

        Args:
            x_0: [B, ...] 原始干净数据 (如图像或特征向量)
            y: [B, ...] 条件特征 (可选)
            loss_fn: 损失函数，默认为 F.mse_loss

        Returns:
            torch.Tensor: 标量 Loss
        """
        self.train()
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_noisy = self.scheduler.add_noise(x_0, noise, t)
        model_output = self(x_noisy, t, y=y)
        target = self.scheduler.get_target(x_0, noise, t, self.pred_type)
        return loss_fn(model_output, target)


    @torch.no_grad()
    def denoise(self, x, y=None, null_y=None, guidance_scale=1.0,
                num_inference_steps=None, start_step=0, end_idx=None):
        """
        灵活的去噪引擎：支持指定推理步数区间。
        - start_step: 去噪循环的起始步索引 (0 代表从最强噪声处开始)
        - end_step: 去噪循环的结束步索引 (None 代表跑到结束)
        """
        # 1. 设定总体的推演步数 (交由 Scheduler 构建完整时间轴)
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps, device=x.device)

        # 2. 获取完整的时间步数组
        timesteps = self.scheduler.timesteps
        if end_idx is None:
            end_idx = len(timesteps)

        active_timesteps = timesteps[start_step:end_idx]
        x_0_pred, x_t = None, x

        if guidance_scale > 1.0 and y is not None and null_y is None:
            null_y = torch.zeros_like(y)

        # 4. 仅遍历切片后的 active_timesteps
        for t_curr in active_timesteps:
            t_tensor = torch.full((x_t.shape[0],), t_curr.item(), dtype=torch.long, device=x_t.device)

            if guidance_scale > 1.0 and y is not None:
                x_t_double = torch.cat([x_t, x_t], dim=0)
                t_tensor_double = torch.cat([t_tensor, t_tensor], dim=0)
                y_double = torch.cat([y, null_y], dim=0)

                output_double = self._raw_predict(x_t_double, t_tensor_double, y_double)
                output_cond, output_uncond = output_double.chunk(2)
                model_output = output_uncond + guidance_scale * (output_cond - output_uncond)
            else:
                model_output = self._raw_predict(x_t, t_tensor, y)

            x_0_pred, x_t = self.scheduler.step(model_output, t_curr.item(), x_t, pred_type=self.pred_type)

        return x_0_pred, x_t

    @torch.no_grad()
    def reconstruct(self, x_T, y=None, guidance_scale=1.0, num_inference_steps=None):
        self.eval()
        return self.denoise(
            x_T,
            y=y,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            t_start_ratio=1.0
        )[0]

    @torch.no_grad()
    def refine(self, x_0: torch.Tensor, strength: float = 0.5, y=None, guidance_scale=1.0, num_inference_steps=None):
        self.eval()
        # 处理不需要重绘的情况
        if strength <= 0.0:
            return x_0
        # 1. 确保 scheduler 的步数配置正确
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps, device=x_0.device)
        timesteps = self.scheduler.timesteps

        # 2. 根据 strength 找到对应的起始时间步索引
        start_idx = int((1.0 - strength) * len(timesteps))
        start_idx = max(0, min(start_idx, len(timesteps) - 1))

        # 获取具体的 timestep 整数值，用于前向加噪
        t_start_val = timesteps[start_idx]

        # 3. 修复原版的 Bug：补齐 noise 参数，且 add_noise 只有一个返回值
        batch_size = x_0.shape[0]
        t_tensor = torch.full((batch_size,), t_start_val.item(), dtype=torch.long, device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t_start = self.scheduler.add_noise(x_0, noise, t_tensor)

        # 4. 调用 denoise，传入 t_start_ratio 让去噪循环知道从哪里开始
        return self.denoise(
            x_t_start,
            y=y,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            t_start_ratio=strength
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