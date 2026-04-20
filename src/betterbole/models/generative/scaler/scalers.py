from .base import BaseScaler, BaseStreamingScaler
import torch
"""
Z-Score 标准化
"""
class TorchStandardScaler(BaseScaler):
    """Z-Score 标准化 (均值0，方差1)"""
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True)
        self.std = torch.clamp(self.std, min=1e-8)
        self.is_fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.is_fitted, "Call fit() first!"
        return (x - self.mean) / self.std

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        assert self.is_fitted, "Call fit() first!"
        return (x_norm * self.std) + self.mean

    def to(self, device: torch.device):
        super().to(device)
        if self.is_fitted:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        return self


class StreamingStandardScaler(BaseStreamingScaler):
    def __init__(self):
        super().__init__()
        self.sum_x = None
        self.sum_x_sq = None
        self.mean = None
        self.std = None

    @torch.no_grad()
    def collect(self, batch: torch.Tensor):
        self.total_samples += batch.size(0)
        batch_sum = batch.sum(dim=0, keepdim=True)
        batch_sum_sq = (batch ** 2).sum(dim=0, keepdim=True)

        if self.sum_x is None:
            self.sum_x = batch_sum
            self.sum_x_sq = batch_sum_sq
        else:
            self.sum_x += batch_sum
            self.sum_x_sq += batch_sum_sq

    def finalize(self):
        if self.total_samples == 0:
            raise ValueError("没有收到任何数据，无法 finalize！")

        # 计算 E[X]
        self.mean = self.sum_x / self.total_samples
        # 计算 E[X^2]
        mean_of_sq = self.sum_x_sq / self.total_samples
        # 根据期望公式计算方差: Var(X) = E[X^2] - (E[X])^2
        variance = mean_of_sq - (self.mean ** 2)
        # 钳制负数（浮点精度问题），并开方得到标准差
        self.std = torch.sqrt(torch.clamp(variance, min=1e-8))
        self.is_fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.is_fitted, "必须先调用 finalize()！"
        return (x - self.mean) / self.std

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        assert self.is_fitted, "必须先调用 finalize()！"
        return (x_norm * self.std) + self.mean

    def to(self, device: torch.device):
        super().to(device)
        if self.is_fitted:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        return self