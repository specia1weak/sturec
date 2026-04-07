from abc import ABC, abstractmethod
import torch


class BaseScaler(ABC):
    def __init__(self):
        self.is_fitted = False
        self.device = torch.device('cpu')

    @abstractmethod
    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        """扫描全局数据，计算并保存统计量"""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """正向：将业务数据转换为模型友好的分布"""
        pass

    @abstractmethod
    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        """逆向：将模型生成的分布还原为业务数据"""
        pass

    def to(self, device: torch.device):
        self.device = device
        return self


from abc import ABC, abstractmethod
import torch

class BaseStreamingScaler(ABC):
    """
    流式数据标准化的抽象基类。
    专为海量数据、OOM场景、或数据需要分批到达的场景设计。
    """
    def __init__(self):
        self.is_fitted = False
        self.total_samples = 0  # 记录流过的总样本数
        self.device = torch.device('cpu')

    @abstractmethod
    @torch.no_grad()
    def collect(self, batch: torch.Tensor):
        """
        阶段一：增量收集。
        接收一个 batch 的数据，更新内部的累加器（但不计算最终统计量）。
        """
        pass

    @abstractmethod
    def finalize(self):
        """
        阶段二：结算盘点。
        所有 batch 流转完毕后调用，计算并锁定最终的均值、方差或极值。
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """阶段三：正向转换（业务数据 -> 模型分布）"""
        pass

    @abstractmethod
    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        """阶段四：逆向还原（模型分布 -> 业务数据）"""
        pass

    def to(self, device: torch.device):
        """设备迁移工具"""
        self.device = device
        return self