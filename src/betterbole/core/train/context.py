from dataclasses import dataclass
from typing import Any, Dict, Optional, Iterable
import torch

from betterbole.core.interaction import Interaction
from betterbole.evaluate.manager import EvaluatorManager
from betterbole.utils.recorder import ExplicitFeatureRecorder
from betterbole.utils.time import CudaNamedTimer


@dataclass
class TrainContext:
    # 1. 训练进度统计（通常是只读的）
    epoch: int
    global_step: int
    batch_idx: int

    # 2. 核心操纵杆
    optimizer: torch.optim.Optimizer

    # 3. 实验与全局组件
    manager: Any  # 你的 SchemaManager
    cfg: Any  # 全局配置参数
    timer: Any  # 允许模型给特殊的内部步骤计时
    recorder: Any = None
    # 4. 扩展字典（兜底方案，防止后续还要加奇怪的东西）
    kwargs: Dict[str, Any] = None


@dataclass
class TrainerDataLoaders:
    train: Iterable[Interaction]
    valid: Iterable[Interaction]
    test: Optional[Iterable[Interaction]] = None  # 扩展性好，支持可选参数

@dataclass
class TrainerComponents:
    evaluator_manager: EvaluatorManager

    recorder: ExplicitFeatureRecorder = ExplicitFeatureRecorder()
    timer: CudaNamedTimer = CudaNamedTimer()
