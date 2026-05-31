from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from betterbole.core.train.context import TrainContext


MetricPath = Union[str, Sequence[str], Tuple[str, ...]]


class TwoStageRetrainMixin:
    """
    Internal search -> retrain phase controller.

    The outer trainer remains unchanged. Models opt into this mixin and:
    1. train in search mode for `search_epochs`
    2. track the best search checkpoint using validation metrics
    3. switch themselves into retrain mode inside `on_eval_epoch_end`
    4. clear optimizer state and freeze search-only parameters
    """

    def _init_two_stage_retrain(
            self,
            *,
            search_epochs: int,
            metric_path: MetricPath = ("overall", "auc"),
            restore_best_before_retrain: bool = True,
    ) -> None:
        self.search_epochs = max(1, int(search_epochs))
        self.retrain_metric_path = metric_path
        self.restore_best_before_retrain = bool(restore_best_before_retrain)

        self.in_retrain_phase = False
        self._search_best_metric = float("-inf")
        self._search_best_state: Optional[Dict[str, Any]] = None
        self._retrain_start_epoch: Optional[int] = None

    def on_train_epoch_start(self, ctx: TrainContext) -> None:
        del ctx

    def on_train_epoch_end(self, ctx: TrainContext) -> None:
        del ctx

    def _capture_two_stage_state(self) -> Dict[str, Any]:
        return {
            "model_state": copy.deepcopy(self.state_dict()),
        }

    def _restore_two_stage_state(self, state: Dict[str, Any]) -> None:
        model_state = state.get("model_state")
        if model_state is not None:
            self.load_state_dict(model_state)

    def _prepare_retrain_phase(self) -> None:
        """
        Override in child classes.

        Typical actions:
        - harden masks / set ticket=True
        - freeze mask-network parameters
        - optionally re-enable only backbone params
        """
        return None

    def _extract_two_stage_metric(self, metrics: Optional[Dict[str, Any]]) -> Optional[float]:
        if not metrics:
            return None

        path = self.retrain_metric_path
        if isinstance(path, str):
            value = metrics.get(path)
            if isinstance(value, (int, float)):
                return float(value)
            return None

        value: Any = metrics
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _update_search_best_state(self, metrics: Optional[Dict[str, Any]]) -> None:
        if self.in_retrain_phase:
            return
        metric_value = self._extract_two_stage_metric(metrics)
        if metric_value is None:
            return
        if metric_value > self._search_best_metric:
            self._search_best_metric = metric_value
            self._search_best_state = self._capture_two_stage_state()

    def _reset_optimizer_for_retrain(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.state.clear()

    def _switch_to_retrain_if_needed(self, metrics: Optional[Dict[str, Any]], ctx: TrainContext) -> bool:
        self._update_search_best_state(metrics)

        if self.in_retrain_phase:
            return False
        if int(ctx.epoch) < self.search_epochs:
            return False

        if self.restore_best_before_retrain and self._search_best_state is not None:
            self._restore_two_stage_state(self._search_best_state)
        self._prepare_retrain_phase()
        self._reset_optimizer_for_retrain(ctx.optimizer)
        self.in_retrain_phase = True
        self._retrain_start_epoch = int(ctx.epoch)
        return True

    def two_stage_status(self) -> Dict[str, Any]:
        return {
            "in_retrain_phase": bool(self.in_retrain_phase),
            "search_best_metric": 0.0 if self._search_best_metric == float("-inf") else float(self._search_best_metric),
            "search_epochs": int(self.search_epochs),
            "retrain_start_epoch": self._retrain_start_epoch,
        }
