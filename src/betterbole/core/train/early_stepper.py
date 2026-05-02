from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

@dataclass
class EarlyStepper:
    patience: int = 5
    min_delta: float = 0.0
    preferred_evaluators: tuple[str, ...] = ("overall", "all")
    preferred_metrics: tuple[str, ...] = (
        "auc",
        "gauc",
        "ndcg@10",
        "ndcg@20",
        "ndcg",
        "hr@10",
        "hr@20",
        "hr",
        "recall@10",
        "recall@20",
        "recall",
        "logloss",
        "loss",
    )

    def __post_init__(self):
        self.best_metric_value: Optional[float] = None
        self.best_metric_name: Optional[str] = None
        self.best_evaluator_name: Optional[str] = None
        self.best_epoch: Optional[int] = None
        self.bad_epoch_count: int = 0

    def _ordered_evaluator_names(self, summary_dict: dict[str, dict[str, float]]) -> list[str]:
        names = list(summary_dict.keys())
        ordered = [name for name in self.preferred_evaluators if name in summary_dict]
        ordered.extend(name for name in names if name not in ordered)
        return ordered

    def _pick_metric(self, summary_dict: dict[str, dict[str, float]]):
        if not summary_dict:
            return None

        for evaluator_name in self._ordered_evaluator_names(summary_dict):
            metrics_dict = summary_dict[evaluator_name]
            if not metrics_dict:
                continue
            lowered = {metric_name.lower(): metric_name for metric_name in metrics_dict}
            for preferred_metric in self.preferred_metrics:
                metric_name = lowered.get(preferred_metric.lower())
                if metric_name is not None:
                    return evaluator_name, metric_name, float(metrics_dict[metric_name])
            metric_name, metric_value = next(iter(metrics_dict.items()))
            return evaluator_name, metric_name, float(metric_value)
        return None

    @staticmethod
    def _metric_mode(metric_name: str) -> str:
        metric_name = metric_name.lower()
        if metric_name in {"loss", "logloss", "mae", "mse", "rmse"}:
            return "min"
        return "max"

    def step(
        self,
        summary_dict: Optional[dict[str, dict[str, float]]],
        epoch: Optional[int] = None,
    ) -> tuple[bool, bool]:
        if summary_dict is None:
            return False, False

        early_metric = self._pick_metric(summary_dict)
        if early_metric is None:
            return False, False

        evaluator_name, metric_name, metric_value = early_metric
        mode = self._metric_mode(metric_name)

        if self.best_metric_value is None:
            improved = True
        elif mode == "max":
            improved = metric_value > self.best_metric_value + self.min_delta
        else:
            improved = metric_value < self.best_metric_value - self.min_delta

        if improved:
            self.best_metric_value = metric_value
            self.best_metric_name = metric_name
            self.best_evaluator_name = evaluator_name
            self.best_epoch = epoch
            self.bad_epoch_count = 0
            return True, False

        self.bad_epoch_count += 1
        return False, self.bad_epoch_count >= self.patience
