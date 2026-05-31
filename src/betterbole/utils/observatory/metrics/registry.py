from typing import Dict, Tuple


DEFAULT_TENSOR_METRICS: Tuple[str, ...] = (
    "feature_mean",
    "feature_var",
    "effective_rank",
    "participation_ratio",
    "stable_rank",
    "dead_dim_ratio",
    "mean_abs_corr",
    "max_abs_corr",
)

ROUTING_METRICS: Tuple[str, ...] = (
    "gate_mean",
    "gate_var",
    "gate_entropy",
    "expert_usage",
    "expert_dead_ratio",
)

VQ_METRICS: Tuple[str, ...] = (
    "used_code_ratio",
    "code_entropy",
    "quantized_cos",
    "residual_norm",
)

METRIC_GROUPS: Dict[str, Tuple[str, ...]] = {
    "default_tensor": DEFAULT_TENSOR_METRICS,
    "routing": ROUTING_METRICS,
    "vq": VQ_METRICS,
}
