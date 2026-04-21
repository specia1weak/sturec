from collections.abc import Iterable

from betterbole.models.utils.general import MLP


def to_dims(value, default):
    if value is None:
        value = default

    if isinstance(value, int):
        dims = (int(value),)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        dims = tuple(int(v) for v in value)
    else:
        raise TypeError(f"Unsupported dims type: {type(value)}")

    if not dims:
        raise ValueError("dims can not be empty")
    return dims


def default_backbone_dims(input_dim: int, depth: int = 2) -> tuple[int, ...]:
    input_dim = max(1, int(input_dim))
    if depth <= 1:
        return (input_dim,)
    hidden = max(1, input_dim)
    return tuple(hidden for _ in range(depth))


def build_mlp(
        input_dim: int,
        hidden_dims,
        *,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
):
    dims = to_dims(hidden_dims, default_backbone_dims(input_dim))
    return MLP(input_dim, *dims, dropout_rate=dropout_rate, activation=activation, batch_norm=batch_norm)
