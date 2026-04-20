from collections.abc import Iterable

from betterbole.models.utils.general import MLP, ModuleFactory as MF


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


def build_expert_factory(in_dim, expert_dims, *, dropout_rate=0.1, activation="relu", batch_norm=False):
    dims = to_dims(expert_dims, ())
    return lambda: MLP(
        in_dim,
        *dims,
        dropout_rate=dropout_rate,
        activation=activation,
        batch_norm=batch_norm,
    )


def build_gate_factory(in_dim, out_dim):
    return MF.build_gate(in_dim, out_dim)


def build_tower_factory(in_dim):
    return MF.build_tower(in_dim)
