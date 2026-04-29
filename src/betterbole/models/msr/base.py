import inspect
import warnings
from abc import ABC

from betterbole.emb import SchemaManager
from betterbole.models.base import BaseModel


class MSRModel(BaseModel):
    def __init__(self, manager: SchemaManager, num_domains: int, **kwargs):
        super(MSRModel, self).__init__(manager)
        self.num_domains = num_domains

    @classmethod
    def from_manager(cls, manager: SchemaManager, num_domains: int, **kwargs) -> 'MSRModel':
        init_signature = inspect.signature(cls.__init__)
        parameters = init_signature.parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_var_kwargs:
            return cls(manager, num_domains, **kwargs)

        valid_kwargs = {}
        ignored_keys = []
        for key, value in kwargs.items():
            if key in parameters and key not in {"self", "manager", "num_domains"}:
                valid_kwargs[key] = value
            else:
                ignored_keys.append(key)

        if ignored_keys:
            warnings.warn(
                f"{cls.__name__}.from_manager ignored unsupported kwargs: {sorted(ignored_keys)}",
                stacklevel=2,
            )
        return cls(manager, num_domains, **valid_kwargs)
