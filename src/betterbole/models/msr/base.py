from abc import ABC

from betterbole.emb import SchemaManager
from betterbole.models.base import BaseModel


class MSRModel(ABC, BaseModel):
    def __init__(self, manager: SchemaManager, num_domains: int, **kwargs):
        super(MSRModel, self).__init__(manager)
        self.num_domains = num_domains

    @classmethod
    def from_manager(cls, manager: SchemaManager, num_domains: int, **kwargs) -> 'MSRModel':
        return cls(manager, num_domains, **kwargs)