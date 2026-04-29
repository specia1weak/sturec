from typing import Dict, Type, Union

from betterbole.emb import SchemaManager
from betterbole.models.msr.automtl import AutoMTLModel
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.epnet import EPNetModel
from betterbole.models.msr.hierrec import HierRec
from betterbole.models.msr.m3oe import M3oEModel, M3oEVersion1Model, M3oEVersion2Model
from betterbole.models.msr.m2m import M2MModel
from betterbole.models.msr.mmoe import MMoEModel
from betterbole.models.msr.ppnet import PPNetModel
from betterbole.models.msr.ple import PLEModel
from betterbole.models.msr.sharedbottom import SharedBottomModel
from betterbole.models.msr.star import STARModel


MODEL_REGISTRY: Dict[str, Type[MSRModel]] = {
    "sharedbottom": SharedBottomModel,
    "mmoe": MMoEModel,
    "ple": PLEModel,
    "star": STARModel,
    "m3oe": M3oEModel,
    "m3oe_v1": M3oEVersion1Model,
    "m3oe_v2": M3oEVersion2Model,
    "m2m": M2MModel,
    "ppnet": PPNetModel,
    "epnet": EPNetModel,
    "hierrec": HierRec,
    "automtl": AutoMTLModel,
}


def build_model(
        schema_manager: SchemaManager,
        num_domains: int,
        model_cls: Union[str, Type[MSRModel]],
        **model_kwargs,
):

    if isinstance(model_cls, str):
        model_name = model_cls
        model_cls = MODEL_REGISTRY.get(model_name.lower())
    else:
        model_name = model_cls.__name__
    if model_cls is None:
        raise ValueError(f"Unknown model_name={model_name}. Available: {sorted(MODEL_REGISTRY.keys())}")
    return model_cls.from_manager(schema_manager, num_domains, **dict(model_kwargs or {}))


__all__ = [
    "MSRModel",
    "HierRec",
    "SharedBottomModel",
    "MMoEModel",
    "PLEModel",
    "STARModel",
    "M3oEModel",
    "M3oEVersion1Model",
    "M3oEVersion2Model",
    "M2MModel",
    "PPNetModel",
    "EPNetModel",
    "AutoMTLModel",
    "MODEL_REGISTRY",
    "build_model",
]
