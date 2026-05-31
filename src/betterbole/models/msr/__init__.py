from typing import Dict, Type, Union

from betterbole.emb import SchemaManager
from betterbole.models.msr.automtl import AutoMTLModel
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.crocodile_v1 import CrocodileV1Model
from betterbole.models.msr.feature_gate import FeatureGateModel
from betterbole.models.msr.hamur import HAMURModel
from betterbole.models.msr.hierrec import HierRec
from betterbole.models.msr.m2m import M2MModel
from betterbole.models.msr.m3oe import M3oEModel, M3oEVersion1Model, M3oEVersion2Model
from betterbole.models.msr.mmoe import MMoEModel
from betterbole.models.msr.multiemb import MultiEmbModel
from betterbole.models.msr.pareto import ParetoModel
from betterbole.models.msr.pepnet import EPNetModel, PEPNetModel, PPNetModel
from betterbole.models.msr.ple import PLEModel, PLEVersion1Model
from betterbole.models.msr.riple import RIPLEModel
from betterbole.models.msr.sdsp import SDSPMMoEModel, SDSPPLEModel
from betterbole.models.msr.sharedbottom import SharedBottomLessModel, SharedBottomModel, SharedBottomPlusModel
from betterbole.models.msr.ssim import SSIMModel
from betterbole.models.msr.star import STARModel


MODEL_REGISTRY: Dict[str, Type[MSRModel]] = {
    "sharedbottom": SharedBottomModel,
    "sharedbottom_less": SharedBottomLessModel,
    "sharedbottom_plus": SharedBottomPlusModel,
    "mmoe": MMoEModel,
    "ple": PLEModel,
    "ple_v1": PLEVersion1Model,
    "star": STARModel,
    "m3oe": M3oEModel,
    "m3oe_v1": M3oEVersion1Model,
    "m3oe_v2": M3oEVersion2Model,
    "m2m": M2MModel,
    "ppnet": PPNetModel,
    "epnet": EPNetModel,
    "pepnet": PEPNetModel,
    "feature_gate": FeatureGateModel,
    "crocodile": CrocodileV1Model,
    "pareto": ParetoModel,
    "hierrec": HierRec,
    "automtl": AutoMTLModel,
    "riple": RIPLEModel,
    "hamur": HAMURModel,
    "sdsp_ple": SDSPPLEModel,
    "sdsp_mmoe": SDSPMMoEModel,
    "ssim": SSIMModel,
    "multiemb": MultiEmbModel,
}

def update_register(**kwargs) -> Dict[str, Type[MSRModel]]:
    for name, model_cls in kwargs.items():
        if not isinstance(name, str):
            raise TypeError(f"Model registry key must be str, got {type(name)}")
        if not isinstance(model_cls, type) or not issubclass(model_cls, MSRModel):
            raise TypeError(
                f"Model registry value for '{name}' must be an MSRModel subclass, got {model_cls!r}"
            )
        MODEL_REGISTRY[name.lower()] = model_cls
    return MODEL_REGISTRY


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
    "MODEL_REGISTRY",
    "update_register",
    "build_model",
    "HierRec",
    "SharedBottomModel",
    "SharedBottomLessModel",
    "SharedBottomPlusModel",
    "MMoEModel",
    "PLEModel",
    "PLEVersion1Model",
    "STARModel",
    "M3oEModel",
    "M3oEVersion1Model",
    "M3oEVersion2Model",
    "RIPLEModel",
    "SDSPMMoEModel",
    "SDSPPLEModel",
    "SSIMModel",
    "MultiEmbModel",
    "M2MModel",
    "PPNetModel",
    "EPNetModel",
    "PEPNetModel",
    "FeatureGateModel",
    "CrocodileV1Model",
    "ParetoModel",
    "AutoMTLModel",
    "HAMURModel",
]
