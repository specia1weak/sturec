from betterbole.models.msr.backbone.epnet import EPNetBackbone
from betterbole.models.msr.backbone.m2m import M2MBackbone
from betterbole.models.msr.backbone.m3oe import M3oEBackbone
from betterbole.models.msr.backbone.mmoe import MMoEBackbone
from betterbole.models.msr.backbone.ple import PLEBackbone
from betterbole.models.msr.backbone.ppnet import PPNetBackbone
from betterbole.models.msr.backbone.sharedbottom import SharedBottomBackbone
from betterbole.models.msr.backbone.star import STARBackbone


BACKBONE_REGISTRY = {
    "sharedbottom": SharedBottomBackbone,
    "shared_bottom": SharedBottomBackbone,
    "mmoe": MMoEBackbone,
    "ple": PLEBackbone,
    "star": STARBackbone,
    "m2m": M2MBackbone,
    "epnet": EPNetBackbone,
    "ppnet": PPNetBackbone,
    "m3oe": M3oEBackbone,
}


def build_backbone(
        name: str,
        input_dim: int,
        num_domains: int,
        **kwargs,
):
    key = name.lower()
    if key not in BACKBONE_REGISTRY:
        supported = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(f"Unsupported backbone: {name}. Supported: {supported}")
    return BACKBONE_REGISTRY[key](
        input_dim=input_dim,
        num_domains=num_domains,
        **kwargs,
    )
