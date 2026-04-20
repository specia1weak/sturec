from betterbole.models.backbone.mmoe import SingleLayerMMoE, SingleLayerMTLMMoE
from betterbole.models.backbone.ple import PLE, PLEVersion1, PLEVersion2, PLEVersion3, PLEVersion4
from betterbole.models.backbone.shabtm import SharedBottomLess, SharedBottomPlus
from betterbole.models.backbone.star import STAR, StarPle


def _count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


BACKBONE_REGISTRY = {
    "ple": PLE,
    "small_ple": PLE,
    "pleversion1": PLEVersion1,
    "pleversion2": PLEVersion2,
    "pleversion3": PLEVersion3,
    "pleversion4": PLEVersion4,
    "mmoe": SingleLayerMMoE,
    "small_mmoe": SingleLayerMMoE,
    "singlelayermmoe": SingleLayerMMoE,
    "single_layer_mmoe": SingleLayerMMoE,
    "sharedbottomless": SharedBottomLess,
    "shared_bottom_less": SharedBottomLess,
    "sharedbottomplus": SharedBottomPlus,
    "shared_bottom_plus": SharedBottomPlus,
    "star": STAR,
    "starple": StarPle,
    "star_ple": StarPle,
}


def resolve_backbone(backbone_type: str):
    backbone_name = backbone_type.lower()
    try:
        return BACKBONE_REGISTRY[backbone_name]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKBONE_REGISTRY))
        raise ValueError(f"Unsupported backbone_type='{backbone_type}'. Supported: [{supported}]") from exc


def build(backbone_type: str, num_domains: int, input_dim: int, size: str = "small", **overrides):
    if size.lower() != "small":
        raise ValueError(f"Unsupported size='{size}'. Currently only 'small' preset is available.")
    backbone_cls = resolve_backbone(backbone_type)
    return backbone_cls(emb_size=input_dim, num_domains=num_domains, **overrides)


class SmallBackboneFactory:
    @staticmethod
    def build(backbone_type: str, num_domains: int, input_dim: int, **overrides):
        return build(
            backbone_type=backbone_type,
            num_domains=num_domains,
            input_dim=input_dim,
            **overrides,
        )

    @staticmethod
    def derive_ple_config(input_dim: int):
        return {"expert_dims": PLE.default_expert_dims(input_dim)}

    @staticmethod
    def derive_mmoe_config(input_dim: int):
        return {"expert_dims": SingleLayerMMoE.default_expert_dims(input_dim)}

    @staticmethod
    def estimate_params(num_domains: int, input_dim: int):
        ple_model = build("ple", num_domains=num_domains, input_dim=input_dim)
        mmoe_model = build("mmoe", num_domains=num_domains, input_dim=input_dim)
        return {
            "ple_params": _count_params(ple_model),
            "mmoe_params": _count_params(mmoe_model),
            "ratio_mmoe_over_ple": _count_params(mmoe_model) / max(1, _count_params(ple_model)),
            "ple_config": SmallBackboneFactory.derive_ple_config(input_dim),
            "mmoe_config": SmallBackboneFactory.derive_mmoe_config(input_dim),
        }


class SmallBacboneFactory(SmallBackboneFactory):
    pass


__all__ = [
    "PLE",
    "PLEVersion1",
    "PLEVersion2",
    "PLEVersion3",
    "PLEVersion4",
    "SingleLayerMMoE",
    "SingleLayerMTLMMoE",
    "SharedBottomLess",
    "SharedBottomPlus",
    "STAR",
    "StarPle",
    "BACKBONE_REGISTRY",
    "resolve_backbone",
    "SmallBackboneFactory",
    "SmallBacboneFactory",
    "build",
]
