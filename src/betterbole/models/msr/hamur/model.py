from typing import Optional

import torch

from betterbole.emb import SchemaManager
from betterbole.models.msr.hamur.adapter import build_mlp_net
from betterbole.models.msr.hamur.adapter_dcn import build_dcn_net
from betterbole.models.msr.hamur.adapter_wd import build_widedeep_net
from betterbole.models.msr.hamur.common import (
    HAMURBaseModel,
    build_feature_view,
    infer_widedeep_fields,
)


def _resolve_backbone(backbone: str, depth):
    key = str(backbone).lower().replace("-", "").replace("_", "")
    if key in {"mlp2", "mlp2layer"}:
        return "mlp", 2
    if key in {"mlp7", "mlp7layer"}:
        return "mlp", 7
    if key == "mlp":
        return "mlp", int(depth or 2)
    if key in {"dcn", "dcnmd"}:
        return "dcn", None
    if key in {"widedeep", "widedeepmd", "wdmd"}:
        return "widedeep", None
    raise ValueError(f"Unsupported HAMUR backbone: {backbone}")


class HAMURModel(HAMURBaseModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            backbone: str = "mlp",
            depth: int = 2,
            use_adapter: bool = True,
            feature_fields=None,
            wide_fields=None,
            deep_fields=None,
            fcn_dims=None,
            mlp_dims=None,
            hyper_dims=None,
            k: int = 35,
            adapter_bottleneck_dim: int = 32,
            n_cross_layers: int = 2,
            dropout: Optional[float] = None,
            activation: str = "relu",
    ):
        super().__init__(manager, num_domains)
        self.backbone, self.depth = _resolve_backbone(backbone, depth)
        self.use_adapter = bool(use_adapter)
        self.activation = activation
        self._mode = None

        if self.backbone == "mlp":
            if wide_fields is not None or deep_fields is not None:
                raise ValueError("HAMUR mlp backbone does not use wide_fields/deep_fields")
            self._mode = "single"
            self.input_view = build_feature_view(
                self.omni_embedding,
                include_fields=feature_fields,
            )
            self.input_dim = self.input_view.embedding_dim
            self.network = build_mlp_net(
                input_dim=self.input_dim,
                num_domains=num_domains,
                depth=self.depth,
                use_adapter=self.use_adapter,
                fcn_dims=fcn_dims,
                hyper_dims=hyper_dims or (64,),
                k=k,
                bottleneck_dim=adapter_bottleneck_dim,
            )
            return

        if feature_fields is not None:
            raise ValueError(f"HAMUR {self.backbone} backbone does not use feature_fields")

        if self.backbone == "dcn":
            self._mode = "single"
            self.input_view = build_feature_view(
                self.omni_embedding,
                include_fields=None,
            )
            self.input_dim = self.input_view.embedding_dim
            self.network = build_dcn_net(
                input_dim=self.input_dim,
                num_domains=num_domains,
                use_adapter=self.use_adapter,
                n_cross_layers=n_cross_layers,
                mlp_dims=mlp_dims or (256, 128),
                hyper_dims=hyper_dims or (64,),
                k=k,
                bottleneck_dim=adapter_bottleneck_dim,
                dropout=0.0 if dropout is None else dropout,
                activation=activation,
            )
            return

        if self.backbone == "widedeep":
            self._mode = "wide_deep"
            wide_fields, deep_fields = infer_widedeep_fields(
                manager,
                wide_fields=wide_fields,
                deep_fields=deep_fields,
            )
            self.wide_view = build_feature_view(
                self.omni_embedding,
                include_fields=wide_fields,
            )
            self.deep_view = build_feature_view(
                self.omni_embedding,
                include_fields=deep_fields,
            )
            self.wide_dim = self.wide_view.embedding_dim
            self.deep_dim = self.deep_view.embedding_dim
            self.network = build_widedeep_net(
                wide_dim=self.wide_dim,
                deep_dim=self.deep_dim,
                num_domains=num_domains,
                use_adapter=self.use_adapter,
                mlp_dims=mlp_dims or (256, 128),
                hyper_dims=hyper_dims or (64,),
                k=k,
                bottleneck_dim=adapter_bottleneck_dim,
                dropout=0.2 if dropout is None else dropout,
                activation=activation,
            )
            return

        raise ValueError(f"Unsupported HAMUR backbone: {self.backbone}")

    def forward(self, *args):
        return self.network(*args)

    def predict(self, interaction):
        domain_ids = interaction[self.DOMAIN].long()
        if self._mode == "single":
            x = self._flatten_embedding(self.input_view(interaction))
            return self.forward(x, domain_ids)
        if self._mode == "wide_deep":
            wide_x = self._flatten_embedding(self.wide_view(interaction))
            deep_x = self._flatten_embedding(self.deep_view(interaction))
            return self.forward(wide_x, deep_x, domain_ids)
        raise RuntimeError(f"Unknown HAMUR predict mode: {self._mode}")
