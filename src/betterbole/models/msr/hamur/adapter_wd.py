import torch

from betterbole.emb import SchemaManager
from betterbole.models.msr.hamur.common import (
    HAMURAdapterCell,
    HAMURLR,
    HAMURMLP,
    HAMURWideDeepModel,
    build_hyper_network,
    normalize_dims,
    select_by_domain_mask,
)


class _HAMURWideDeepNet(torch.nn.Module):
    def __init__(
            self,
            wide_dim: int,
            deep_dim: int,
            num_domains: int,
            *,
            mlp_dims,
            dropout: float = 0.2,
            activation: str = "relu",
    ):
        super().__init__()
        self.num_domains = int(num_domains)
        self.mlp_dims = normalize_dims(mlp_dims, name="mlp_dims")
        self.linear = HAMURLR(wide_dim)
        self.mlp = HAMURMLP(
            deep_dim,
            output_layer=True,
            dims=self.mlp_dims,
            dropout=dropout,
            activation=activation,
        )

    def forward(
            self,
            wide_x: torch.Tensor,
            deep_x: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        for _ in range(self.num_domains):
            y_wide = self.linear(wide_x)
            y_deep = self.mlp(deep_x)
            outputs.append(torch.sigmoid(y_wide + y_deep))
        return select_by_domain_mask(outputs, domain_ids)


class _HAMURWideDeepAdapterNet(torch.nn.Module):
    def __init__(
            self,
            wide_dim: int,
            deep_dim: int,
            num_domains: int,
            *,
            mlp_dims,
            hyper_dims=(64,),
            k: int = 65,
            bottleneck_dim: int = 32,
            dropout: float = 0.2,
            activation: str = "relu",
    ):
        super().__init__()
        self.num_domains = int(num_domains)
        self.mlp_dims = normalize_dims(mlp_dims, name="mlp_dims")
        self.k = int(k)
        self.linear = HAMURLR(wide_dim)
        self.mlp = HAMURMLP(
            deep_dim,
            output_layer=False,
            dims=self.mlp_dims,
            dropout=dropout,
            activation=activation,
        )
        self.mlp_final = HAMURLR(self.mlp_dims[-1])
        self.hyper_net = build_hyper_network(wide_dim + deep_dim, hyper_dims, self.k * self.k)
        self.adapter = HAMURAdapterCell(
            input_dim=self.mlp_dims[-1],
            k=self.k,
            bottleneck_dim=bottleneck_dim,
        )

    def forward(
            self,
            wide_x: torch.Tensor,
            deep_x: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        hyper_out = self.hyper_net(torch.cat((wide_x, deep_x), dim=1)).reshape(-1, self.k, self.k)
        outputs = []
        for _ in range(self.num_domains):
            y_wide = self.linear(wide_x)
            y_deep = self.mlp(deep_x)
            mlp_out = self.adapter(y_deep, hyper_out)
            mlp_out = self.mlp_final(mlp_out)
            outputs.append(torch.sigmoid(y_wide + mlp_out))
        return select_by_domain_mask(outputs, domain_ids)


def build_widedeep_net(
        wide_dim: int,
        deep_dim: int,
        num_domains: int,
        *,
        use_adapter: bool,
        mlp_dims=(256, 128),
        hyper_dims=(64,),
        k: int = 65,
        bottleneck_dim: int = 32,
        dropout: float = 0.2,
        activation: str = "relu",
) -> torch.nn.Module:
    if use_adapter:
        return _HAMURWideDeepAdapterNet(
            wide_dim=wide_dim,
            deep_dim=deep_dim,
            num_domains=num_domains,
            mlp_dims=mlp_dims,
            hyper_dims=hyper_dims,
            k=k,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            activation=activation,
        )
    return _HAMURWideDeepNet(
        wide_dim=wide_dim,
        deep_dim=deep_dim,
        num_domains=num_domains,
        mlp_dims=mlp_dims,
        dropout=dropout,
        activation=activation,
    )


class WideDeep_MD(HAMURWideDeepModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            mlp_dims=(256, 128),
            dropout: float = 0.2,
            activation: str = "relu",
            wide_fields=None,
            deep_fields=None,
    ):
        super().__init__(
            manager,
            num_domains,
            wide_fields=wide_fields,
            deep_fields=deep_fields,
        )
        self.network = build_widedeep_net(
            wide_dim=self.wide_dim,
            deep_dim=self.deep_dim,
            num_domains=num_domains,
            use_adapter=False,
            mlp_dims=mlp_dims,
            dropout=dropout,
            activation=activation,
        )

    def forward(
            self,
            wide_x: torch.Tensor,
            deep_x: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(wide_x, deep_x, domain_ids)


class WideDeep_MD_adp(HAMURWideDeepModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            mlp_dims=(256, 128),
            hyper_dims=(64,),
            k: int = 32,
            dropout: float = 0.2,
            activation: str = "relu",
            wide_fields=None,
            deep_fields=None,
    ):
        super().__init__(
            manager,
            num_domains,
            wide_fields=wide_fields,
            deep_fields=deep_fields,
        )
        self.network = build_widedeep_net(
            wide_dim=self.wide_dim,
            deep_dim=self.deep_dim,
            num_domains=num_domains,
            use_adapter=True,
            mlp_dims=mlp_dims,
            hyper_dims=hyper_dims,
            k=k,
            bottleneck_dim=32,
            dropout=dropout,
            activation=activation,
        )

    def forward(
            self,
            wide_x: torch.Tensor,
            deep_x: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(wide_x, deep_x, domain_ids)
