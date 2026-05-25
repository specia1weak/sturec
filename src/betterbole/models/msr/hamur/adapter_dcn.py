import torch

from betterbole.emb import SchemaManager
from betterbole.models.msr.hamur.common import (
    HAMURAdapterCell,
    HAMURCrossNetwork,
    HAMURLR,
    HAMURMLP,
    HAMURSingleInputModel,
    build_hyper_network,
    normalize_dims,
    select_by_domain_mask,
)


class _HAMURDCNNet(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            *,
            n_cross_layers: int = 2,
            mlp_dims=(256, 128),
            dropout: float = 0.0,
            activation: str = "relu",
    ):
        super().__init__()
        self.num_domains = int(num_domains)
        self.n_cross_layers = int(n_cross_layers)
        self.mlp_dims = normalize_dims(mlp_dims, name="mlp_dims")
        self.cn = HAMURCrossNetwork(input_dim, self.n_cross_layers)
        self.mlp = HAMURMLP(
            input_dim,
            output_layer=False,
            dims=self.mlp_dims,
            dropout=dropout,
            activation=activation,
        )
        self.linear = HAMURLR(input_dim + self.mlp_dims[-1])

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        outputs = []
        for _ in range(self.num_domains):
            cn_out = self.cn(x)
            mlp_out = self.mlp(x)
            y = self.linear(torch.cat([cn_out, mlp_out], dim=1))
            outputs.append(torch.sigmoid(y))
        return select_by_domain_mask(outputs, domain_ids)


class _HAMURDCNAdapterNet(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            *,
            n_cross_layers: int = 2,
            mlp_dims=(256, 128),
            hyper_dims=(64,),
            k: int = 65,
            bottleneck_dim: int = 32,
            dropout: float = 0.0,
            activation: str = "relu",
    ):
        super().__init__()
        self.num_domains = int(num_domains)
        self.n_cross_layers = int(n_cross_layers)
        self.mlp_dims = normalize_dims(mlp_dims, name="mlp_dims")
        self.k = int(k)
        self.cn = HAMURCrossNetwork(input_dim, self.n_cross_layers)
        self.mlp = HAMURMLP(
            input_dim,
            output_layer=False,
            dims=self.mlp_dims,
            dropout=dropout,
            activation=activation,
        )
        self.linear = HAMURLR(input_dim + self.mlp_dims[-1])
        self.hyper_net = build_hyper_network(input_dim, hyper_dims, self.k * self.k)
        self.adapter = HAMURAdapterCell(
            input_dim=self.mlp_dims[-1],
            k=self.k,
            bottleneck_dim=bottleneck_dim,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        outputs = []
        for _ in range(self.num_domains):
            hyper_out = self.hyper_net(x).reshape(-1, self.k, self.k)
            cn_out = self.cn(x)
            mlp_out = self.mlp(x)
            mlp_out = self.adapter(mlp_out, hyper_out)
            y = self.linear(torch.cat([cn_out, mlp_out], dim=1))
            outputs.append(torch.sigmoid(y))
        return select_by_domain_mask(outputs, domain_ids)


def build_dcn_net(
        input_dim: int,
        num_domains: int,
        *,
        use_adapter: bool,
        n_cross_layers: int = 2,
        mlp_dims=(256, 128),
        hyper_dims=(64,),
        k: int = 65,
        bottleneck_dim: int = 32,
        dropout: float = 0.0,
        activation: str = "relu",
) -> torch.nn.Module:
    if use_adapter:
        return _HAMURDCNAdapterNet(
            input_dim=input_dim,
            num_domains=num_domains,
            n_cross_layers=n_cross_layers,
            mlp_dims=mlp_dims,
            hyper_dims=hyper_dims,
            k=k,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            activation=activation,
        )
    return _HAMURDCNNet(
        input_dim=input_dim,
        num_domains=num_domains,
        n_cross_layers=n_cross_layers,
        mlp_dims=mlp_dims,
        dropout=dropout,
        activation=activation,
    )


class DCN_MD(HAMURSingleInputModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            n_cross_layers: int = 2,
            mlp_dims=(256, 128),
            dropout: float = 0.0,
            activation: str = "relu",
            feature_fields=None,
    ):
        super().__init__(manager, num_domains, feature_fields=feature_fields)
        self.network = build_dcn_net(
            input_dim=self.input_dim,
            num_domains=num_domains,
            use_adapter=False,
            n_cross_layers=n_cross_layers,
            mlp_dims=mlp_dims,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.network(x, domain_ids)


class DCN_MD_adp(HAMURSingleInputModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            n_cross_layers: int = 2,
            mlp_dims=(256, 128),
            hyper_dims=(128,),
            k: int = 32,
            dropout: float = 0.0,
            activation: str = "relu",
            feature_fields=None,
    ):
        super().__init__(manager, num_domains, feature_fields=feature_fields)
        self.network = build_dcn_net(
            input_dim=self.input_dim,
            num_domains=num_domains,
            use_adapter=True,
            n_cross_layers=n_cross_layers,
            mlp_dims=mlp_dims,
            hyper_dims=hyper_dims,
            k=k,
            bottleneck_dim=32,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.network(x, domain_ids)
