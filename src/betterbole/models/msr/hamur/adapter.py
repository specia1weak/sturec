import torch
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.hamur.common import (
    HAMURAdapterCell,
    HAMURSingleInputModel,
    build_hyper_network,
    normalize_dims,
    select_by_domain_mask,
)


class _DomainTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dims):
        super().__init__()
        self.hidden_dims = normalize_dims(hidden_dims, name="fcn_dims")
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        current_dim = int(input_dim)
        for hidden_dim in self.hidden_dims:
            self.linears.append(nn.Linear(current_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(current_dim, 1)

    def forward(self, x: torch.Tensor, adapter_hooks=None) -> torch.Tensor:
        adapter_hooks = adapter_hooks or {}
        out = x
        for index, (linear, norm) in enumerate(zip(self.linears, self.norms)):
            out = linear(out)
            out = norm(out)
            out = self.relu(out)
            adapter_hook = adapter_hooks.get(index)
            if adapter_hook is not None:
                out = adapter_hook(out)
        return self.output_layer(out)


def _make_adapter_hook(adapter: HAMURAdapterCell, hyper_out: torch.Tensor):
    return lambda hidden, adp=adapter, hyper=hyper_out: adp(hidden, hyper)


class _HAMURMLPNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            fcn_dims,
            *,
            adapter_positions=(),
            hyper_dims=(64,),
            k: int = 65,
            bottleneck_dim: int = 32,
    ):
        super().__init__()
        self.fcn_dims = normalize_dims(fcn_dims, name="fcn_dims")
        self.domain_towers = nn.ModuleList([
            _DomainTower(input_dim, self.fcn_dims) for _ in range(num_domains)
        ])
        self.adapter_positions = tuple(int(position) for position in adapter_positions)

        if self.adapter_positions:
            self.k = int(k)
            self.hyper_net = build_hyper_network(input_dim, hyper_dims, self.k * self.k)
            self.adapters = nn.ModuleList([
                HAMURAdapterCell(
                    input_dim=self.fcn_dims[position],
                    k=self.k,
                    bottleneck_dim=bottleneck_dim,
                )
                for position in self.adapter_positions
            ])
        else:
            self.k = None
            self.hyper_net = None
            self.adapters = nn.ModuleList()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        outputs = []
        for tower in self.domain_towers:
            adapter_hooks = None
            if self.adapter_positions:
                hyper_out = self.hyper_net(x).reshape(-1, self.k, self.k)
                adapter_hooks = {
                    position: _make_adapter_hook(adapter, hyper_out)
                    for position, adapter in zip(self.adapter_positions, self.adapters)
                }
            logits = tower(x, adapter_hooks=adapter_hooks)
            outputs.append(torch.sigmoid(logits))
        return select_by_domain_mask(outputs, domain_ids)


def build_mlp_net(
        input_dim: int,
        num_domains: int,
        *,
        depth: int,
        use_adapter: bool,
        fcn_dims=None,
        hyper_dims=(64,),
        k: int = 65,
        bottleneck_dim: int = 32,
) -> _HAMURMLPNet:
    depth = int(depth)
    if fcn_dims is None:
        if depth == 2:
            fcn_dims = (256, 128)
        elif depth == 7:
            fcn_dims = (1024, 512, 512, 256, 256, 64, 64)
        else:
            raise ValueError(f"Unsupported MLP depth: {depth}")

    fcn_dims = normalize_dims(fcn_dims, name="fcn_dims")
    if len(fcn_dims) != depth:
        raise ValueError(f"MLP depth={depth} expects {depth} hidden dims, got {len(fcn_dims)}")

    adapter_positions = ()
    if use_adapter:
        if depth == 2:
            adapter_positions = (1,)
        elif depth == 7:
            adapter_positions = (5, 6)
        else:
            raise ValueError(f"Unsupported adapter configuration for MLP depth: {depth}")

    return _HAMURMLPNet(
        input_dim=input_dim,
        num_domains=num_domains,
        fcn_dims=fcn_dims,
        adapter_positions=adapter_positions,
        hyper_dims=hyper_dims,
        k=k,
        bottleneck_dim=bottleneck_dim,
    )


class Mlp_2_Layer(HAMURSingleInputModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            fcn_dims=(256, 128),
            feature_fields=None,
    ):
        super().__init__(manager, num_domains, feature_fields=feature_fields)
        self.network = build_mlp_net(
            input_dim=self.input_dim,
            num_domains=num_domains,
            depth=2,
            use_adapter=False,
            fcn_dims=fcn_dims,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.network(x, domain_ids)


class Mlp_7_Layer(HAMURSingleInputModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            fcn_dims=(1024, 512, 512, 256, 256, 64, 64),
            feature_fields=None,
    ):
        super().__init__(manager, num_domains, feature_fields=feature_fields)
        self.network = build_mlp_net(
            input_dim=self.input_dim,
            num_domains=num_domains,
            depth=7,
            use_adapter=False,
            fcn_dims=fcn_dims,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.network(x, domain_ids)


class MLP_adap_2_layer_1_adp(HAMURSingleInputModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            fcn_dims=(256, 128),
            hyper_dims=(64,),
            k: int = 32,
            feature_fields=None,
    ):
        super().__init__(manager, num_domains, feature_fields=feature_fields)
        self.network = build_mlp_net(
            input_dim=self.input_dim,
            num_domains=num_domains,
            depth=2,
            use_adapter=True,
            fcn_dims=fcn_dims,
            hyper_dims=hyper_dims,
            k=k,
            bottleneck_dim=32,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.network(x, domain_ids)


class MLP_adap_7_layer_2_adp(HAMURSingleInputModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            fcn_dims=(1024, 512, 512, 256, 256, 64, 64),
            hyper_dims=(64,),
            k: int = 32,
            feature_fields=None,
    ):
        super().__init__(manager, num_domains, feature_fields=feature_fields)
        self.network = build_mlp_net(
            input_dim=self.input_dim,
            num_domains=num_domains,
            depth=7,
            use_adapter=True,
            fcn_dims=fcn_dims,
            hyper_dims=hyper_dims,
            k=k,
            bottleneck_dim=32,
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.network(x, domain_ids)
