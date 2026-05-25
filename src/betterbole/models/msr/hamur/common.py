from collections.abc import Iterable

import torch
from torch import nn

from betterbole.core.enum_type import FeatureSource
from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import EmbView
from betterbole.emb.schema import EmbType
from betterbole.models.msr.base import MSRModel
from betterbole.models.utils.activation import activation_layer


HAMUR_TARGET_SOURCES = (
    FeatureSource.USER_ID,
    FeatureSource.USER,
    FeatureSource.ITEM_ID,
    FeatureSource.ITEM,
    FeatureSource.INTERACTION,
)


def normalize_dims(dims, *, name: str) -> tuple[int, ...]:
    if dims is None:
        raise ValueError(f"{name} can not be None")
    if isinstance(dims, int):
        return (int(dims),)
    if isinstance(dims, Iterable) and not isinstance(dims, (str, bytes)):
        values = tuple(int(dim) for dim in dims)
        if not values:
            raise ValueError(f"{name} can not be empty")
        return values
    raise TypeError(f"Unsupported {name} type: {type(dims)}")


def build_feature_view(
        omni_embedding,
        include_fields=None,
        exclude_fields=(),
) -> EmbView:
    if isinstance(include_fields, (str, bytes)):
        include_fields = (include_fields,)
    if include_fields is None:
        return EmbView(
            omni_embedding,
            target_sources=HAMUR_TARGET_SOURCES,
            exclude_fields=tuple(exclude_fields or ()),
        )
    return EmbView(
        omni_embedding,
        include_fields=tuple(include_fields),
    )

def infer_widedeep_fields(
        manager: SchemaManager,
        wide_fields=None,
        deep_fields=None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(wide_fields, (str, bytes)):
        wide_fields = (wide_fields,)
    if isinstance(deep_fields, (str, bytes)):
        deep_fields = (deep_fields,)

    if wide_fields is not None and deep_fields is not None:
        return tuple(wide_fields), tuple(deep_fields)

    inferred_wide = []
    inferred_deep = []
    all_fields = []
    for setting in manager.settings:
        field_name = setting.field_name
        if field_name not in all_fields:
            all_fields.append(field_name)
        if setting.emb_type in {EmbType.DENSE, EmbType.VECTOR_DENSE}:
            inferred_wide.append(field_name)
        else:
            inferred_deep.append(field_name)

    if wide_fields is None and deep_fields is None:
        if not inferred_wide or not inferred_deep:
            raise ValueError(
                "HAMUR widedeep mode could not infer both wide_fields and deep_fields. "
                "Please pass them explicitly."
            )
        return tuple(inferred_wide), tuple(inferred_deep)

    if wide_fields is None:
        deep_fields = tuple(deep_fields)
        wide_fields = tuple(field for field in all_fields if field not in set(deep_fields))
        return tuple(wide_fields), deep_fields

    wide_fields = tuple(wide_fields)
    deep_fields = tuple(field for field in all_fields if field not in set(wide_fields))
    return wide_fields, tuple(deep_fields)


def select_by_domain_mask(domain_outputs: list[torch.Tensor], domain_ids: torch.Tensor) -> torch.Tensor:
    final = torch.zeros_like(domain_outputs[0])
    for domain_index, output in enumerate(domain_outputs):
        domain_mask = (domain_ids == domain_index).unsqueeze(1)
        final = torch.where(domain_mask, output, final)
    return final.squeeze(1)


class HAMURLR(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class HAMURMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            *,
            output_layer: bool = True,
            dims=None,
            dropout: float = 0.0,
            activation: str = "relu",
    ):
        super().__init__()
        hidden_dims = list(normalize_dims(dims, name="dims"))
        layers = []
        current_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            current_dim = hidden_dim
        if output_layer:
            layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class HAMURCrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = int(num_layers)
        self.w = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(self.num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros((input_dim,))) for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for index in range(self.num_layers):
            xw = self.w[index](x)
            x = x0 * xw + self.b[index] + x
        return x


class HAMURAdapterCell(nn.Module):
    def __init__(
            self,
            input_dim: int,
            k: int,
            bottleneck_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.k = int(k)
        self.bottleneck_dim = int(bottleneck_dim)
        self.sigmoid = activation_layer("sigmoid")
        self.eps = 1e-5

        self.u_down = nn.Parameter(torch.ones((self.input_dim, self.k)))
        self.v_down = nn.Parameter(torch.ones((self.k, self.bottleneck_dim)))
        self.u_up = nn.Parameter(torch.ones((self.bottleneck_dim, self.k)))
        self.v_up = nn.Parameter(torch.ones((self.k, self.input_dim)))

        self.bias_down = nn.Parameter(torch.zeros((self.bottleneck_dim,)))
        self.bias_up = nn.Parameter(torch.zeros((self.input_dim,)))
        self.gamma = nn.Parameter(torch.ones((self.input_dim,)))
        self.bias = nn.Parameter(torch.zeros((self.input_dim,)))

    def forward(self, x: torch.Tensor, hyper_out: torch.Tensor) -> torch.Tensor:
        down_weight = torch.einsum("mi,bij,jn->bmn", self.u_down, hyper_out, self.v_down)
        tmp_out = torch.einsum("bf,bfj->bj", x, down_weight)
        tmp_out = tmp_out + self.bias_down
        tmp_out = self.sigmoid(tmp_out)

        up_weight = torch.einsum("mi,bij,jn->bmn", self.u_up, hyper_out, self.v_up)
        tmp_out = torch.einsum("bf,bfj->bj", tmp_out, up_weight)
        tmp_out = tmp_out + self.bias_up

        mean = tmp_out.mean(dim=0)
        var = tmp_out.var(dim=0)
        x_norm = (tmp_out - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.bias
        return out + x


def build_hyper_network(input_dim: int, hyper_dims, output_dim: int) -> nn.Sequential:
    hyper_stream = [*normalize_dims(hyper_dims, name="hyper_dims"), int(output_dim)]
    current_dim = int(input_dim)
    layers = []
    for hidden_dim in hyper_stream:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.0))
        current_dim = hidden_dim
    return nn.Sequential(*layers)


class HAMURBaseModel(MSRModel):
    def __init__(self, manager: SchemaManager, num_domains: int):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field

    @staticmethod
    def _flatten_embedding(embedding: torch.Tensor) -> torch.Tensor:
        return torch.flatten(embedding, start_dim=1)

    @staticmethod
    def _clamp_probability(probabilities: torch.Tensor) -> torch.Tensor:
        return probabilities.clamp(min=1e-7, max=1.0 - 1e-7)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        probabilities = self.predict(interaction)
        return nn.functional.binary_cross_entropy(
            self._clamp_probability(probabilities),
            labels,
        )


class HAMURSingleInputModel(HAMURBaseModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            feature_fields=None,
    ):
        super().__init__(manager, num_domains)
        self.input_view = build_feature_view(
            self.omni_embedding,
            include_fields=feature_fields,
        )
        self.input_dim = self.input_view.embedding_dim

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return self._flatten_embedding(x), interaction[self.DOMAIN].long()

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)


class HAMURWideDeepModel(HAMURBaseModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            wide_fields=None,
            deep_fields=None,
    ):
        super().__init__(manager, num_domains)
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

    def encode_features(self, interaction):
        wide_x = self._flatten_embedding(self.wide_view(interaction))
        deep_x = self._flatten_embedding(self.deep_view(interaction))
        domain_ids = interaction[self.DOMAIN].long()
        return wide_x, deep_x, domain_ids

    def predict(self, interaction):
        wide_x, deep_x, domain_ids = self.encode_features(interaction)
        return self.forward(wide_x, deep_x, domain_ids)
