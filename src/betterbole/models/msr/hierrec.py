from typing import Iterable

import torch
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
"""
@inproceedings{gao2024hierrec,
  title={HierRec: Scenario-Aware Hierarchical Modeling for Multi-scenario Recommendations},
  author={Gao, Jingtong and Chen, Bo and Zhu, Menghui and Zhao, Xiangyu and Li, Xiaopeng and Wang, Yuhao and Wang, Yichao and Guo, Huifeng and Tang, Ruiming},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={653--662},
  year={2024}
}
"""

class HierRecMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float,
                 out_softmax: bool = False, output_layer: bool = False):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        if out_softmax:
            layers.append(nn.Softmax(dim=1))
        if output_layer:
            layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class HierRecImplicitGenerator(nn.Module):
    def __init__(self, im_num: int, embed_dim: int, num_fields: int, inout_dim, dropout: float = 0.2):
        super().__init__()
        self.im_num = im_num
        self.embed_dim = embed_dim
        self.num_fields = num_fields
        self.emb_all_dim = embed_dim * num_fields
        self.inout_dim = inout_dim

        self.embed_atten = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,
            batch_first=True,
            dropout=dropout,
        )
        self.dnn11 = nn.Linear(self.emb_all_dim, inout_dim[0] * inout_dim[1])
        self.dnn12 = nn.Linear(self.emb_all_dim, inout_dim[1] * inout_dim[2])
        self.bg11 = nn.Linear(self.emb_all_dim, inout_dim[1])
        self.bg12 = nn.Linear(self.emb_all_dim, inout_dim[2])

    def forward(self, emb_x: torch.Tensor, im_para: torch.Tensor):
        batch_size = emb_x.size(0)
        mixed_emb = torch.mul(emb_x.unsqueeze(1), im_para).reshape(-1, self.num_fields, self.embed_dim)
        mixed_emb, _ = self.embed_atten(mixed_emb, mixed_emb, mixed_emb)
        mixed_emb = mixed_emb.contiguous().view(batch_size, self.im_num, self.emb_all_dim)

        net1 = self.dnn11(mixed_emb).view(batch_size, self.im_num, self.inout_dim[0], self.inout_dim[1])
        net2 = self.dnn12(mixed_emb).view(batch_size, self.im_num, self.inout_dim[1], self.inout_dim[2])
        net1_bias = self.bg11(mixed_emb).unsqueeze(2)
        net2_bias = self.bg12(mixed_emb).unsqueeze(2)
        return net1, net2, net1_bias, net2_bias


class HierRecExplicitGenerator(nn.Module):
    def __init__(self, embed_dim: int, num_fields: int, inout_dim, im_num: int,
                 dropout: float = 0.2):
        super().__init__()
        if embed_dim % im_num != 0:
            raise ValueError("'embed_dim % im_num' must be an integer")

        self.dnn11 = nn.Linear(embed_dim, inout_dim[0] * inout_dim[1])
        self.dnn12 = nn.Linear(embed_dim, inout_dim[1] * inout_dim[2])
        self.bg11 = nn.Linear(embed_dim, inout_dim[1])
        self.bg12 = nn.Linear(embed_dim, inout_dim[2])

        self.atten_part = embed_dim // im_num
        self.dnn2 = nn.ModuleList(
            [nn.Linear(self.atten_part, num_fields) for _ in range(im_num)]
        )
        self.norm_scale = nn.Softmax(dim=-1)

        self.inout_dim = inout_dim

    def forward(self, domain_emb: torch.Tensor):
        net1 = self.dnn11(domain_emb).view(-1, self.inout_dim[0], self.inout_dim[1])
        net2 = self.dnn12(domain_emb).view(-1, self.inout_dim[1], self.inout_dim[2])
        net1_bias = self.bg11(domain_emb).unsqueeze(1)
        net2_bias = self.bg12(domain_emb).unsqueeze(1)

        para2 = []
        for i, layer in enumerate(self.dnn2):
            start = i * self.atten_part
            end = (i + 1) * self.atten_part
            para2.append(layer(domain_emb[:, start:end]).unsqueeze(1))
        para2 = torch.cat(para2, dim=1)
        para2 = self.norm_scale(para2).unsqueeze(-1)
        return net1, net2, net1_bias, net2_bias, para2


class HierRec(MSRModel):
    def __init__(self, manager: SchemaManager, num_domains: int,
                 embed_dim: int = 32,
                 hidden_dims: Iterable[int] = (64, 16, 64, 64),
                 im_num: int = 4,
                 dropout: float = 0.2):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.embed_dim = embed_dim
        self.im_num = im_num
        self.hidden_dims = tuple(hidden_dims)
        if len(self.hidden_dims) != 4:
            raise ValueError(
                "HierRec hidden_dims must contain exactly 4 values: "
                "(dnn1_dim, shared_bottleneck_dim, shared_output_dim, head_hidden_dim)."
            )
        dnn1_dim, shared_bottleneck_dim, shared_output_dim, head_hidden_dim = self.hidden_dims
        self.shared_output_dim = shared_output_dim

        self.domain_fields = set(manager.domain_fields)
        self.feature_settings = [
            setting for setting in manager.settings
            if setting.field_name not in self.domain_fields
        ]
        if not self.feature_settings:
            raise ValueError("HierRec requires at least one non-domain feature.")
        invalid_settings = [
            setting.field_name for setting in self.feature_settings
            if setting.embedding_dim != embed_dim
        ]
        if invalid_settings:
            raise ValueError(
                "HierRec requires all non-domain feature embedding dims to equal "
                f"embed_dim={embed_dim}, but got mismatched fields: {invalid_settings}"
            )
        if self.omni_embedding.domain_id.embedding_dim != embed_dim:
            raise ValueError(
                "HierRec requires domain_id embedding dim to equal "
                f"embed_dim={embed_dim}, but got {self.omni_embedding.domain_id.embedding_dim}"
            )

        self.num_fields = len(self.feature_settings)
        self.emb_all_dim = embed_dim * self.num_fields

        # Keep a flat config surface while mapping to the original HierRec stage semantics.
        self.dnn1 = HierRecMLP(self.emb_all_dim, [dnn1_dim], dropout=dropout)
        self.explicit = HierRecExplicitGenerator(
            embed_dim=embed_dim,
            num_fields=self.num_fields,
            inout_dim=[dnn1_dim, shared_bottleneck_dim, shared_output_dim],
            im_num=im_num,
            dropout=dropout,
        )
        self.implicit = HierRecImplicitGenerator(
            im_num=im_num,
            embed_dim=embed_dim,
            num_fields=self.num_fields,
            inout_dim=[shared_output_dim, shared_bottleneck_dim, shared_output_dim],
            dropout=dropout,
        )
        self.out_trans = HierRecMLP(im_num * shared_output_dim, [shared_output_dim], dropout=dropout)
        self.dnn3 = HierRecMLP(shared_output_dim, [head_hidden_dim], dropout=dropout, output_layer=True)

    def concat_embed_input_fields(self, interaction) -> torch.Tensor:
        named_embs = self.omni_embedding.whole_without_domain(interaction, split_by="name")
        stacked = []
        for setting in self.feature_settings:
            field_emb = named_embs[setting.field_name]
            stacked.append(field_emb.unsqueeze(1))
        return torch.cat(stacked, dim=1)

    def get_domain_embedding(self, interaction) -> torch.Tensor:
        domain_emb = self.omni_embedding.domain_id(interaction)
        return domain_emb.reshape(domain_emb.size(0), -1)

    def forward(self, emb_x: torch.Tensor, domain_emb: torch.Tensor) -> torch.Tensor:
        explicit_net_1, explicit_net_2, explicit_bias_1, explicit_bias_2, implicit_para = self.explicit(domain_emb)
        implicit_net_1, implicit_net_2, implicit_bias_1, implicit_bias_2 = self.implicit(emb_x, implicit_para)

        x = self.dnn1(emb_x.reshape(emb_x.size(0), self.emb_all_dim))
        x = torch.matmul(x.unsqueeze(1), explicit_net_1) + explicit_bias_1
        x = torch.matmul(x, explicit_net_2) + explicit_bias_2
        x = torch.matmul(x.unsqueeze(1), implicit_net_1) + implicit_bias_1
        x = torch.matmul(x, implicit_net_2) + implicit_bias_2
        x = x.squeeze(-2).reshape(emb_x.size(0), self.im_num * self.shared_output_dim)
        x = self.out_trans(x)
        return self.dnn3(x).squeeze(-1)

    def predict(self, interaction) -> torch.Tensor:
        emb_x = self.concat_embed_input_fields(interaction)
        domain_emb = self.get_domain_embedding(interaction)
        return self.forward(emb_x, domain_emb)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)
