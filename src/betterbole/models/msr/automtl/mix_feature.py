from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from betterbole.models.msr.automtl.modules import GateFunc


class MixFeature(nn.Module):
    def __init__(self, num_fields: int, interaction_layer: nn.Module):
        super().__init__()
        self.num_fields = int(num_fields)
        self.interaction_layer = interaction_layer
        self.single_beta = nn.Parameter(torch.zeros(self.num_fields))
        self.inter_beta = nn.Parameter(torch.zeros(self.num_fields))
        self.selection_gate = GateFunc.apply
        self.in_warmup = True

    @property
    def module_str(self) -> str:
        single_probs = torch.sigmoid(self.single_beta).detach().cpu().numpy()
        inter_probs = torch.sigmoid(self.inter_beta).detach().cpu().numpy()
        return f"Single-fea select probs: {single_probs}, Inter-fea select probs: {inter_probs}"

    @property
    def chosen_indexes(self) -> tuple[list[int], list[int]]:
        single_probs = torch.sigmoid(self.single_beta).detach().cpu().numpy()
        inter_probs = torch.sigmoid(self.inter_beta).detach().cpu().numpy()
        return (
            np.where(single_probs > 0.6)[0].tolist(),
            np.where(inter_probs > 0.6)[0].tolist(),
        )

    def set_chosen_op_active(self) -> None:
        self.in_warmup = False

    def forward(self, sparse_embs: torch.Tensor, dense_features: Optional[torch.Tensor]) -> torch.Tensor:
        single_selection = self.selection_gate(
            torch.sigmoid(self.single_beta), self.in_warmup
        ).view(1, self.num_fields, 1)
        inter_selection = self.selection_gate(
            torch.sigmoid(self.inter_beta), self.in_warmup
        ).view(1, self.num_fields, 1)
        single_feature = (single_selection * sparse_embs).flatten(start_dim=1)
        inter_feature = self.interaction_layer(inter_selection * sparse_embs)
        if dense_features is None:
            return torch.cat((single_feature, inter_feature), dim=1)
        return torch.cat((single_feature, inter_feature, dense_features), dim=1)

    def export_arch(
            self,
            single_beta: Optional[list[float]] = None,
            inter_beta: Optional[list[float]] = None,
    ) -> tuple[list[float], list[float]]:
        if single_beta is None or inter_beta is None:
            self.single_beta.data = (torch.sigmoid(self.single_beta.data) > 0.5).float()
            self.inter_beta.data = (torch.sigmoid(self.inter_beta.data) > 0.5).float()
        else:
            self.single_beta.data = torch.tensor(single_beta, device=self.single_beta.device)
            self.inter_beta.data = torch.tensor(inter_beta, device=self.inter_beta.device)
        self.single_beta.requires_grad = False
        self.inter_beta.requires_grad = False
        self.in_warmup = False
        return self.single_beta.tolist(), self.inter_beta.tolist()
