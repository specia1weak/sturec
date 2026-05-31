from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.train.context import TrainContext
from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.msr.ple.layers import PLELayer, select_domain_output
from betterbole.models.utils.common import to_dims


class PLEBalancedV1Encoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = int(num_domains)
        self.output_dim = int(expert_dims[-1])
        self.layers = nn.ModuleList()

        layer_input_dim = int(input_dim)
        for _ in range(int(num_levels)):
            layer = PLELayer(
                input_dim=layer_input_dim,
                num_domains=num_domains,
                expert_dims=expert_dims,
                num_specific_experts=num_specific_experts,
                num_shared_experts=num_shared_experts,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            self.layers.append(layer)
            layer_input_dim = layer.output_dim

    def forward_all(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        task_inputs = [x for _ in range(self.num_domains)]
        shared_input = x
        for layer in self.layers:
            task_inputs, shared_input = layer(task_inputs, shared_input)
        return task_inputs, shared_input

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        task_outputs, shared_output = self.forward_all(x)
        return select_domain_output(task_outputs=task_outputs, domain_ids=domain_ids), shared_output


class PLEBalancedV1Model(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
            shared_aux_weight: float = 0.2,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim
        self.shared_aux_weight = float(shared_aux_weight)

        self.encoder = PLEBalancedV1Encoder(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_levels=num_levels,
            num_specific_experts=num_specific_experts,
            num_shared_experts=num_shared_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.encoder.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )
        self.shared_head = nn.Linear(self.encoder.output_dim, 1)

        self._latest_task_loss = torch.tensor(0.0)
        self._latest_shared_aux_loss = torch.tensor(0.0)
        self._latest_debug = {}
        self.apply_xavier_initialization()

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long().view(-1)

    def _forward_dict(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        specific_hidden, shared_hidden = self.encoder(x, domain_ids)
        specific_logits = self.head(specific_hidden, domain_ids)
        shared_logits = self.shared_head(shared_hidden).squeeze(-1)
        return {
            "specific_hidden": specific_hidden,
            "shared_hidden": shared_hidden,
            "specific_logits": specific_logits,
            "shared_logits": shared_logits,
        }

    def _balanced_shared_bce(
            self,
            shared_logits: torch.Tensor,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_domain_losses = shared_logits.new_zeros(self.num_domains)
        present_mask = torch.zeros(self.num_domains, dtype=torch.bool, device=shared_logits.device)
        loss_list = []
        for domain_idx in range(self.num_domains):
            mask = domain_ids == domain_idx
            if not mask.any():
                continue
            domain_loss = F.binary_cross_entropy_with_logits(shared_logits[mask], labels[mask])
            per_domain_losses[domain_idx] = domain_loss
            present_mask[domain_idx] = True
            loss_list.append(domain_loss)

        if loss_list:
            balanced_loss = torch.stack(loss_list).mean()
        else:
            balanced_loss = shared_logits.new_zeros(())
        return balanced_loss, per_domain_losses, present_mask

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_dict(x, domain_ids)["specific_logits"]

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_dict(x, domain_ids)

        task_loss = F.binary_cross_entropy_with_logits(forward_dict["specific_logits"], labels)
        shared_aux_loss, per_domain_losses, present_mask = self._balanced_shared_bce(
            forward_dict["shared_logits"],
            labels,
            domain_ids,
        )
        total_loss = task_loss + self.shared_aux_weight * shared_aux_loss

        self._latest_task_loss = task_loss.detach()
        self._latest_shared_aux_loss = shared_aux_loss.detach()
        self._latest_debug = {
            "specific_hidden": forward_dict["specific_hidden"].detach(),
            "shared_hidden": forward_dict["shared_hidden"].detach(),
            "specific_logits": forward_dict["specific_logits"].detach(),
            "shared_logits": forward_dict["shared_logits"].detach(),
            "shared_domain_losses": per_domain_losses.detach(),
            "shared_present_mask": present_mask.detach(),
        }
        return total_loss

    def debug_state(self) -> Dict[str, float]:
        if not self._latest_debug:
            return {
                "task_loss": 0.0,
                "shared_aux_loss": 0.0,
                "shared_present_domains": 0.0,
            }
        present_mask = self._latest_debug["shared_present_mask"]
        return {
            "task_loss": float(self._latest_task_loss.item()),
            "shared_aux_loss": float(self._latest_shared_aux_loss.item()),
            "shared_present_domains": float(present_mask.sum().item()),
        }

    def contribution_state(self) -> Dict[str, float]:
        if not self._latest_debug:
            return {
                "shared_feature_var": 0.0,
                "specific_feature_var": 0.0,
                "shared_logit_var": 0.0,
                "specific_logit_var": 0.0,
                "feature_var_shared_ratio": 0.0,
                "logit_var_shared_ratio": 0.0,
                "shared_domain_loss_gap": 0.0,
                "shared_domain_loss_mean": 0.0,
                "shared_abs_mean": 0.0,
                "specific_abs_mean": 0.0,
                "shared_positive_ratio": 0.0,
                "specific_positive_ratio": 0.0,
                "shared_low_var_dim_ratio": 0.0,
                "specific_low_var_dim_ratio": 0.0,
            }

        shared_hidden = self._latest_debug["shared_hidden"].float()
        specific_hidden = self._latest_debug["specific_hidden"].float()
        shared_logits = self._latest_debug["shared_logits"].float()
        specific_logits = self._latest_debug["specific_logits"].float()
        per_domain_losses = self._latest_debug["shared_domain_losses"].float()
        present_mask = self._latest_debug["shared_present_mask"]

        shared_feature_var = float(shared_hidden.var(unbiased=False).item())
        specific_feature_var = float(specific_hidden.var(unbiased=False).item())
        shared_logit_var = float(shared_logits.var(unbiased=False).item())
        specific_logit_var = float(specific_logits.var(unbiased=False).item())
        feature_var_total = shared_feature_var + specific_feature_var + 1e-12
        logit_var_total = shared_logit_var + specific_logit_var + 1e-12
        shared_per_dim_var = shared_hidden.var(dim=0, unbiased=False)
        specific_per_dim_var = specific_hidden.var(dim=0, unbiased=False)

        if present_mask.any():
            active_losses = per_domain_losses[present_mask]
            shared_domain_loss_gap = float((active_losses.max() - active_losses.min()).item())
            shared_domain_loss_mean = float(active_losses.mean().item())
        else:
            shared_domain_loss_gap = 0.0
            shared_domain_loss_mean = 0.0

        return {
            "shared_feature_var": shared_feature_var,
            "specific_feature_var": specific_feature_var,
            "shared_logit_var": shared_logit_var,
            "specific_logit_var": specific_logit_var,
            "feature_var_shared_ratio": shared_feature_var / feature_var_total,
            "logit_var_shared_ratio": shared_logit_var / logit_var_total,
            "shared_domain_loss_gap": shared_domain_loss_gap,
            "shared_domain_loss_mean": shared_domain_loss_mean,
            "shared_abs_mean": float(shared_hidden.abs().mean().item()),
            "specific_abs_mean": float(specific_hidden.abs().mean().item()),
            "shared_positive_ratio": float((shared_hidden > 0).float().mean().item()),
            "specific_positive_ratio": float((specific_hidden > 0).float().mean().item()),
            "shared_low_var_dim_ratio": float((shared_per_dim_var < 1e-6).float().mean().item()),
            "specific_low_var_dim_ratio": float((specific_per_dim_var < 1e-6).float().mean().item()),
        }

    def _record_debug_tensors(self, recorder) -> None:
        if recorder is None or not self._latest_debug:
            return

        recorder.record("ple_balanced_v1_shared_hidden", self._latest_debug["shared_hidden"])
        recorder.record("ple_balanced_v1_specific_hidden", self._latest_debug["specific_hidden"])
        recorder.record("ple_balanced_v1_shared_logits", self._latest_debug["shared_logits"].unsqueeze(-1))
        recorder.record("ple_balanced_v1_specific_logits", self._latest_debug["specific_logits"].unsqueeze(-1))
        recorder.record("ple_balanced_v1_shared_domain_losses", self._latest_debug["shared_domain_losses"].unsqueeze(-1))

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        loss = self.calculate_loss(batch_interaction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        self._record_debug_tensors(ctx.recorder)
        return float(loss.item())

    def on_train_epoch_start(self, ctx: TrainContext) -> None:
        del ctx

    def on_train_epoch_end(self, ctx: TrainContext) -> None:
        del ctx

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx: TrainContext) -> None:
        del metrics, ctx
        debug = self.debug_state()
        contrib = self.contribution_state()
        print(
            "[PLE-Balanced-v1 Debug] "
            f"task={debug['task_loss']:.5f} "
            f"shared_aux={debug['shared_aux_loss']:.5f} "
            f"present={int(debug['shared_present_domains'])}/{self.num_domains} "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_ratio={contrib['logit_var_shared_ratio']:.3f} "
            f"shared_gap={contrib['shared_domain_loss_gap']:.5f} "
            f"abs_mean(shared/specific)={contrib['shared_abs_mean']:.4f}/{contrib['specific_abs_mean']:.4f} "
            f"active(shared/specific)={contrib['shared_positive_ratio']:.3f}/{contrib['specific_positive_ratio']:.3f} "
            f"low_var(shared/specific)={contrib['shared_low_var_dim_ratio']:.3f}/{contrib['specific_low_var_dim_ratio']:.3f}"
        )
