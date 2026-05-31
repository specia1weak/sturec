from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import roc_auc_score

from betterbole.core.train.context import TrainContext
from betterbole.emb import SchemaManager
from betterbole.experiment import WORKSPACE
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.msr.ple.layers import PLELayer, select_domain_output
from betterbole.models.utils.common import to_dims
from betterbole.models.utils.general import MLP
from betterbole.utils.observatory import PlotSeries, RelationOptions, TensorDisplayConfig, TensorMonitorOptions
from betterbole.utils.observatory.metrics import compute_spectrum
from betterbole.utils.observatory.plots import plot_multi_series, plot_ranked_profile, plot_step_dim_heatmap, plot_topk_bar


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(inputs: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversalFunction.apply(inputs, lambda_)


class PLEAdversarialEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_levels: int = 1,
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
        specific_hidden = select_domain_output(task_outputs=task_outputs, domain_ids=domain_ids)
        return specific_hidden, shared_output


class PLEAdversarialModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_levels: int = 1,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
            shared_aux_weight: float = 0.05,
            shared_domain_adv_weight: float = 0.005,
            shared_domain_adv_lambda: float = 0.2,
            specific_aux_weight: float = 0.0,
            specific_aux_hardness_power: float = 0.0,
            interaction_gain_weight: float = 0.0,
            interaction_gain_margin: float = 0.0,
            interaction_unique_weight: float = 0.0,
            interaction_gate_hardness_weight: float = 0.0,
            counterfactual_logit_margin_weight: float = 0.0,
            counterfactual_logit_margin: float = 0.1,
            counterfactual_noise_std: float = 0.05,
            specific_domain_weight: float = 0.01,
            domain_head_hidden_dims: Iterable[int] = (64,),
            domain_head_dropout_rate: float = 0.0,
            latent_hidden_dims: Iterable[int] = None,
            specific_feature_mode: str = "specific",
            interaction_fusion_mode: str = "add",
            interaction_gate_hidden_dims: Iterable[int] = (64,),
            interaction_gate_temperature: float = 1.0,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim
        self.shared_aux_weight = float(shared_aux_weight)
        self.shared_domain_adv_weight = float(shared_domain_adv_weight)
        self.shared_domain_adv_lambda = float(shared_domain_adv_lambda)
        self.specific_aux_weight = float(specific_aux_weight)
        self.specific_aux_hardness_power = float(specific_aux_hardness_power)
        self.interaction_gain_weight = float(interaction_gain_weight)
        self.interaction_gain_margin = float(interaction_gain_margin)
        self.interaction_unique_weight = float(interaction_unique_weight)
        self.interaction_gate_hardness_weight = float(interaction_gate_hardness_weight)
        self.counterfactual_logit_margin_weight = float(counterfactual_logit_margin_weight)
        self.counterfactual_logit_margin = float(counterfactual_logit_margin)
        self.counterfactual_noise_std = float(counterfactual_noise_std)
        self.specific_domain_weight = float(specific_domain_weight)
        self.interaction_fusion_mode = str(interaction_fusion_mode or "add").lower()
        valid_fusion_modes = {"add", "gated_add"}
        if self.interaction_fusion_mode not in valid_fusion_modes:
            raise ValueError(
                "interaction_fusion_mode must be one of %s, got %s"
                % (sorted(valid_fusion_modes), self.interaction_fusion_mode)
            )
        self.interaction_gate_temperature = float(interaction_gate_temperature)

        self.encoder = PLEAdversarialEncoder(
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
        output_dim = self.encoder.output_dim
        latent_dims = tuple(to_dims(latent_hidden_dims, (max(128, output_dim * 2), output_dim)))
        if latent_dims[-1] != output_dim:
            latent_dims = tuple(latent_dims) + (output_dim,)
        self.specific_feature_mode = str(specific_feature_mode or "specific").lower()
        valid_specific_modes = {"specific", "latent", "specific_latent", "latent_residual", "latent_residual_only"}
        if self.specific_feature_mode not in valid_specific_modes:
            raise ValueError(
                "specific_feature_mode must be one of %s, got %s"
                % (sorted(valid_specific_modes), self.specific_feature_mode)
            )
        self.latent_projector = MLP(
            self.input_dim,
            *latent_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.latent_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.specific_feature_fusion = None
        if self.specific_feature_mode in {"specific_latent", "latent_residual", "latent_residual_only"}:
            self.specific_feature_fusion = MLP(
                output_dim * 2,
                output_dim,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )
        self.shared_readout_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.shared_head = nn.Linear(output_dim, 1)
        self.interaction_gate = None
        if self.interaction_fusion_mode == "gated_add":
            gate_hidden_dims = to_dims(interaction_gate_hidden_dims, (64,))
            self.interaction_gate = MLP(
                output_dim * 3 + 2,
                *gate_hidden_dims,
                1,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )

        domain_hidden_dims = to_dims(domain_head_hidden_dims, (max(1, output_dim // 2),))
        self.shared_domain_discriminator = MLP(
            output_dim,
            *domain_hidden_dims,
            self.num_domains,
            dropout_rate=domain_head_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.specific_domain_classifier = MLP(
            output_dim,
            *domain_hidden_dims,
            self.num_domains,
            dropout_rate=domain_head_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

        self._latest_task_loss = torch.tensor(0.0)
        self._latest_shared_aux_loss = torch.tensor(0.0)
        self._latest_shared_domain_adv_loss = torch.tensor(0.0)
        self._latest_specific_aux_loss = torch.tensor(0.0)
        self._latest_interaction_gain_loss = torch.tensor(0.0)
        self._latest_interaction_unique_loss = torch.tensor(0.0)
        self._latest_interaction_gate_loss = torch.tensor(0.0)
        self._latest_counterfactual_logit_margin_loss = torch.tensor(0.0)
        self._latest_specific_domain_loss = torch.tensor(0.0)
        self._latest_debug = {}
        self._observatory_initialized = False
        self._observatory_steps = []
        self._observatory_scalar_history = {}
        self._eval_branch_cache = {
            "labels": [],
            "domain_ids": [],
            "shared_logits": [],
            "specific_logits": [],
            "fused_logits": [],
            "specific_gate": [],
        }
        self.apply_xavier_initialization()

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long().view(-1)

    def _build_specific_feature(
            self,
            specific_hidden_raw: torch.Tensor,
            latent_hidden: torch.Tensor,
            shared_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if self.specific_feature_mode == "specific":
            return specific_hidden_raw
        if self.specific_feature_mode == "latent":
            return latent_hidden
        if self.specific_feature_mode == "specific_latent":
            return self.specific_feature_fusion(torch.cat([specific_hidden_raw, latent_hidden], dim=-1))
        if self.specific_feature_mode == "latent_residual":
            residual_hidden = latent_hidden - shared_hidden.detach()
            return self.specific_feature_fusion(torch.cat([latent_hidden, residual_hidden], dim=-1))
        if self.specific_feature_mode == "latent_residual_only":
            return latent_hidden - shared_hidden.detach()
        raise RuntimeError("Unexpected specific_feature_mode=%s" % self.specific_feature_mode)

    def _forward_dict(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        specific_hidden_raw, shared_hidden_raw = self.encoder(x, domain_ids)
        shared_hidden = self.shared_readout_norm(shared_hidden_raw)
        latent_hidden = self.latent_norm(self.latent_projector(x))
        residual_hidden = None
        if self.specific_feature_mode in {"latent_residual", "latent_residual_only"}:
            residual_hidden = latent_hidden - shared_hidden.detach()
        specific_hidden = self._build_specific_feature(
            specific_hidden_raw=specific_hidden_raw,
            latent_hidden=latent_hidden,
            shared_hidden=shared_hidden,
        )
        decoded = self._decode_from_hiddens(shared_hidden, specific_hidden, domain_ids)
        return {
            "latent_hidden": latent_hidden,
            "residual_hidden": residual_hidden,
            "specific_hidden_raw": specific_hidden_raw,
            "specific_hidden": specific_hidden,
            "shared_hidden_raw": shared_hidden_raw,
            "shared_hidden": shared_hidden,
            **decoded,
        }

    def _balanced_ctr_bce(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
            sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_domain_losses = logits.new_zeros(self.num_domains)
        present_mask = torch.zeros(self.num_domains, dtype=torch.bool, device=logits.device)
        losses = []
        for domain_idx in range(self.num_domains):
            mask = domain_ids == domain_idx
            if not mask.any():
                continue
            per_sample_loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], reduction="none")
            if sample_weights is not None:
                weights = sample_weights[mask]
                domain_loss = (weights * per_sample_loss).sum() / weights.sum().clamp_min(1e-12)
            else:
                domain_loss = per_sample_loss.mean()
            per_domain_losses[domain_idx] = domain_loss
            present_mask[domain_idx] = True
            losses.append(domain_loss)
        if losses:
            return torch.stack(losses).mean(), per_domain_losses, present_mask
        return logits.new_zeros(()), per_domain_losses, present_mask

    def _balanced_domain_ce(
            self,
            domain_logits: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_domain_losses = domain_logits.new_zeros(self.num_domains)
        present_mask = torch.zeros(self.num_domains, dtype=torch.bool, device=domain_logits.device)
        losses = []
        for domain_idx in range(self.num_domains):
            mask = domain_ids == domain_idx
            if not mask.any():
                continue
            domain_loss = F.cross_entropy(domain_logits[mask], domain_ids[mask])
            per_domain_losses[domain_idx] = domain_loss
            present_mask[domain_idx] = True
            losses.append(domain_loss)
        if losses:
            return torch.stack(losses).mean(), per_domain_losses, present_mask
        return domain_logits.new_zeros(()), per_domain_losses, present_mask

    def _balanced_scalar_mean(
            self,
            values: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_domain_values = values.new_zeros(self.num_domains)
        present_mask = torch.zeros(self.num_domains, dtype=torch.bool, device=values.device)
        losses = []
        for domain_idx in range(self.num_domains):
            mask = domain_ids == domain_idx
            if not mask.any():
                continue
            domain_value = values[mask].mean()
            per_domain_values[domain_idx] = domain_value
            present_mask[domain_idx] = True
            losses.append(domain_value)
        if losses:
            return torch.stack(losses).mean(), per_domain_values, present_mask
        return values.new_zeros(()), per_domain_values, present_mask

    def _interaction_gain_loss(
            self,
            shared_logits: torch.Tensor,
            fused_logits: torch.Tensor,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_loss = F.binary_cross_entropy_with_logits(shared_logits.detach(), labels, reduction="none")
        fused_loss = F.binary_cross_entropy_with_logits(fused_logits, labels, reduction="none")
        margin_violation = torch.relu(fused_loss - shared_loss + self.interaction_gain_margin)
        return self._balanced_scalar_mean(margin_violation, domain_ids)

    @staticmethod
    def _centered_normalize(x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=0, keepdim=True)
        return F.normalize(centered, p=2, dim=-1)

    def _interaction_unique_loss(
            self,
            shared_hidden: torch.Tensor,
            specific_hidden: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_norm = self._centered_normalize(shared_hidden)
        specific_norm = self._centered_normalize(specific_hidden)
        sample_cos_sq = F.cosine_similarity(shared_norm, specific_norm, dim=-1).pow(2)
        return self._balanced_scalar_mean(sample_cos_sq, domain_ids)

    def _decode_from_hiddens(
            self,
            shared_hidden: torch.Tensor,
            specific_hidden: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        specific_logits = self.head(specific_hidden, domain_ids)
        shared_logits = self.shared_head(shared_hidden).squeeze(-1)
        specific_gate = shared_logits.new_ones(shared_logits.shape)
        specific_gate_logits = None
        if self.interaction_gate is not None:
            gate_inputs = torch.cat(
                [
                    shared_hidden,
                    specific_hidden,
                    (shared_hidden - specific_hidden).abs(),
                    shared_logits.unsqueeze(-1),
                    specific_logits.unsqueeze(-1),
                ],
                dim=-1,
            )
            specific_gate_logits = self.interaction_gate(gate_inputs).squeeze(-1)
            specific_gate = torch.sigmoid(specific_gate_logits / max(self.interaction_gate_temperature, 1e-6))
        fused_logits = shared_logits + specific_gate * specific_logits
        shared_domain_logits = self.shared_domain_discriminator(
            gradient_reversal(shared_hidden, self.shared_domain_adv_lambda)
        )
        specific_domain_logits = self.specific_domain_classifier(specific_hidden)
        return {
            "specific_logits": specific_logits,
            "shared_logits": shared_logits,
            "fused_logits": fused_logits,
            "specific_gate": specific_gate,
            "specific_gate_logits": specific_gate_logits,
            "shared_domain_logits": shared_domain_logits,
            "specific_domain_logits": specific_domain_logits,
        }

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_dict(x, domain_ids)["fused_logits"]

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_dict(x, domain_ids)
        if not self.training and self.LABEL in interaction:
            labels = interaction[self.LABEL].float().view(-1)
            self._eval_branch_cache["labels"].append(labels.detach().cpu())
            self._eval_branch_cache["domain_ids"].append(domain_ids.detach().cpu())
            self._eval_branch_cache["shared_logits"].append(forward_dict["shared_logits"].detach().cpu())
            self._eval_branch_cache["specific_logits"].append(forward_dict["specific_logits"].detach().cpu())
            self._eval_branch_cache["fused_logits"].append(forward_dict["fused_logits"].detach().cpu())
            self._eval_branch_cache["specific_gate"].append(forward_dict["specific_gate"].detach().cpu())
        return forward_dict["fused_logits"]

    def _clear_eval_branch_cache(self) -> None:
        for key in self._eval_branch_cache:
            self._eval_branch_cache[key].clear()

    @staticmethod
    def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        if labels.size == 0:
            return 0.0
        unique = np.unique(labels)
        if unique.size < 2:
            return 0.0
        return float(roc_auc_score(labels, scores))

    def _branch_eval_summary(self) -> Dict[str, float]:
        if not self._eval_branch_cache["labels"]:
            return {}

        labels = torch.cat(self._eval_branch_cache["labels"], dim=0).numpy()
        domain_ids = torch.cat(self._eval_branch_cache["domain_ids"], dim=0).numpy()
        shared_logits = torch.cat(self._eval_branch_cache["shared_logits"], dim=0).numpy()
        specific_logits = torch.cat(self._eval_branch_cache["specific_logits"], dim=0).numpy()
        fused_logits = torch.cat(self._eval_branch_cache["fused_logits"], dim=0).numpy()
        specific_gate = torch.cat(self._eval_branch_cache["specific_gate"], dim=0).numpy()

        summary = {
            "auc_shared": self._safe_auc(labels, shared_logits),
            "auc_specific": self._safe_auc(labels, specific_logits),
            "auc_fused": self._safe_auc(labels, fused_logits),
        }
        abs_shared = np.abs(shared_logits)
        abs_specific = np.abs(specific_logits)
        abs_total = abs_shared + abs_specific + 1e-12
        summary["mean_abs_shared_ratio"] = float((abs_shared / abs_total).mean())
        summary["mean_abs_specific_ratio"] = float((abs_specific / abs_total).mean())
        summary["logit_corr"] = float(np.corrcoef(shared_logits, specific_logits)[0, 1]) if labels.size >= 2 else 0.0
        summary["specific_gate_mean"] = float(specific_gate.mean())
        summary["specific_gate_std"] = float(specific_gate.std())

        for domain_idx in range(self.num_domains):
            mask = domain_ids == domain_idx
            if mask.sum() < 2:
                continue
            summary["domain%d_auc_shared" % domain_idx] = self._safe_auc(labels[mask], shared_logits[mask])
            summary["domain%d_auc_specific" % domain_idx] = self._safe_auc(labels[mask], specific_logits[mask])
            summary["domain%d_auc_fused" % domain_idx] = self._safe_auc(labels[mask], fused_logits[mask])
            summary["domain%d_specific_gate_mean" % domain_idx] = float(specific_gate[mask].mean())
        return summary

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_dict(x, domain_ids)

        task_loss = F.binary_cross_entropy_with_logits(forward_dict["fused_logits"], labels)
        shared_aux_loss, shared_domain_losses, shared_present_mask = self._balanced_ctr_bce(
            forward_dict["shared_logits"],
            labels,
            domain_ids,
        )
        specific_aux_weights = None
        if self.specific_aux_weight > 0.0 and self.specific_aux_hardness_power > 0.0:
            hardness = (labels - torch.sigmoid(forward_dict["shared_logits"].detach())).abs().clamp_min(1e-6)
            specific_aux_weights = hardness.pow(self.specific_aux_hardness_power)
        specific_aux_loss, specific_aux_domain_losses, specific_aux_present_mask = self._balanced_ctr_bce(
            forward_dict["specific_logits"],
            labels,
            domain_ids,
            sample_weights=specific_aux_weights,
        )
        interaction_gain_loss, interaction_gain_domain_losses, interaction_gain_present_mask = self._interaction_gain_loss(
            forward_dict["shared_logits"],
            forward_dict["fused_logits"],
            labels,
            domain_ids,
        )
        interaction_unique_loss, interaction_unique_domain_losses, interaction_unique_present_mask = self._interaction_unique_loss(
            forward_dict["shared_hidden"],
            forward_dict["specific_hidden"],
            domain_ids,
        )
        gate_target = (labels - torch.sigmoid(forward_dict["shared_logits"].detach())).abs()
        gate_alignment = F.binary_cross_entropy(
            forward_dict["specific_gate"].clamp(1e-6, 1.0 - 1e-6),
            gate_target,
            reduction="none",
        )
        interaction_gate_loss, interaction_gate_domain_losses, interaction_gate_present_mask = self._balanced_scalar_mean(
            gate_alignment,
            domain_ids,
        )
        counterfactual_logit_margin_loss = labels.new_zeros(())
        counterfactual_logit_gap = None
        counterfactual_specific_gate_gap = None
        counterfactual_logit_gap_domain_losses = labels.new_zeros(self.num_domains)
        counterfactual_logit_gap_present_mask = torch.zeros(self.num_domains, dtype=torch.bool, device=labels.device)
        if self.counterfactual_logit_margin_weight > 0.0:
            noisy_specific_hidden = forward_dict["specific_hidden"] + torch.randn_like(forward_dict["specific_hidden"]) * self.counterfactual_noise_std
            counterfactual_forward_dict = self._decode_from_hiddens(
                forward_dict["shared_hidden"],
                noisy_specific_hidden,
                domain_ids,
            )
            counterfactual_logit_gap = (forward_dict["fused_logits"] - counterfactual_forward_dict["fused_logits"]).abs()
            counterfactual_specific_gate_gap = (forward_dict["specific_gate"] - counterfactual_forward_dict["specific_gate"]).abs()
            counterfactual_margin_violation = torch.relu(self.counterfactual_logit_margin - counterfactual_logit_gap)
            counterfactual_logit_margin_loss, counterfactual_logit_gap_domain_losses, counterfactual_logit_gap_present_mask = self._balanced_scalar_mean(
                counterfactual_margin_violation,
                domain_ids,
            )
        shared_domain_adv_loss, shared_adv_domain_losses, shared_adv_present_mask = self._balanced_domain_ce(
            forward_dict["shared_domain_logits"],
            domain_ids,
        )
        specific_domain_loss, specific_domain_losses, specific_present_mask = self._balanced_domain_ce(
            forward_dict["specific_domain_logits"],
            domain_ids,
        )

        total_loss = (
            task_loss
            + self.shared_aux_weight * shared_aux_loss
            + self.specific_aux_weight * specific_aux_loss
            + self.interaction_gain_weight * interaction_gain_loss
            + self.interaction_unique_weight * interaction_unique_loss
            + self.interaction_gate_hardness_weight * interaction_gate_loss
            + self.counterfactual_logit_margin_weight * counterfactual_logit_margin_loss
            + self.shared_domain_adv_weight * shared_domain_adv_loss
            + self.specific_domain_weight * specific_domain_loss
        )

        self._latest_task_loss = task_loss.detach()
        self._latest_shared_aux_loss = shared_aux_loss.detach()
        self._latest_shared_domain_adv_loss = shared_domain_adv_loss.detach()
        self._latest_specific_aux_loss = specific_aux_loss.detach()
        self._latest_interaction_gain_loss = interaction_gain_loss.detach()
        self._latest_interaction_unique_loss = interaction_unique_loss.detach()
        self._latest_interaction_gate_loss = interaction_gate_loss.detach()
        self._latest_counterfactual_logit_margin_loss = counterfactual_logit_margin_loss.detach()
        self._latest_specific_domain_loss = specific_domain_loss.detach()
        self._latest_debug = {
            "latent_hidden": forward_dict["latent_hidden"].detach(),
            "residual_hidden": forward_dict["residual_hidden"].detach() if forward_dict["residual_hidden"] is not None else None,
            "specific_hidden_raw": forward_dict["specific_hidden_raw"].detach(),
            "specific_hidden": forward_dict["specific_hidden"].detach(),
            "shared_hidden_raw": forward_dict["shared_hidden_raw"].detach(),
            "shared_hidden": forward_dict["shared_hidden"].detach(),
            "specific_logits": forward_dict["specific_logits"].detach(),
            "shared_logits": forward_dict["shared_logits"].detach(),
            "fused_logits": forward_dict["fused_logits"].detach(),
            "specific_gate": forward_dict["specific_gate"].detach(),
            "specific_gate_logits": forward_dict["specific_gate_logits"].detach() if forward_dict["specific_gate_logits"] is not None else None,
            "shared_domain_logits": forward_dict["shared_domain_logits"].detach(),
            "specific_domain_logits": forward_dict["specific_domain_logits"].detach(),
            "labels": labels.detach(),
            "domain_ids": domain_ids.detach(),
            "shared_domain_losses": shared_domain_losses.detach(),
            "shared_present_mask": shared_present_mask.detach(),
            "specific_aux_domain_losses": specific_aux_domain_losses.detach(),
            "specific_aux_present_mask": specific_aux_present_mask.detach(),
            "interaction_gain_domain_losses": interaction_gain_domain_losses.detach(),
            "interaction_gain_present_mask": interaction_gain_present_mask.detach(),
            "interaction_unique_domain_losses": interaction_unique_domain_losses.detach(),
            "interaction_unique_present_mask": interaction_unique_present_mask.detach(),
            "interaction_gate_domain_losses": interaction_gate_domain_losses.detach(),
            "interaction_gate_present_mask": interaction_gate_present_mask.detach(),
            "counterfactual_logit_gap": counterfactual_logit_gap.detach() if counterfactual_logit_gap is not None else None,
            "counterfactual_specific_gate_gap": counterfactual_specific_gate_gap.detach() if counterfactual_specific_gate_gap is not None else None,
            "counterfactual_logit_gap_domain_losses": counterfactual_logit_gap_domain_losses.detach(),
            "counterfactual_logit_gap_present_mask": counterfactual_logit_gap_present_mask.detach(),
            "shared_adv_domain_losses": shared_adv_domain_losses.detach(),
            "shared_adv_present_mask": shared_adv_present_mask.detach(),
            "specific_domain_losses": specific_domain_losses.detach(),
            "specific_present_mask": specific_present_mask.detach(),
        }
        return total_loss

    def debug_state(self) -> Dict[str, float]:
        if not self._latest_debug:
            return {
                "task_loss": 0.0,
                "shared_aux_loss": 0.0,
                "shared_domain_adv_loss": 0.0,
                "specific_aux_loss": 0.0,
                "interaction_gain_loss": 0.0,
                "interaction_unique_loss": 0.0,
                "interaction_gate_loss": 0.0,
                "counterfactual_logit_margin_loss": 0.0,
                "specific_domain_loss": 0.0,
            }
        return {
            "task_loss": float(self._latest_task_loss.item()),
            "shared_aux_loss": float(self._latest_shared_aux_loss.item()),
            "shared_domain_adv_loss": float(self._latest_shared_domain_adv_loss.item()),
            "specific_aux_loss": float(self._latest_specific_aux_loss.item()),
            "interaction_gain_loss": float(self._latest_interaction_gain_loss.item()),
            "interaction_unique_loss": float(self._latest_interaction_unique_loss.item()),
            "interaction_gate_loss": float(self._latest_interaction_gate_loss.item()),
            "counterfactual_logit_margin_loss": float(self._latest_counterfactual_logit_margin_loss.item()),
            "specific_domain_loss": float(self._latest_specific_domain_loss.item()),
        }

    def _observatory_output_dir(self, ctx: TrainContext) -> Path:
        model_name = getattr(ctx.cfg, "model", None)
        output_name = str(model_name or getattr(ctx.cfg, "experiment_name", "ple_adv"))
        return WORKSPACE / ctx.cfg.dataset_name / "observatory" / output_name

    def _setup_observatory(self, recorder) -> None:
        if recorder is None or self._observatory_initialized:
            return
        if not hasattr(recorder, "register"):
            return

        vector_options = TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=12,
                topk_display_dims=8,
                rank_by="variance",
            )
        )
        scalar_options = TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=False,
                max_display_dims=1,
                topk_display_dims=1,
            )
        )
        for name in [
            "ple_adv_latent_hidden",
            "ple_adv_shared_hidden",
            "ple_adv_shared_hidden_raw",
            "ple_adv_specific_hidden",
            "ple_adv_specific_hidden_raw",
            "ple_adv_shared_domain_logits",
            "ple_adv_specific_domain_logits",
        ]:
            recorder.register(name, vector_options)
        for name in [
            "ple_adv_shared_logits",
            "ple_adv_specific_logits",
            "ple_adv_fused_logits",
            "ple_adv_specific_gate",
            "ple_adv_counterfactual_logit_gap",
            "ple_adv_counterfactual_gate_gap",
            "ple_adv_shared_domain_losses",
            "ple_adv_specific_aux_domain_losses",
            "ple_adv_interaction_gain_domain_losses",
            "ple_adv_interaction_unique_domain_losses",
            "ple_adv_interaction_gate_domain_losses",
            "ple_adv_shared_adv_domain_losses",
            "ple_adv_specific_domain_losses",
        ]:
            recorder.register(name, scalar_options)
        if hasattr(recorder, "configure_relations"):
            recorder.configure_relations(
                RelationOptions(
                    enabled=True,
                    rank=8,
                    max_pairs=8,
                    names=(
                        "ple_adv_shared_hidden",
                        "ple_adv_shared_hidden_raw",
                        "ple_adv_specific_hidden",
                    ),
                )
            )
        self._observatory_initialized = True

    @staticmethod
    def _topk_dim_var(feature_tensor: torch.Tensor, topk: int = 12):
        flat = feature_tensor.detach().float().reshape(feature_tensor.shape[0], -1)
        dim_var = flat.var(dim=0, unbiased=False).cpu()
        topk = min(int(topk), int(dim_var.numel()))
        if topk <= 0:
            return [], []
        top_indices = torch.topk(dim_var, k=topk).indices.sort().values
        top_values = dim_var[top_indices]
        return top_indices.tolist(), top_values.tolist()

    def _append_observatory_scalars(self, step: int) -> None:
        if not self._latest_debug:
            return
        if self._observatory_steps and self._observatory_steps[-1] == int(step):
            return

        shared_spectrum = compute_spectrum(self._latest_debug["shared_hidden"])
        specific_spectrum = compute_spectrum(self._latest_debug["specific_hidden"])
        contrib = self.contribution_state()

        shared_top1 = float((shared_spectrum.energy[:1].sum() / shared_spectrum.total_energy.clamp_min(1e-12)).item()) if shared_spectrum.energy.numel() > 0 else 0.0
        specific_top1 = float((specific_spectrum.energy[:1].sum() / specific_spectrum.total_energy.clamp_min(1e-12)).item()) if specific_spectrum.energy.numel() > 0 else 0.0

        scalar_values = {
            "shared_eff_rank": float(shared_spectrum.effective_rank),
            "specific_eff_rank": float(specific_spectrum.effective_rank),
            "shared_top1_energy": shared_top1,
            "specific_top1_energy": specific_top1,
            "shared_feature_var": float(contrib["shared_feature_var"]),
            "specific_feature_var": float(contrib["specific_feature_var"]),
            "shared_domain_acc": float(contrib["shared_domain_acc"]),
            "specific_domain_acc": float(contrib["specific_domain_acc"]),
            "shared_low_var_dim_ratio": float(contrib["shared_low_var_dim_ratio"]),
            "specific_low_var_dim_ratio": float(contrib["specific_low_var_dim_ratio"]),
            "shared_specific_align": float(contrib["shared_specific_align"]),
            "specific_aux_loss": float(self._latest_specific_aux_loss.item()),
            "interaction_gain_loss": float(self._latest_interaction_gain_loss.item()),
            "interaction_unique_loss": float(self._latest_interaction_unique_loss.item()),
            "interaction_gate_loss": float(self._latest_interaction_gate_loss.item()),
            "counterfactual_logit_margin_loss": float(self._latest_counterfactual_logit_margin_loss.item()),
            "counterfactual_gap_mean": float(contrib["counterfactual_gap_mean"]),
            "counterfactual_gate_gap_mean": float(contrib["counterfactual_gate_gap_mean"]),
            "interaction_gate_mean": float(contrib["interaction_gate_mean"]),
            "interaction_gate_gap": float(contrib["interaction_gate_gap"]),
            "specific_gate_mean": float(contrib["specific_gate_mean"]),
            "specific_gate_std": float(contrib["specific_gate_std"]),
            "gate_shared_hard_corr": float(contrib["gate_shared_hard_corr"]),
        }
        self._observatory_steps.append(int(step))
        for key, value in scalar_values.items():
            self._observatory_scalar_history.setdefault(key, []).append(float(value))

    def _export_observatory_artifacts(self, ctx: TrainContext) -> None:
        if not self._latest_debug:
            return
        recorder = ctx.recorder
        output_dir = self._observatory_output_dir(ctx)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._observatory_steps:
            plot_multi_series(
                [
                    PlotSeries("shared", self._observatory_steps, self._observatory_scalar_history.get("shared_eff_rank", [])),
                    PlotSeries("specific", self._observatory_steps, self._observatory_scalar_history.get("specific_eff_rank", [])),
                ],
                title="ple_adv_eff_rank_series",
                xlabel="step",
                ylabel="eff_rank",
                save_path=output_dir / "ple_adv_eff_rank_series.png",
            )
            plot_multi_series(
                [
                    PlotSeries("shared_top1", self._observatory_steps, self._observatory_scalar_history.get("shared_top1_energy", [])),
                    PlotSeries("specific_top1", self._observatory_steps, self._observatory_scalar_history.get("specific_top1_energy", [])),
                ],
                title="ple_adv_collapse_series",
                xlabel="step",
                ylabel="top1_energy",
                save_path=output_dir / "ple_adv_collapse_series.png",
            )
            plot_multi_series(
                [
                    PlotSeries("shared_var", self._observatory_steps, self._observatory_scalar_history.get("shared_feature_var", [])),
                    PlotSeries("specific_var", self._observatory_steps, self._observatory_scalar_history.get("specific_feature_var", [])),
                ],
                title="ple_adv_feature_var_series",
                xlabel="step",
                ylabel="feature_var",
                save_path=output_dir / "ple_adv_feature_var_series.png",
            )
            plot_multi_series(
                [
                    PlotSeries("shared_domain_acc", self._observatory_steps, self._observatory_scalar_history.get("shared_domain_acc", [])),
                    PlotSeries("specific_domain_acc", self._observatory_steps, self._observatory_scalar_history.get("specific_domain_acc", [])),
                    PlotSeries("align", self._observatory_steps, self._observatory_scalar_history.get("shared_specific_align", [])),
                ],
                title="ple_adv_domain_signal_series",
                xlabel="step",
                ylabel="value",
                save_path=output_dir / "ple_adv_domain_signal_series.png",
            )
            plot_multi_series(
                [
                    PlotSeries("gate_mean", self._observatory_steps, self._observatory_scalar_history.get("specific_gate_mean", [])),
                    PlotSeries("gate_std", self._observatory_steps, self._observatory_scalar_history.get("specific_gate_std", [])),
                    PlotSeries("gate_hard_corr", self._observatory_steps, self._observatory_scalar_history.get("gate_shared_hard_corr", [])),
                ],
                title="ple_adv_gate_series",
                xlabel="step",
                ylabel="value",
                save_path=output_dir / "ple_adv_gate_series.png",
            )
            plot_multi_series(
                [
                    PlotSeries("cf_gap", self._observatory_steps, self._observatory_scalar_history.get("counterfactual_gap_mean", [])),
                    PlotSeries("cf_gate_gap", self._observatory_steps, self._observatory_scalar_history.get("counterfactual_gate_gap_mean", [])),
                    PlotSeries("cf_margin_loss", self._observatory_steps, self._observatory_scalar_history.get("counterfactual_logit_margin_loss", [])),
                ],
                title="ple_adv_counterfactual_series",
                xlabel="step",
                ylabel="value",
                save_path=output_dir / "ple_adv_counterfactual_series.png",
            )

        feature_names = [
            ("latent_hidden", self._latest_debug["latent_hidden"]),
            ("shared_hidden", self._latest_debug["shared_hidden"]),
            ("specific_hidden", self._latest_debug["specific_hidden"]),
        ]
        for short_name, tensor in feature_names:
            dim_idx, top_values = self._topk_dim_var(tensor, topk=12)
            if dim_idx:
                plot_topk_bar(
                    dim_idx=dim_idx,
                    values=top_values,
                    title="ple_adv_%s_top_dim_var" % short_name,
                    ylabel="batch_var",
                    save_path=output_dir / ("ple_adv_%s_top_dim_var.png" % short_name),
                )
            spectrum = compute_spectrum(tensor)
            if spectrum.singular_values.numel() > 0:
                plot_ranked_profile(
                    values=spectrum.singular_values.detach().cpu().numpy(),
                    title="ple_adv_%s_singular_values" % short_name,
                    ylabel="singular_value",
                    save_path=output_dir / ("ple_adv_%s_singular_values.png" % short_name),
                    descending=True,
                )

        if recorder is not None:
            _, shared_step_dim = recorder.get_step_dim_matrix("ple_adv_shared_hidden", "batch_var")
            if shared_step_dim.size > 0:
                plot_step_dim_heatmap(
                    values=shared_step_dim,
                    title="ple_adv_shared_hidden_step_dim_var",
                    save_path=output_dir / "ple_adv_shared_hidden_step_dim_var.png",
                )
            _, specific_step_dim = recorder.get_step_dim_matrix("ple_adv_specific_hidden", "batch_var")
            if specific_step_dim.size > 0:
                plot_step_dim_heatmap(
                    values=specific_step_dim,
                    title="ple_adv_specific_hidden_step_dim_var",
                    save_path=output_dir / "ple_adv_specific_hidden_step_dim_var.png",
                )

    def _domain_loss_stats(self, losses: torch.Tensor, present_mask: torch.Tensor) -> Tuple[float, float]:
        if present_mask.any():
            active_losses = losses[present_mask]
            return float(active_losses.mean().item()), float((active_losses.max() - active_losses.min()).item())
        return 0.0, 0.0

    def _domain_accuracy_entropy(self, logits: torch.Tensor, domain_ids: torch.Tensor) -> Tuple[float, float]:
        if logits.numel() == 0:
            return 0.0, 0.0
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        acc = float((preds == domain_ids).float().mean().item())
        entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()).item())
        return acc, entropy

    def contribution_state(self) -> Dict[str, float]:
        if not self._latest_debug:
            return {
                "shared_feature_var": 0.0,
                "specific_feature_var": 0.0,
                "residual_feature_var": 0.0,
                "shared_logit_var": 0.0,
                "specific_logit_var": 0.0,
                "feature_var_shared_ratio": 0.0,
                "logit_var_shared_ratio": 0.0,
                "fused_shared_logit_ratio": 0.0,
                "shared_domain_loss_gap": 0.0,
                "shared_adv_loss_gap": 0.0,
                "interaction_gain_gap": 0.0,
                "interaction_unique_gap": 0.0,
                "interaction_gate_gap": 0.0,
                "interaction_gate_mean": 0.0,
                "counterfactual_gap_mean": 0.0,
                "counterfactual_gap_gap": 0.0,
                "counterfactual_gate_gap_mean": 0.0,
                "counterfactual_gate_gap_std": 0.0,
                "specific_domain_loss_gap": 0.0,
                "shared_domain_acc": 0.0,
                "shared_domain_entropy": 0.0,
                "specific_domain_acc": 0.0,
                "specific_domain_entropy": 0.0,
                "shared_specific_align": 0.0,
                "shared_specific_logit_corr": 0.0,
                "specific_gate_mean": 0.0,
                "specific_gate_std": 0.0,
                "specific_gate_low_ratio": 0.0,
                "specific_gate_high_ratio": 0.0,
                "gate_shared_hard_corr": 0.0,
                "gate_gain_corr": 0.0,
                "gain_violation_ratio": 0.0,
                "shared_low_var_dim_ratio": 0.0,
                "specific_low_var_dim_ratio": 0.0,
            }

        shared_hidden = self._latest_debug["shared_hidden"].float()
        specific_hidden = self._latest_debug["specific_hidden"].float()
        residual_hidden = self._latest_debug["residual_hidden"]
        residual_hidden = residual_hidden.float() if residual_hidden is not None else None
        shared_logits = self._latest_debug["shared_logits"].float()
        specific_logits = self._latest_debug["specific_logits"].float()
        fused_logits = self._latest_debug["fused_logits"].float()
        shared_domain_logits = self._latest_debug["shared_domain_logits"].float()
        specific_domain_logits = self._latest_debug["specific_domain_logits"].float()
        labels = self._latest_debug["labels"].float()
        domain_ids = self._latest_debug["domain_ids"].long()

        shared_feature_var = float(shared_hidden.var(unbiased=False).item())
        specific_feature_var = float(specific_hidden.var(unbiased=False).item())
        residual_feature_var = float(residual_hidden.var(unbiased=False).item()) if residual_hidden is not None else 0.0
        shared_logit_var = float(shared_logits.var(unbiased=False).item())
        specific_logit_var = float(specific_logits.var(unbiased=False).item())
        fused_logit_var = float(fused_logits.var(unbiased=False).item())
        shared_per_dim_var = shared_hidden.var(dim=0, unbiased=False)
        specific_per_dim_var = specific_hidden.var(dim=0, unbiased=False)
        feature_var_total = shared_feature_var + specific_feature_var + 1e-12
        logit_var_total = shared_logit_var + specific_logit_var + 1e-12

        shared_domain_loss_mean, shared_domain_loss_gap = self._domain_loss_stats(
            self._latest_debug["shared_domain_losses"].float(),
            self._latest_debug["shared_present_mask"],
        )
        shared_adv_loss_mean, shared_adv_loss_gap = self._domain_loss_stats(
            self._latest_debug["shared_adv_domain_losses"].float(),
            self._latest_debug["shared_adv_present_mask"],
        )
        specific_domain_loss_mean, specific_domain_loss_gap = self._domain_loss_stats(
            self._latest_debug["specific_domain_losses"].float(),
            self._latest_debug["specific_present_mask"],
        )
        interaction_gain_mean, interaction_gain_gap = self._domain_loss_stats(
            self._latest_debug["interaction_gain_domain_losses"].float(),
            self._latest_debug["interaction_gain_present_mask"],
        )
        interaction_unique_mean, interaction_unique_gap = self._domain_loss_stats(
            self._latest_debug["interaction_unique_domain_losses"].float(),
            self._latest_debug["interaction_unique_present_mask"],
        )
        interaction_gate_mean, interaction_gate_gap = self._domain_loss_stats(
            self._latest_debug["interaction_gate_domain_losses"].float(),
            self._latest_debug["interaction_gate_present_mask"],
        )
        counterfactual_gap_mean = 0.0
        counterfactual_gap_gap = 0.0
        counterfactual_gate_gap_mean = 0.0
        counterfactual_gate_gap_std = 0.0
        if self._latest_debug["counterfactual_logit_gap"] is not None:
            counterfactual_gap_mean, counterfactual_gap_gap = self._domain_loss_stats(
                self._latest_debug["counterfactual_logit_gap_domain_losses"].float(),
                self._latest_debug["counterfactual_logit_gap_present_mask"],
            )
            counterfactual_gate_gap = self._latest_debug["counterfactual_specific_gate_gap"]
            if counterfactual_gate_gap is not None:
                counterfactual_gate_gap = counterfactual_gate_gap.float()
                counterfactual_gate_gap_mean = float(counterfactual_gate_gap.mean().item())
                counterfactual_gate_gap_std = float(counterfactual_gate_gap.std(unbiased=False).item())
        shared_domain_acc, shared_domain_entropy = self._domain_accuracy_entropy(shared_domain_logits, domain_ids)
        specific_domain_acc, specific_domain_entropy = self._domain_accuracy_entropy(specific_domain_logits, domain_ids)
        specific_gate = self._latest_debug["specific_gate"].float()
        per_sample_shared_loss = F.binary_cross_entropy_with_logits(shared_logits, labels, reduction="none")
        per_sample_fused_loss = F.binary_cross_entropy_with_logits(fused_logits, labels, reduction="none")
        per_sample_gain = per_sample_shared_loss - per_sample_fused_loss
        specific_gate_centered = specific_gate - specific_gate.mean()
        shared_hardness_centered = per_sample_shared_loss - per_sample_shared_loss.mean()
        gain_centered = per_sample_gain - per_sample_gain.mean()
        gate_shared_hard_corr = 0.0
        gate_gain_corr = 0.0
        if specific_gate.numel() >= 2:
            gate_std = specific_gate_centered.std(unbiased=False).clamp_min(1e-12)
            shared_hard_std = shared_hardness_centered.std(unbiased=False).clamp_min(1e-12)
            gain_std = gain_centered.std(unbiased=False).clamp_min(1e-12)
            gate_shared_hard_corr = float((specific_gate_centered * shared_hardness_centered).mean().div(gate_std * shared_hard_std).item())
            gate_gain_corr = float((specific_gate_centered * gain_centered).mean().div(gate_std * gain_std).item())

        return {
            "shared_feature_var": shared_feature_var,
            "specific_feature_var": specific_feature_var,
            "residual_feature_var": residual_feature_var,
            "shared_logit_var": shared_logit_var,
            "specific_logit_var": specific_logit_var,
            "fused_logit_var": fused_logit_var,
            "feature_var_shared_ratio": shared_feature_var / feature_var_total,
            "logit_var_shared_ratio": shared_logit_var / logit_var_total,
            "fused_shared_logit_ratio": shared_logit_var / (fused_logit_var + 1e-12),
            "shared_domain_loss_mean": shared_domain_loss_mean,
            "shared_domain_loss_gap": shared_domain_loss_gap,
            "shared_adv_loss_mean": shared_adv_loss_mean,
            "shared_adv_loss_gap": shared_adv_loss_gap,
            "interaction_gain_mean": interaction_gain_mean,
            "interaction_gain_gap": interaction_gain_gap,
            "interaction_unique_mean": interaction_unique_mean,
            "interaction_unique_gap": interaction_unique_gap,
            "interaction_gate_mean": interaction_gate_mean,
            "interaction_gate_gap": interaction_gate_gap,
            "counterfactual_gap_mean": counterfactual_gap_mean,
            "counterfactual_gap_gap": counterfactual_gap_gap,
            "counterfactual_gate_gap_mean": counterfactual_gate_gap_mean,
            "counterfactual_gate_gap_std": counterfactual_gate_gap_std,
            "specific_domain_loss_mean": specific_domain_loss_mean,
            "specific_domain_loss_gap": specific_domain_loss_gap,
            "shared_domain_acc": shared_domain_acc,
            "shared_domain_entropy": shared_domain_entropy,
            "specific_domain_acc": specific_domain_acc,
            "specific_domain_entropy": specific_domain_entropy,
            "shared_specific_align": float(F.cosine_similarity(shared_hidden, specific_hidden, dim=-1).mean().item()),
            "shared_specific_logit_corr": float(
                torch.corrcoef(torch.stack([shared_logits, specific_logits], dim=0))[0, 1].item()
            ) if shared_logits.numel() >= 2 else 0.0,
            "specific_gate_mean": float(specific_gate.mean().item()),
            "specific_gate_std": float(specific_gate.std(unbiased=False).item()),
            "specific_gate_low_ratio": float((specific_gate < 0.1).float().mean().item()),
            "specific_gate_high_ratio": float((specific_gate > 0.9).float().mean().item()),
            "gate_shared_hard_corr": gate_shared_hard_corr,
            "gate_gain_corr": gate_gain_corr,
            "gain_violation_ratio": float(
                (per_sample_fused_loss > per_sample_shared_loss).float().mean().item()
            ),
            "shared_abs_mean": float(shared_hidden.abs().mean().item()),
            "specific_abs_mean": float(specific_hidden.abs().mean().item()),
            "residual_abs_mean": float(residual_hidden.abs().mean().item()) if residual_hidden is not None else 0.0,
            "shared_positive_ratio": float((shared_hidden > 0).float().mean().item()),
            "specific_positive_ratio": float((specific_hidden > 0).float().mean().item()),
            "residual_positive_ratio": float((residual_hidden > 0).float().mean().item()) if residual_hidden is not None else 0.0,
            "shared_low_var_dim_ratio": float((shared_per_dim_var < 1e-6).float().mean().item()),
            "specific_low_var_dim_ratio": float((specific_per_dim_var < 1e-6).float().mean().item()),
            "residual_low_var_dim_ratio": float((residual_hidden.var(dim=0, unbiased=False) < 1e-6).float().mean().item()) if residual_hidden is not None else 0.0,
        }

    def _record_debug_tensors(self, recorder, step: Optional[int] = None) -> None:
        if recorder is None or not self._latest_debug:
            return
        self._setup_observatory(recorder)
        recorder.record("ple_adv_latent_hidden", self._latest_debug["latent_hidden"], step=step)
        recorder.record("ple_adv_shared_hidden", self._latest_debug["shared_hidden"], step=step)
        recorder.record("ple_adv_shared_hidden_raw", self._latest_debug["shared_hidden_raw"], step=step)
        recorder.record("ple_adv_specific_hidden_raw", self._latest_debug["specific_hidden_raw"], step=step)
        recorder.record("ple_adv_specific_hidden", self._latest_debug["specific_hidden"], step=step)
        recorder.record("ple_adv_shared_logits", self._latest_debug["shared_logits"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_specific_logits", self._latest_debug["specific_logits"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_fused_logits", self._latest_debug["fused_logits"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_specific_gate", self._latest_debug["specific_gate"].unsqueeze(-1), step=step)
        if self._latest_debug["counterfactual_logit_gap"] is not None:
            recorder.record("ple_adv_counterfactual_logit_gap", self._latest_debug["counterfactual_logit_gap"].unsqueeze(-1), step=step)
        if self._latest_debug["counterfactual_specific_gate_gap"] is not None:
            recorder.record("ple_adv_counterfactual_gate_gap", self._latest_debug["counterfactual_specific_gate_gap"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_shared_domain_logits", self._latest_debug["shared_domain_logits"], step=step)
        recorder.record("ple_adv_specific_domain_logits", self._latest_debug["specific_domain_logits"], step=step)
        recorder.record("ple_adv_shared_domain_losses", self._latest_debug["shared_domain_losses"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_specific_aux_domain_losses", self._latest_debug["specific_aux_domain_losses"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_interaction_gain_domain_losses", self._latest_debug["interaction_gain_domain_losses"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_interaction_unique_domain_losses", self._latest_debug["interaction_unique_domain_losses"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_interaction_gate_domain_losses", self._latest_debug["interaction_gate_domain_losses"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_shared_adv_domain_losses", self._latest_debug["shared_adv_domain_losses"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_specific_domain_losses", self._latest_debug["specific_domain_losses"].unsqueeze(-1), step=step)

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        loss = self.calculate_loss(batch_interaction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        step = ctx.global_step + 1
        self._record_debug_tensors(ctx.recorder, step=step)
        self._append_observatory_scalars(step=step)
        if ctx.recorder is not None and step % 200 == 0:
            debug = self.debug_state()
            contrib = self.contribution_state()
            shared_spectrum = compute_spectrum(self._latest_debug["shared_hidden"])
            specific_spectrum = compute_spectrum(self._latest_debug["specific_hidden"])
            shared_top1 = float((shared_spectrum.energy[:1].sum() / shared_spectrum.total_energy.clamp_min(1e-12)).item()) if shared_spectrum.energy.numel() > 0 else 0.0
            specific_top1 = float((specific_spectrum.energy[:1].sum() / specific_spectrum.total_energy.clamp_min(1e-12)).item()) if specific_spectrum.energy.numel() > 0 else 0.0
            print(
                "[PLE-Adv Recorder] "
                f"step={step} "
                f"mode={self.specific_feature_mode} "
                f"task={debug['task_loss']:.5f} "
                f"shared_aux={debug['shared_aux_loss']:.5f} "
                f"specific_aux={debug['specific_aux_loss']:.5f} "
                f"gain={debug['interaction_gain_loss']:.5f} "
                f"unique={debug['interaction_unique_loss']:.5f} "
                f"gate_loss={debug['interaction_gate_loss']:.5f} "
                f"cf_margin={debug['counterfactual_logit_margin_loss']:.5f} "
                f"shared_adv={debug['shared_domain_adv_loss']:.5f} "
                f"specific_domain={debug['specific_domain_loss']:.5f} "
                f"rank(shared/specific)={shared_spectrum.effective_rank:.2f}/{specific_spectrum.effective_rank:.2f} "
                f"top1(shared/specific)={shared_top1:.3f}/{specific_top1:.3f} "
                f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
                f"logit_ratio={contrib['fused_shared_logit_ratio']:.3f} "
                f"align={contrib['shared_specific_align']:.3f} "
                f"logit_corr={contrib['shared_specific_logit_corr']:.3f} "
                f"gain_bad={contrib['gain_violation_ratio']:.3f} "
                f"cf(gap/gate_gap)={contrib['counterfactual_gap_mean']:.3f}/{contrib['counterfactual_gate_gap_mean']:.3f} "
                f"gate(mean/std/hard/gain)={contrib['specific_gate_mean']:.3f}/{contrib['specific_gate_std']:.3f}/{contrib['gate_shared_hard_corr']:.3f}/{contrib['gate_gain_corr']:.3f} "
                f"adv_acc={contrib['shared_domain_acc']:.3f} "
                f"spec_acc={contrib['specific_domain_acc']:.3f}"
            )
            print(
                ctx.recorder.get_window_stats(
                    names=[
                        "ple_adv_shared_hidden",
                        "ple_adv_specific_hidden",
                        "ple_adv_shared_domain_logits",
                        "ple_adv_specific_domain_logits",
                    ],
                    include_relations=True,
                    relation_names=[
                        "ple_adv_shared_hidden",
                        "ple_adv_shared_hidden_raw",
                        "ple_adv_specific_hidden",
                    ],
                )
            )
        return float(loss.item())

    def on_train_epoch_start(self, ctx: TrainContext) -> None:
        del ctx
        self._clear_eval_branch_cache()

    def on_train_epoch_end(self, ctx: TrainContext) -> None:
        del ctx
        self._clear_eval_branch_cache()

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx: TrainContext) -> None:
        del metrics
        debug = self.debug_state()
        contrib = self.contribution_state()
        shared_spectrum = compute_spectrum(self._latest_debug["shared_hidden"]) if self._latest_debug else compute_spectrum(torch.zeros(2, 2))
        specific_spectrum = compute_spectrum(self._latest_debug["specific_hidden"]) if self._latest_debug else compute_spectrum(torch.zeros(2, 2))
        shared_top1 = float((shared_spectrum.energy[:1].sum() / shared_spectrum.total_energy.clamp_min(1e-12)).item()) if shared_spectrum.energy.numel() > 0 else 0.0
        specific_top1 = float((specific_spectrum.energy[:1].sum() / specific_spectrum.total_energy.clamp_min(1e-12)).item()) if specific_spectrum.energy.numel() > 0 else 0.0
        print(
            "[PLE-Adv Debug] "
            f"mode={self.specific_feature_mode} "
            f"task={debug['task_loss']:.5f} "
            f"shared_aux={debug['shared_aux_loss']:.5f} "
            f"specific_aux={debug['specific_aux_loss']:.5f} "
            f"gain={debug['interaction_gain_loss']:.5f} "
            f"unique={debug['interaction_unique_loss']:.5f} "
            f"gate_loss={debug['interaction_gate_loss']:.5f} "
            f"cf_margin={debug['counterfactual_logit_margin_loss']:.5f} "
            f"shared_adv={debug['shared_domain_adv_loss']:.5f} "
            f"specific_domain={debug['specific_domain_loss']:.5f} "
            f"rank(shared/specific)={shared_spectrum.effective_rank:.2f}/{specific_spectrum.effective_rank:.2f} "
            f"top1(shared/specific)={shared_top1:.3f}/{specific_top1:.3f} "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_ratio={contrib['fused_shared_logit_ratio']:.3f} "
            f"align={contrib['shared_specific_align']:.3f} "
            f"logit_corr={contrib['shared_specific_logit_corr']:.3f} "
            f"gain_bad={contrib['gain_violation_ratio']:.3f} "
            f"cf(gap/gate_gap)={contrib['counterfactual_gap_mean']:.3f}/{contrib['counterfactual_gate_gap_mean']:.3f} "
            f"gate(mean/std/low/high/corr)={contrib['specific_gate_mean']:.3f}/{contrib['specific_gate_std']:.3f}/{contrib['specific_gate_low_ratio']:.3f}/{contrib['specific_gate_high_ratio']:.3f}/{contrib['gate_shared_hard_corr']:.3f} "
            f"shared_adv(acc/ent)={contrib['shared_domain_acc']:.3f}/{contrib['shared_domain_entropy']:.3f} "
            f"specific_probe(acc/ent)={contrib['specific_domain_acc']:.3f}/{contrib['specific_domain_entropy']:.3f} "
            f"loss_gap(shared/adv/gain/uniq/spec)={contrib['shared_domain_loss_gap']:.5f}/{contrib['shared_adv_loss_gap']:.5f}/{contrib['interaction_gain_gap']:.5f}/{contrib['interaction_unique_gap']:.5f}/{contrib['specific_domain_loss_gap']:.5f} "
            f"abs_mean(shared/specific)={contrib['shared_abs_mean']:.4f}/{contrib['specific_abs_mean']:.4f} "
            f"active(shared/specific)={contrib['shared_positive_ratio']:.3f}/{contrib['specific_positive_ratio']:.3f} "
            f"low_var(shared/specific)={contrib['shared_low_var_dim_ratio']:.3f}/{contrib['specific_low_var_dim_ratio']:.3f}"
        )
        if ctx.recorder is not None:
            print(
                ctx.recorder.get_window_stats(
                    names=[
                        "ple_adv_shared_hidden",
                        "ple_adv_specific_hidden",
                        "ple_adv_shared_domain_logits",
                        "ple_adv_specific_domain_logits",
                    ],
                    include_relations=True,
                    relation_names=[
                        "ple_adv_shared_hidden",
                        "ple_adv_shared_hidden_raw",
                        "ple_adv_specific_hidden",
                    ],
                )
            )
            self._export_observatory_artifacts(ctx)
        branch_summary = self._branch_eval_summary()
        if branch_summary:
            print(
                "[PLE-Adv Branch Eval] "
                f"auc(shared/specific/fused)={branch_summary.get('auc_shared', 0.0):.6f}/"
                f"{branch_summary.get('auc_specific', 0.0):.6f}/"
                f"{branch_summary.get('auc_fused', 0.0):.6f} "
                f"abs_ratio(shared/specific)={branch_summary.get('mean_abs_shared_ratio', 0.0):.3f}/"
                f"{branch_summary.get('mean_abs_specific_ratio', 0.0):.3f} "
                f"gate(mean/std)={branch_summary.get('specific_gate_mean', 0.0):.3f}/"
                f"{branch_summary.get('specific_gate_std', 0.0):.3f} "
                f"logit_corr={branch_summary.get('logit_corr', 0.0):.3f}"
            )
            for domain_idx in range(self.num_domains):
                shared_key = "domain%d_auc_shared" % domain_idx
                specific_key = "domain%d_auc_specific" % domain_idx
                fused_key = "domain%d_auc_fused" % domain_idx
                if shared_key in branch_summary:
                    print(
                        "[PLE-Adv Branch Domain] "
                        f"domain={domain_idx} "
                        f"auc(shared/specific/fused)="
                        f"{branch_summary[shared_key]:.6f}/"
                        f"{branch_summary[specific_key]:.6f}/"
                        f"{branch_summary[fused_key]:.6f} "
                        f"gate={branch_summary.get('domain%d_specific_gate_mean' % domain_idx, 0.0):.3f}"
                    )
        self._clear_eval_branch_cache()
