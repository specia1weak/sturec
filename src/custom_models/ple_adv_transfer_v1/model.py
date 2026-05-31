from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.models.msr.components import DomainTowerHead
from custom_models.ple_adv.model import gradient_reversal, PLEAdversarialEncoder, PLEAdversarialModel
from betterbole.models.utils.common import to_dims
from betterbole.models.utils.container import MultiScenarioContainer
from betterbole.models.utils.general import MLP


class TransferExpertPool(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_experts: int,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            gate_hidden_dims: Iterable[int] = (64,),
            gate_temperature: float = 1.0,
            topk: int = 0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.topk = max(0, int(topk))
        self.experts = nn.ModuleList([
            MLP(
                self.input_dim,
                self.hidden_dim,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_experts)
        ])
        gate_hidden_dims = to_dims(gate_hidden_dims, (64,))
        self.router = MLP(
            self.input_dim,
            *gate_hidden_dims,
            self.num_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logits = self.router(x)
        gate_weights = torch.softmax(gate_logits / self.gate_temperature, dim=-1)
        if self.topk > 0 and self.topk < self.num_experts:
            top_values, top_indices = torch.topk(gate_weights, k=self.topk, dim=-1)
            sparse_weights = torch.zeros_like(gate_weights)
            sparse_weights.scatter_(1, top_indices, top_values)
            gate_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        pooled = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return pooled, gate_weights, expert_outputs


class PLEAdversarialTransferV1Model(PLEAdversarialModel):
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
            transfer_num_experts: int = 4,
            transfer_topk: int = 2,
            transfer_gate_hidden_dims: Iterable[int] = (64,),
            transfer_gate_temperature: float = 1.0,
            transfer_aux_weight: float = 0.05,
            private_aux_weight: float = 0.02,
            private_residual_scale: float = 1.0,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_levels=num_levels,
            num_specific_experts=num_specific_experts,
            num_shared_experts=num_shared_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            shared_aux_weight=shared_aux_weight,
            shared_domain_adv_weight=shared_domain_adv_weight,
            shared_domain_adv_lambda=shared_domain_adv_lambda,
            specific_aux_weight=specific_aux_weight,
            specific_aux_hardness_power=specific_aux_hardness_power,
            interaction_gain_weight=interaction_gain_weight,
            interaction_gain_margin=interaction_gain_margin,
            interaction_unique_weight=interaction_unique_weight,
            interaction_gate_hardness_weight=interaction_gate_hardness_weight,
            counterfactual_logit_margin_weight=counterfactual_logit_margin_weight,
            counterfactual_logit_margin=counterfactual_logit_margin,
            counterfactual_noise_std=counterfactual_noise_std,
            specific_domain_weight=specific_domain_weight,
            domain_head_hidden_dims=domain_head_hidden_dims,
            domain_head_dropout_rate=domain_head_dropout_rate,
            latent_hidden_dims=latent_hidden_dims,
            specific_feature_mode=specific_feature_mode,
            interaction_fusion_mode=interaction_fusion_mode,
            interaction_gate_hidden_dims=interaction_gate_hidden_dims,
            interaction_gate_temperature=interaction_gate_temperature,
        )
        self.input_view = self.omni_embedding.whole_without_domain
        self.input_dim = self.input_view.embedding_dim
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
        self.latent_projector = MLP(
            self.input_dim,
            *latent_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.latent_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.transfer_aux_weight = float(transfer_aux_weight)
        self.private_aux_weight = float(private_aux_weight)
        self.private_residual_scale = float(private_residual_scale)
        self.transfer_pool = TransferExpertPool(
            input_dim=output_dim,
            hidden_dim=output_dim,
            num_experts=transfer_num_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            gate_hidden_dims=transfer_gate_hidden_dims,
            gate_temperature=transfer_gate_temperature,
            topk=transfer_topk,
        )
        self.transfer_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.private_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.specific_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.private_experts = MultiScenarioContainer(
            num_domains=num_domains,
            network_factory=lambda: MLP(
                output_dim * 2,
                output_dim,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            ),
        )
        self.transfer_head = nn.Linear(output_dim, 1)
        self.private_head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )
        self.apply_xavier_initialization()

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long().view(-1)

    def _forward_dict(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        specific_hidden_raw, shared_hidden_raw = self.encoder(x, domain_ids)
        shared_hidden = self.shared_readout_norm(shared_hidden_raw)
        common_hidden = shared_hidden.detach()
        latent_hidden = self.latent_norm(self.latent_projector(x))
        residual_hidden = latent_hidden - common_hidden

        transfer_hidden_raw, transfer_gate_weights, transfer_all_hidden = self.transfer_pool(common_hidden)
        transfer_hidden = self.transfer_norm(transfer_hidden_raw)

        private_input = torch.cat([common_hidden, self.private_residual_scale * residual_hidden], dim=-1)
        private_hidden_raw = self.private_experts(private_input, domain_ids)
        private_hidden = self.private_norm(private_hidden_raw)

        specific_hidden = self.specific_norm(transfer_hidden + private_hidden)

        shared_logits = self.shared_head(shared_hidden).squeeze(-1)
        transfer_logits = self.transfer_head(transfer_hidden).squeeze(-1)
        private_logits = self.private_head(private_hidden, domain_ids)
        specific_logits = transfer_logits + private_logits

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
        specific_domain_logits = self.specific_domain_classifier(private_hidden)
        return {
            "latent_hidden": latent_hidden,
            "residual_hidden": residual_hidden,
            "specific_hidden_raw": specific_hidden_raw,
            "shared_hidden_raw": shared_hidden_raw,
            "shared_hidden": shared_hidden,
            "common_hidden": common_hidden,
            "transfer_hidden_raw": transfer_hidden_raw,
            "transfer_hidden": transfer_hidden,
            "transfer_gate_weights": transfer_gate_weights,
            "transfer_all_hidden": transfer_all_hidden,
            "private_input": private_input,
            "private_hidden_raw": private_hidden_raw,
            "private_hidden": private_hidden,
            "specific_hidden": specific_hidden,
            "shared_logits": shared_logits,
            "transfer_logits": transfer_logits,
            "private_logits": private_logits,
            "specific_logits": specific_logits,
            "fused_logits": fused_logits,
            "specific_gate": specific_gate,
            "specific_gate_logits": specific_gate_logits,
            "shared_domain_logits": shared_domain_logits,
            "specific_domain_logits": specific_domain_logits,
        }

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
        transfer_aux_loss = F.binary_cross_entropy_with_logits(forward_dict["transfer_logits"], labels)
        private_aux_loss, private_aux_domain_losses, private_aux_present_mask = self._balanced_ctr_bce(
            forward_dict["private_logits"],
            labels,
            domain_ids,
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
            perm = torch.randperm(labels.size(0), device=labels.device)
            swapped_private_input = forward_dict["private_input"][perm]
            swapped_private_hidden = self.private_norm(self.private_experts(swapped_private_input, domain_ids))
            swapped_private_logits = self.private_head(swapped_private_hidden, domain_ids)
            counterfactual_specific_logits = forward_dict["transfer_logits"] + swapped_private_logits
            counterfactual_fused_logits = forward_dict["shared_logits"] + forward_dict["specific_gate"].detach() * counterfactual_specific_logits
            counterfactual_logit_gap = (forward_dict["fused_logits"] - counterfactual_fused_logits).abs()
            counterfactual_specific_gate_gap = torch.zeros_like(counterfactual_logit_gap)
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
            + self.transfer_aux_weight * transfer_aux_loss
            + self.private_aux_weight * private_aux_loss
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
            "residual_hidden": forward_dict["residual_hidden"].detach(),
            "specific_hidden_raw": forward_dict["specific_hidden_raw"].detach(),
            "specific_hidden": forward_dict["specific_hidden"].detach(),
            "shared_hidden_raw": forward_dict["shared_hidden_raw"].detach(),
            "shared_hidden": forward_dict["shared_hidden"].detach(),
            "common_hidden": forward_dict["common_hidden"].detach(),
            "transfer_hidden_raw": forward_dict["transfer_hidden_raw"].detach(),
            "transfer_hidden": forward_dict["transfer_hidden"].detach(),
            "transfer_gate_weights": forward_dict["transfer_gate_weights"].detach(),
            "private_hidden_raw": forward_dict["private_hidden_raw"].detach(),
            "private_hidden": forward_dict["private_hidden"].detach(),
            "private_input": forward_dict["private_input"].detach(),
            "transfer_logits": forward_dict["transfer_logits"].detach(),
            "private_logits": forward_dict["private_logits"].detach(),
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
            "private_aux_domain_losses": private_aux_domain_losses.detach(),
            "private_aux_present_mask": private_aux_present_mask.detach(),
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
        debug = super().debug_state()
        debug.update(
            {
                "transfer_aux_weight": float(self.transfer_aux_weight),
                "private_aux_weight": float(self.private_aux_weight),
            }
        )
        return debug

    def contribution_state(self) -> Dict[str, float]:
        stats = super().contribution_state()
        if not self._latest_debug:
            stats.update(
                {
                    "transfer_feature_var": 0.0,
                    "private_feature_var": 0.0,
                    "transfer_private_align": 0.0,
                    "transfer_logit_var": 0.0,
                    "private_logit_var": 0.0,
                    "transfer_gate_entropy": 0.0,
                    "transfer_top1_mean": 0.0,
                }
            )
            return stats
        transfer_hidden = self._latest_debug["transfer_hidden"].float()
        private_hidden = self._latest_debug["private_hidden"].float()
        transfer_logits = self._latest_debug["transfer_logits"].float()
        private_logits = self._latest_debug["private_logits"].float()
        transfer_gate_weights = self._latest_debug["transfer_gate_weights"].float()
        gate_entropy = -(transfer_gate_weights * transfer_gate_weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
        stats.update(
            {
                "transfer_feature_var": float(transfer_hidden.var(unbiased=False).item()),
                "private_feature_var": float(private_hidden.var(unbiased=False).item()),
                "transfer_private_align": float(F.cosine_similarity(transfer_hidden, private_hidden, dim=-1).mean().item()),
                "transfer_logit_var": float(transfer_logits.var(unbiased=False).item()),
                "private_logit_var": float(private_logits.var(unbiased=False).item()),
                "transfer_gate_entropy": float(gate_entropy.item()),
                "transfer_top1_mean": float(transfer_gate_weights.max(dim=-1).values.mean().item()),
            }
        )
        return stats

    def _record_debug_tensors(self, recorder, step: Optional[int] = None) -> None:
        super()._record_debug_tensors(recorder, step=step)
        if recorder is None or not self._latest_debug:
            return
        recorder.record("ple_adv_transfer_hidden", self._latest_debug["transfer_hidden"], step=step)
        recorder.record("ple_adv_private_hidden", self._latest_debug["private_hidden"], step=step)
        recorder.record("ple_adv_transfer_logits", self._latest_debug["transfer_logits"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_private_logits", self._latest_debug["private_logits"].unsqueeze(-1), step=step)
        recorder.record("ple_adv_transfer_gate_weights", self._latest_debug["transfer_gate_weights"], step=step)

    def custom_train_step(self, batch_interaction, ctx):
        loss_value = super().custom_train_step(batch_interaction, ctx)
        step = ctx.global_step + 1
        if ctx.recorder is not None and step % 200 == 0 and self._latest_debug:
            contrib = self.contribution_state()
            print(
                "[PLE-Adv Transfer] "
                f"step={step} "
                f"var(transfer/private)={contrib['transfer_feature_var']:.4f}/{contrib['private_feature_var']:.4f} "
                f"logit_var(transfer/private)={contrib['transfer_logit_var']:.4f}/{contrib['private_logit_var']:.4f} "
                f"align={contrib['transfer_private_align']:.4f} "
                f"gate(ent/top1)={contrib['transfer_gate_entropy']:.4f}/{contrib['transfer_top1_mean']:.4f}"
            )
        return loss_value
