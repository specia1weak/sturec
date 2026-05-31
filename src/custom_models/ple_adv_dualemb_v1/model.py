from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import OmniEmbLayer
from betterbole.models.msr.ple.layers import PLELayer, select_domain_output
from custom_models.ple_adv import PLEAdversarialModel
from betterbole.models.utils.common import to_dims
from betterbole.models.utils.general import MLP


class PLEDualInputAdversarialEncoder(nn.Module):
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
            self.layers.append(
                PLELayer(
                    input_dim=layer_input_dim,
                    num_domains=num_domains,
                    expert_dims=expert_dims,
                    num_specific_experts=num_specific_experts,
                    num_shared_experts=num_shared_experts,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
            )
            layer_input_dim = int(expert_dims[-1])

    def forward_all(
            self,
            specific_x: torch.Tensor,
            shared_x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        task_inputs = [specific_x for _ in range(self.num_domains)]
        shared_input = shared_x
        for layer in self.layers:
            task_inputs, shared_input = layer(task_inputs, shared_input)
        return task_inputs, shared_input

    def forward(
            self,
            specific_x: torch.Tensor,
            shared_x: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        task_outputs, shared_output = self.forward_all(specific_x=specific_x, shared_x=shared_x)
        specific_hidden = select_domain_output(task_outputs=task_outputs, domain_ids=domain_ids)
        return specific_hidden, shared_output


class PLEAdversarialDualEmbV1Model(PLEAdversarialModel):
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
            common_scale: float = 1.0,
            joint_to_shared_scale: float = 0.15,
            specific_residual_weight: float = 0.15,
            input_dropout_rate: float = 0.0,
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
        self.common_embedding = OmniEmbLayer(manager=manager)
        self.common_input_view = self.common_embedding.whole_without_domain
        self.joint_input_view = self.omni_embedding.whole_without_domain
        self.input_view = self.joint_input_view
        self.input_dim = self.joint_input_view.embedding_dim
        self.common_scale = float(common_scale)
        self.joint_to_shared_scale = float(joint_to_shared_scale)
        self.specific_residual_weight = float(specific_residual_weight)

        self.encoder = PLEDualInputAdversarialEncoder(
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

        self.common_input_norm = nn.LayerNorm(self.input_dim, elementwise_affine=False)
        self.joint_input_norm = nn.LayerNorm(self.input_dim, elementwise_affine=False)
        self.shared_input_norm = nn.LayerNorm(self.input_dim, elementwise_affine=False)
        self.specific_input_norm = nn.LayerNorm(self.input_dim, elementwise_affine=False)
        self.common_input_dropout = nn.Dropout(input_dropout_rate)
        self.joint_input_dropout = nn.Dropout(input_dropout_rate)

        self.apply_xavier_initialization()

    def encode_features(self, interaction):
        common_x = self.common_input_view(interaction)
        joint_x = self.joint_input_view(interaction)
        domain_ids = interaction[self.DOMAIN].long().view(-1)
        return {
            "common_x": torch.flatten(common_x, start_dim=1),
            "joint_x": torch.flatten(joint_x, start_dim=1),
        }, domain_ids

    def _build_dual_inputs(self, encoded: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        common_x = self.common_input_dropout(self.common_input_norm(encoded["common_x"]))
        joint_x = self.joint_input_dropout(self.joint_input_norm(encoded["joint_x"]))
        shared_x = self.shared_input_norm(self.common_scale * common_x + self.joint_to_shared_scale * joint_x)
        specific_x = joint_x - self.specific_residual_weight * shared_x.detach()
        specific_x = self.specific_input_norm(specific_x)
        return specific_x, shared_x

    def _forward_dict(self, x: Dict[str, torch.Tensor], domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        specific_x, shared_x = self._build_dual_inputs(x)
        specific_hidden_raw, shared_hidden_raw = self.encoder(
            specific_x=specific_x,
            shared_x=shared_x,
            domain_ids=domain_ids,
        )
        shared_hidden = self.shared_readout_norm(shared_hidden_raw)
        latent_hidden = self.latent_norm(self.latent_projector(specific_x))
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
            "common_x": x["common_x"],
            "joint_x": x["joint_x"],
            "shared_input": shared_x,
            "specific_input": specific_x,
            "latent_hidden": latent_hidden,
            "residual_hidden": residual_hidden,
            "specific_hidden_raw": specific_hidden_raw,
            "specific_hidden": specific_hidden,
            "shared_hidden_raw": shared_hidden_raw,
            "shared_hidden": shared_hidden,
            **decoded,
        }

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_dict(x, domain_ids)

        task_loss = torch.nn.functional.binary_cross_entropy_with_logits(forward_dict["fused_logits"], labels)
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
        gate_alignment = torch.nn.functional.binary_cross_entropy(
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
            "common_x": forward_dict["common_x"].detach(),
            "joint_x": forward_dict["joint_x"].detach(),
            "shared_input": forward_dict["shared_input"].detach(),
            "specific_input": forward_dict["specific_input"].detach(),
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

    def contribution_state(self) -> Dict[str, float]:
        stats = super().contribution_state()
        if not self._latest_debug:
            stats.update(
                {
                    "common_input_var": 0.0,
                    "joint_input_var": 0.0,
                    "shared_input_var": 0.0,
                    "specific_input_var": 0.0,
                    "dual_input_cos": 0.0,
                }
            )
            return stats

        common_x = self._latest_debug["common_x"].float()
        joint_x = self._latest_debug["joint_x"].float()
        shared_input = self._latest_debug["shared_input"].float()
        specific_input = self._latest_debug["specific_input"].float()
        stats.update(
            {
                "common_input_var": float(common_x.var(unbiased=False).item()),
                "joint_input_var": float(joint_x.var(unbiased=False).item()),
                "shared_input_var": float(shared_input.var(unbiased=False).item()),
                "specific_input_var": float(specific_input.var(unbiased=False).item()),
                "dual_input_cos": float(nn.functional.cosine_similarity(shared_input, specific_input, dim=-1).mean().item()),
            }
        )
        return stats

    def _record_debug_tensors(self, recorder, step=None) -> None:
        super()._record_debug_tensors(recorder, step=step)
        if recorder is None or not self._latest_debug:
            return
        recorder.record("ple_adv_dualemb_common_x", self._latest_debug["common_x"], step=step)
        recorder.record("ple_adv_dualemb_joint_x", self._latest_debug["joint_x"], step=step)
        recorder.record("ple_adv_dualemb_shared_input", self._latest_debug["shared_input"], step=step)
        recorder.record("ple_adv_dualemb_specific_input", self._latest_debug["specific_input"], step=step)

    def custom_train_step(self, batch_interaction, ctx):
        loss = super().custom_train_step(batch_interaction, ctx)
        step = ctx.global_step + 1
        if ctx.recorder is not None and step % 200 == 0 and self._latest_debug:
            contrib = self.contribution_state()
            print(
                "[PLE-Adv DualEmb] "
                f"step={step} "
                f"input_var(common/joint/shared/spec)="
                f"{contrib['common_input_var']:.4f}/{contrib['joint_input_var']:.4f}/"
                f"{contrib['shared_input_var']:.4f}/{contrib['specific_input_var']:.4f} "
                f"dual_input_cos={contrib['dual_input_cos']:.4f}"
            )
        return loss
