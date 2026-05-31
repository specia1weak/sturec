from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from custom_models.ple_balanced_v3.model import gradient_reversal
from custom_models.ple_shavq_v2.model import PLESHAVQV2Model
from betterbole.models.utils.general import MLP


class PLESHAVQV3Model(PLESHAVQV2Model):
    """
    V3 adds a residual recovery path on top of the VQ balanced branch.

    The goal is to test whether the VQ branch is useful as a compact shared prior,
    while the residual path restores fine-grained information lost by quantization.
    """

    def __init__(
            self,
            manager,
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
            common_aux_weight: float = 0.02,
            balanced_aux_weight: float = 0.05,
            balanced_domain_adv_weight: float = 0.0,
            balanced_domain_adv_lambda: float = 0.2,
            common_probe_weight: float = 0.01,
            gate_temperature: float = 1.0,
            gate_domain_embed_dim: int = 16,
            gate_hidden_dims: Iterable[int] = (128,),
            branch_dropout_rate: float = 0.0,
            domain_head_hidden_dims: Iterable[int] = (64,),
            codebook_size: int = 64,
            ema_decay: float = 0.99,
            commitment_weight: float = 0.2,
            warmup_samples: int = 8192,
            kmeans_iters: int = 20,
            dead_code_threshold_steps: int = 2000,
            max_revived_codes_per_step: int = 8,
            domain_balanced_ema: bool = True,
            gate_code_embed_dim: int = 16,
            residual_hidden_dims: Iterable[int] = (64,),
            residual_scale: float = 0.5,
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
            common_aux_weight=common_aux_weight,
            balanced_aux_weight=balanced_aux_weight,
            balanced_domain_adv_weight=balanced_domain_adv_weight,
            balanced_domain_adv_lambda=balanced_domain_adv_lambda,
            common_probe_weight=common_probe_weight,
            gate_temperature=gate_temperature,
            gate_domain_embed_dim=gate_domain_embed_dim,
            gate_hidden_dims=gate_hidden_dims,
            branch_dropout_rate=branch_dropout_rate,
            domain_head_hidden_dims=domain_head_hidden_dims,
            codebook_size=codebook_size,
            ema_decay=ema_decay,
            commitment_weight=commitment_weight,
            warmup_samples=warmup_samples,
            kmeans_iters=kmeans_iters,
            dead_code_threshold_steps=dead_code_threshold_steps,
            max_revived_codes_per_step=max_revived_codes_per_step,
            domain_balanced_ema=domain_balanced_ema,
            gate_code_embed_dim=gate_code_embed_dim,
        )

        output_dim = self.encoder.output_dim
        residual_hidden_dims = tuple(residual_hidden_dims) if isinstance(residual_hidden_dims, (tuple, list)) else (residual_hidden_dims,)
        self.residual_scale = float(residual_scale)
        self.residual_projection = MLP(
            output_dim,
            *residual_hidden_dims,
            output_dim,
            dropout_rate=branch_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.residual_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.apply_xavier_initialization()

    def _forward_dict(self, x: torch.Tensor, domain_ids: torch.Tensor):
        forward_dict = super()._forward_dict(x, domain_ids)

        balanced_latent = forward_dict["balanced_latent"]
        balanced_quantized = forward_dict["balanced_quantized"].detach()
        balanced_residual = balanced_latent - balanced_quantized
        balanced_residual_hidden = self.residual_norm(self.residual_projection(balanced_residual))

        balanced_hidden = forward_dict["balanced_hidden"] + self.residual_scale * balanced_residual_hidden
        balanced_logits = self.balanced_head(balanced_hidden).squeeze(-1)
        balanced_domain_logits = self.balanced_domain_discriminator(
            gradient_reversal(balanced_hidden, self.balanced_domain_adv_lambda)
        )

        if self._is_codebook_ready():
            code_embed = self.balanced_code_embedding(forward_dict["balanced_code_indices"])
        else:
            code_embed = self.balanced_code_embedding.weight.new_zeros(
                forward_dict["balanced_code_indices"].size(0),
                self.gate_code_embed_dim,
            )

        gate_inputs = torch.cat(
            [
                forward_dict["specific_hidden"],
                forward_dict["common_hidden"],
                balanced_hidden,
                self.domain_embedding(domain_ids),
                code_embed,
            ],
            dim=-1,
        )
        gate_logits = self.gate_network(gate_inputs) / self.gate_temperature
        gate_weights = torch.softmax(gate_logits, dim=-1)

        specific_weight = gate_weights[:, 0:1]
        common_weight = gate_weights[:, 1:2]
        balanced_weight = gate_weights[:, 2:3]
        fused_hidden = (
            specific_weight * forward_dict["specific_hidden"]
            + common_weight * forward_dict["common_hidden"]
            + balanced_weight * balanced_hidden
        )
        fused_logits = self.head(fused_hidden, domain_ids)

        forward_dict.update(
            {
                "balanced_hidden": balanced_hidden,
                "balanced_logits": balanced_logits,
                "balanced_domain_logits": balanced_domain_logits,
                "balanced_residual": balanced_residual,
                "balanced_residual_hidden": balanced_residual_hidden,
                "gate_weights": gate_weights,
                "fused_hidden": fused_hidden,
                "fused_logits": fused_logits,
                "specific_gated": specific_weight * forward_dict["specific_hidden"],
                "common_gated": common_weight * forward_dict["common_hidden"],
                "balanced_gated": balanced_weight * balanced_hidden,
                "gate_code_embed": code_embed,
            }
        )
        return forward_dict

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_dict(x, domain_ids)

        task_loss = F.binary_cross_entropy_with_logits(forward_dict["fused_logits"], labels)
        common_aux_loss = F.binary_cross_entropy_with_logits(forward_dict["common_logits"], labels)
        balanced_aux_loss, balanced_domain_losses, balanced_present_mask = self._balanced_shared_bce(
            forward_dict["balanced_logits"], labels, domain_ids
        )
        balanced_domain_adv_loss, balanced_domain_adv_losses, balanced_domain_adv_present_mask = self._balanced_domain_ce(
            forward_dict["balanced_domain_logits"], domain_ids
        )
        common_probe_loss, common_probe_losses, common_probe_present_mask = self._balanced_domain_ce(
            forward_dict["common_domain_logits"], domain_ids
        )
        commitment_loss = forward_dict["balanced_commitment_loss"]

        total_loss = (
            task_loss
            + self.common_aux_weight * common_aux_loss
            + self.balanced_aux_weight * balanced_aux_loss
            + self.balanced_domain_adv_weight * balanced_domain_adv_loss
            + self.common_probe_weight * common_probe_loss
            + commitment_loss
        )

        self._latest_task_loss = task_loss.detach()
        self._latest_common_aux_loss = common_aux_loss.detach()
        self._latest_balanced_aux_loss = balanced_aux_loss.detach()
        self._latest_balanced_domain_adv_loss = balanced_domain_adv_loss.detach()
        self._latest_common_probe_loss = common_probe_loss.detach()
        self._latest_commitment_loss = commitment_loss.detach()
        self._latest_debug = {
            "domain_ids": domain_ids.detach(),
            "specific_hidden": forward_dict["specific_hidden"].detach(),
            "shared_hidden_raw": forward_dict["shared_hidden_raw"].detach(),
            "common_hidden_raw": forward_dict["common_hidden_raw"].detach(),
            "balanced_hidden_raw": forward_dict["balanced_hidden_raw"].detach(),
            "common_hidden": forward_dict["common_hidden"].detach(),
            "balanced_hidden": forward_dict["balanced_hidden"].detach(),
            "fused_hidden": forward_dict["fused_hidden"].detach(),
            "common_logits": forward_dict["common_logits"].detach(),
            "balanced_logits": forward_dict["balanced_logits"].detach(),
            "fused_logits": forward_dict["fused_logits"].detach(),
            "common_domain_logits": forward_dict["common_domain_logits"].detach(),
            "balanced_domain_logits": forward_dict["balanced_domain_logits"].detach(),
            "gate_weights": forward_dict["gate_weights"].detach(),
            "specific_gated": forward_dict["specific_gated"].detach(),
            "common_gated": forward_dict["common_gated"].detach(),
            "balanced_gated": forward_dict["balanced_gated"].detach(),
            "balanced_latent": forward_dict["balanced_latent"].detach(),
            "balanced_quantized": forward_dict["balanced_quantized"].detach(),
            "balanced_quantized_cos": forward_dict["balanced_quantized_cos"].detach(),
            "balanced_code_indices": forward_dict["balanced_code_indices"].detach(),
            "balanced_residual": forward_dict["balanced_residual"].detach(),
            "balanced_residual_hidden": forward_dict["balanced_residual_hidden"].detach(),
            "balanced_domain_losses": balanced_domain_losses.detach(),
            "balanced_present_mask": balanced_present_mask.detach(),
            "balanced_domain_adv_losses": balanced_domain_adv_losses.detach(),
            "balanced_domain_adv_present_mask": balanced_domain_adv_present_mask.detach(),
            "common_probe_losses": common_probe_losses.detach(),
            "common_probe_present_mask": common_probe_present_mask.detach(),
        }
        return total_loss

    def contribution_state(self):
        state = super().contribution_state()
        if not self._latest_debug:
            state.update({
                "balanced_residual_var": 0.0,
                "balanced_residual_hidden_var": 0.0,
                "balanced_residual_norm_mean": 0.0,
                "balanced_residual_hidden_abs_mean": 0.0,
            })
            return state

        balanced_residual = self._latest_debug["balanced_residual"].float()
        balanced_residual_hidden = self._latest_debug["balanced_residual_hidden"].float()
        state.update({
            "balanced_residual_var": float(balanced_residual.var(unbiased=False).item()),
            "balanced_residual_hidden_var": float(balanced_residual_hidden.var(unbiased=False).item()),
            "balanced_residual_norm_mean": float(balanced_residual.norm(dim=-1).mean().item()),
            "balanced_residual_hidden_abs_mean": float(balanced_residual_hidden.abs().mean().item()),
        })
        return state

    def _record_debug_tensors(self, recorder) -> None:
        super()._record_debug_tensors(recorder)
        if recorder is None or not self._latest_debug:
            return
        recorder.record("ple_shavq_v3_balanced_residual", self._latest_debug["balanced_residual"])
        recorder.record("ple_shavq_v3_balanced_residual_hidden", self._latest_debug["balanced_residual_hidden"])

    def on_eval_epoch_end(self, metrics, ctx) -> None:
        del metrics, ctx
        debug = self.debug_state()
        contrib = self.contribution_state()
        gate_domain_parts = []
        for domain_idx in range(self.num_domains):
            gate_domain_parts.append(
                "d%d(g=%.2f/%.2f/%.2f,c=%.2f,e=%.2f)" % (
                    domain_idx,
                    contrib["d%d_gate_specific_mean" % domain_idx],
                    contrib["d%d_gate_common_mean" % domain_idx],
                    contrib["d%d_gate_balanced_mean" % domain_idx],
                    contrib["d%d_code_used_ratio" % domain_idx],
                    contrib["d%d_code_entropy" % domain_idx],
                )
            )
        print(
            "[PLE-SHAVQ-v3 Debug] "
            f"task={debug['task_loss']:.5f} "
            f"common_aux={debug['common_aux_loss']:.5f} "
            f"balanced_aux={debug['balanced_aux_loss']:.5f} "
            f"balanced_adv={debug['balanced_domain_adv_loss']:.5f} "
            f"common_probe={debug['common_probe_loss']:.5f} "
            f"commit={debug['commitment_loss']:.5f} "
            f"vq_ready={int(debug['codebook_initialized'])} "
            f"used={int(debug['used_codes'])}/{self.codebook_size} "
            f"usage_ratio={debug['used_code_ratio']:.3f} "
            f"entropy={debug['code_usage_entropy']:.3f} "
            f"var(s/c/v/f)={contrib['specific_var']:.4f}/{contrib['common_var']:.4f}/{contrib['balanced_var']:.4f}/{contrib['fused_var']:.4f} "
            f"vq_var(lat/quant)={contrib['balanced_latent_var']:.4f}/{contrib['balanced_quantized_var']:.4f} "
            f"vq_cos={contrib['balanced_quantized_cos_mean']:.4f} "
            f"res_var={contrib['balanced_residual_var']:.4f} "
            f"res_hidden_var={contrib['balanced_residual_hidden_var']:.4f} "
            f"gate_mean(s/c/v)={contrib['gate_specific_mean']:.3f}/{contrib['gate_common_mean']:.3f}/{contrib['gate_balanced_mean']:.3f} "
            f"domain_acc(common/vq)={contrib['common_domain_acc']:.3f}/{contrib['balanced_domain_acc']:.3f} "
            f"domain_ent(common/vq)={contrib['common_domain_entropy']:.3f}/{contrib['balanced_domain_entropy']:.3f} "
            f"gate_by_domain={' '.join(gate_domain_parts)}"
        )
