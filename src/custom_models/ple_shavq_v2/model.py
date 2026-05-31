from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from custom_models.ple_balanced_v3.model import gradient_reversal
from custom_models.ple_shavq_v1.model import PLESHAVQV1Model
from betterbole.models.utils.general import MLP


class PLESHAVQV2Model(PLESHAVQV1Model):
    """
    V2 adds the VQ code identity to the gate input.
    The intent is to test whether the discrete prototype id helps routing,
    without changing the shared branch itself.
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
            balanced_domain_adv_weight: float = 0.005,
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
            domain_balanced_ema: bool = False,
            gate_code_embed_dim: int = 16,
    ):
        self.gate_code_embed_dim = int(gate_code_embed_dim)
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
        )

        output_dim = self.encoder.output_dim
        self.balanced_code_embedding = nn.Embedding(self.codebook_size, self.gate_code_embed_dim)
        gate_input_dim = output_dim * 3 + gate_domain_embed_dim + self.gate_code_embed_dim
        gate_hidden_dims = tuple(gate_hidden_dims) if isinstance(gate_hidden_dims, (tuple, list)) else (gate_hidden_dims,)
        self.gate_network = MLP(
            gate_input_dim,
            *gate_hidden_dims,
            3,
            dropout_rate=branch_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.apply_xavier_initialization()

    def _forward_dict(self, x: torch.Tensor, domain_ids: torch.Tensor):
        specific_hidden, shared_hidden_raw = self.encoder(x, domain_ids)

        common_hidden_raw = self.common_projection(shared_hidden_raw)
        balanced_hidden_pre_vq = self.balanced_projection(shared_hidden_raw)
        balanced_latent = F.normalize(balanced_hidden_pre_vq, dim=-1)

        if self.training and not self._is_codebook_ready():
            self._collect_warmup_vectors(balanced_latent)
        if (not self.training) and (not self._is_codebook_ready()) and self._warmup_vector_count >= self.codebook_size:
            self._initialize_codebook_from_warmup(device=balanced_latent.device)

        if self._is_codebook_ready():
            balanced_quantized, balanced_code_indices, balanced_quantized_cos = self._quantize(balanced_latent)
            balanced_latent_st = balanced_latent + (balanced_quantized - balanced_latent).detach()
            commitment_loss = self.commitment_weight * F.mse_loss(balanced_latent, balanced_quantized.detach())
            if self.training:
                self._ema_update_codebook(
                    balanced_latent.detach(),
                    balanced_code_indices.detach(),
                    domain_ids.detach(),
                )
        else:
            balanced_quantized = balanced_latent.detach()
            balanced_code_indices = torch.zeros(balanced_latent.size(0), dtype=torch.long, device=balanced_latent.device)
            balanced_quantized_cos = torch.ones(balanced_latent.size(0), dtype=balanced_latent.dtype, device=balanced_latent.device)
            balanced_latent_st = balanced_latent
            commitment_loss = balanced_latent.new_zeros(())

        common_hidden = self.branch_dropout(self.common_norm(common_hidden_raw))
        balanced_hidden = self.branch_dropout(self.balanced_norm(balanced_latent_st))

        common_logits = self.common_head(common_hidden).squeeze(-1)
        balanced_logits = self.balanced_head(balanced_hidden).squeeze(-1)
        common_domain_logits = self.common_domain_probe(common_hidden.detach())
        balanced_domain_logits = self.balanced_domain_discriminator(
            gradient_reversal(balanced_hidden, self.balanced_domain_adv_lambda)
        )

        if self._is_codebook_ready():
            code_embed = self.balanced_code_embedding(balanced_code_indices)
        else:
            code_embed = self.balanced_code_embedding.weight.new_zeros(
                balanced_code_indices.size(0),
                self.gate_code_embed_dim,
            )
        domain_embed = self.domain_embedding(domain_ids)
        gate_inputs = torch.cat(
            [
                specific_hidden,
                common_hidden,
                balanced_hidden,
                domain_embed,
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
            specific_weight * specific_hidden
            + common_weight * common_hidden
            + balanced_weight * balanced_hidden
        )
        fused_logits = self.head(fused_hidden, domain_ids)

        return {
            "specific_hidden": specific_hidden,
            "shared_hidden_raw": shared_hidden_raw,
            "common_hidden_raw": common_hidden_raw,
            "balanced_hidden_raw": balanced_hidden_pre_vq,
            "common_hidden": common_hidden,
            "balanced_hidden": balanced_hidden,
            "fused_hidden": fused_hidden,
            "common_logits": common_logits,
            "balanced_logits": balanced_logits,
            "fused_logits": fused_logits,
            "common_domain_logits": common_domain_logits,
            "balanced_domain_logits": balanced_domain_logits,
            "gate_weights": gate_weights,
            "specific_gated": specific_weight * specific_hidden,
            "common_gated": common_weight * common_hidden,
            "balanced_gated": balanced_weight * balanced_hidden,
            "balanced_latent": balanced_latent,
            "balanced_quantized": balanced_quantized,
            "balanced_quantized_cos": balanced_quantized_cos,
            "balanced_code_indices": balanced_code_indices,
            "balanced_commitment_loss": commitment_loss,
            "gate_code_embed": code_embed,
        }
