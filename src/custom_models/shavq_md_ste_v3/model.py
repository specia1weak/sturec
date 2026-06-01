from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from betterbole.core.train.context import TrainContext
from custom_models.shavq_md_ste_v2.model import SHAVQMDSTEV2Model


class SHAVQMDSTEV3Model(SHAVQMDSTEV2Model):
    """
    Sparse borrowed gradient codebook with residual diversification.

    On top of SHAVQ-MD-STE-v2:
    - keep sparse borrow routing
    - keep foreign_vq_weight preferably near zero
    - add a penalty that discourages target and borrowed residuals from
      becoming too similar when the borrowed domain is foreign
    """

    def __init__(
            self,
            *args,
            residual_diversity_weight: float = 0.01,
            residual_diversity_margin: float = 0.0,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.residual_diversity_weight = float(residual_diversity_weight)
        self.residual_diversity_margin = float(residual_diversity_margin)
        self._latest_residual_diversity_loss: Optional[torch.Tensor] = None

    def _residual_diversity_loss(
            self,
            target_residual: torch.Tensor,
            borrow_residual: torch.Tensor,
            domain_ids: torch.Tensor,
            borrow_domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.residual_diversity_weight <= 0.0:
            return target_residual.new_zeros(())

        foreign_mask = borrow_domain_ids != domain_ids
        if not foreign_mask.any():
            return target_residual.new_zeros(())

        target_norm = F.normalize(target_residual[foreign_mask], dim=-1)
        borrow_norm = F.normalize(borrow_residual[foreign_mask], dim=-1)
        cosine = (target_norm * borrow_norm).sum(dim=-1)
        penalty = F.relu(cosine - self.residual_diversity_margin)
        return self.residual_diversity_weight * penalty.mean()

    def _forward_impl(
            self,
            x: torch.Tensor,
            domain_ids: torch.Tensor,
            domain_context: Optional[torch.Tensor],
            *,
            compute_aux_losses: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self._project(x)

        if self.training and not self._is_codebook_ready():
            self._collect_warmup_vectors(z)

        if (not self.training) and (not self._is_codebook_ready()) and self._warmup_vector_count >= self.warmup_samples:
            self._initialize_codebook_from_warmup(device=z.device)

        if self._is_codebook_ready():
            borrow_domain_ids = self._sample_borrow_domains(domain_ids)
            target_quantized, target_code_indices, target_similarity = self._quantize_selected_domains(z, domain_ids)
            borrow_quantized, borrow_code_indices, borrow_similarity = self._quantize_selected_domains(z, borrow_domain_ids)

            if self.training:
                self._update_domain_usage(
                    domain_ids.detach(),
                    target_code_indices.detach(),
                    borrow_domain_ids.detach(),
                    borrow_code_indices.detach(),
                )

            shared_z = borrow_quantized + (z - z.detach())
            target_residual = z - target_quantized
            borrow_residual = z - borrow_quantized
            specific_z = target_residual

            target_weight = torch.ones(z.size(0), device=z.device, dtype=z.dtype)
            borrow_weight = torch.where(
                borrow_domain_ids == domain_ids,
                torch.ones_like(target_weight),
                torch.full_like(target_weight, self.foreign_vq_weight),
            )
            target_commitment, target_codebook = self._vq_pair_loss(z, target_quantized, target_weight)
            borrow_commitment, borrow_codebook = self._vq_pair_loss(z, borrow_quantized, borrow_weight)
            encoder_commitment_loss = target_commitment + borrow_commitment
            codebook_loss = target_codebook + borrow_codebook
            residual_diversity_loss = self._residual_diversity_loss(
                target_residual,
                borrow_residual,
                domain_ids,
                borrow_domain_ids,
            )
            commitment_loss = encoder_commitment_loss + codebook_loss + residual_diversity_loss
        else:
            borrow_domain_ids = domain_ids
            borrow_quantized = z.detach()
            borrow_code_indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            borrow_similarity = torch.ones(z.size(0), device=z.device)
            target_quantized = z.detach()
            target_code_indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            target_similarity = torch.ones(z.size(0), device=z.device)
            shared_z = z
            specific_z = torch.zeros_like(z)
            encoder_commitment_loss = z.new_zeros(())
            codebook_loss = z.new_zeros(())
            residual_diversity_loss = z.new_zeros(())
            commitment_loss = z.new_zeros(())

        shared_hidden = self.shared_expert(shared_z)
        shared_logits = self.shared_head(shared_hidden).squeeze(-1)

        if domain_context is not None:
            specific_input = torch.cat([specific_z, domain_context], dim=-1)
        else:
            specific_input = specific_z
        specific_hidden = self.specific_expert(specific_input)
        specific_bias_feature, specific_fluctuation_feature = self.specific_bifurcator(specific_hidden)
        specific_logits_raw = self.specific_head(specific_hidden, domain_ids)
        specific_logits = self.specific_head(specific_fluctuation_feature, domain_ids)
        logits = shared_logits + specific_logits

        contrastive_loss = self._contrastive_loss(x) if compute_aux_losses else logits.new_zeros(())
        self._latest_commitment_loss = commitment_loss.detach()
        self._latest_encoder_commitment_loss = encoder_commitment_loss.detach()
        self._latest_codebook_loss = codebook_loss.detach()
        self._latest_residual_diversity_loss = residual_diversity_loss.detach()
        self._latest_contrastive_loss = contrastive_loss.detach()
        self._latest_code_indices = target_code_indices.detach()
        self._latest_quantized_share = borrow_quantized.detach()
        self._latest_borrow_domain_ids = borrow_domain_ids.detach()
        self._latest_debug = {
            "domain_ids": domain_ids.detach(),
            "borrow_domain_ids": borrow_domain_ids.detach(),
            "code_indices": target_code_indices.detach(),
            "borrow_code_indices": borrow_code_indices.detach(),
            "shared_hidden": shared_hidden.detach(),
            "shared_logits": shared_logits.detach(),
            "shared_feature_importance": shared_hidden.detach().abs(),
            "specific_logits_raw": specific_logits_raw.detach(),
            "specific_logits": specific_logits.detach(),
            "specific_hidden": specific_hidden.detach(),
            "specific_feature_bias": specific_bias_feature.detach(),
            "specific_feature_fluctuation": specific_fluctuation_feature.detach(),
            "specific_feature_importance": specific_fluctuation_feature.detach().abs(),
            "shared_norm": shared_z.detach().norm(dim=-1),
            "residual_norm": specific_z.detach().norm(dim=-1),
            "quantized_cos": target_similarity.detach(),
            "borrow_quantized_cos": borrow_similarity.detach(),
            "target_quantized": target_quantized.detach(),
            "borrow_quantized": borrow_quantized.detach(),
        }
        return logits, commitment_loss, contrastive_loss

    def debug_state(self) -> Dict[str, float]:
        state = super().debug_state()
        state["latest_residual_diversity_loss"] = (
            float(self._latest_residual_diversity_loss.item())
            if self._latest_residual_diversity_loss is not None
            else 0.0
        )
        return state

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        loss = self.calculate_loss(batch_interaction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        with torch.no_grad():
            self.domain_codebooks.copy_(F.normalize(self.domain_codebooks, dim=-1))
        self._record_debug_tensors(ctx.recorder)
        if ctx.recorder is not None and (ctx.global_step + 1) % 200 == 0:
            debug = self.debug_state()
            contrib = self.contribution_state()
            print(
                "[SHAVQ-MD-STE-v3 Recorder] "
                f"step={ctx.global_step + 1} "
                f"used(mean/min/max)={debug['mean_used_codes']:.1f}/{debug['min_used_codes']:.1f}/{debug['max_used_codes']:.1f} "
                f"usage_ratio={debug['mean_used_code_ratio']:.3f} "
                f"entropy={debug['mean_code_usage_entropy']:.3f} "
                f"borrow_self={debug['borrow_self_ratio']:.3f} "
                f"vq={debug['latest_commitment_loss']:.5f} "
                f"enc_commit={debug['latest_encoder_commitment_loss']:.5f} "
                f"codebook={debug['latest_codebook_loss']:.5f} "
                f"res_div={debug['latest_residual_diversity_loss']:.5f} "
                f"contrast={debug['latest_contrastive_loss']:.5f} "
                f"feat_var(shared/specific)={contrib['shared_feature_var']:.4f}/{contrib['specific_feature_var']:.4f} "
                f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
                f"logit_var(shared/specific)={contrib['shared_logit_var']:.4f}/{contrib['specific_logit_var']:.4f} "
                f"logit_ratio={contrib['logit_var_shared_ratio']:.3f}"
            )
            print(ctx.recorder.get_window_stats())
        return float(loss.item())

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx) -> None:
        del metrics, ctx
        debug = self.debug_state()
        contrib = self.contribution_state()
        print(
            "[SHAVQ-MD-STE-v3 Debug] "
            f"ready={int(debug['codebook_initialized'])} "
            f"used(mean/min/max)={debug['mean_used_codes']:.1f}/{debug['min_used_codes']:.1f}/{debug['max_used_codes']:.1f} "
            f"usage_ratio={debug['mean_used_code_ratio']:.3f} "
            f"entropy={debug['mean_code_usage_entropy']:.3f} "
            f"borrow_self={debug['borrow_self_ratio']:.3f} "
            f"vq={debug['latest_commitment_loss']:.5f} "
            f"enc_commit={debug['latest_encoder_commitment_loss']:.5f} "
            f"codebook={debug['latest_codebook_loss']:.5f} "
            f"res_div={debug['latest_residual_diversity_loss']:.5f} "
            f"contrast={debug['latest_contrastive_loss']:.5f} "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_ratio={contrib['logit_var_shared_ratio']:.3f}"
        )
