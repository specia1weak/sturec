from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.train.context import TrainContext
from custom_models.shavq_v1.model import SHAVQV1Model


class SHAVQMDSTEV2Model(SHAVQV1Model):
    """
    Multi-domain gradient codebook with sparse borrowed sharing.

    - each domain owns a trainable codebook
    - each sample uses its target-domain codebook for the residual branch
    - shared branch borrows exactly one domain codebook instead of averaging all domains
    - training samples borrow their own domain with high probability and another domain otherwise
    """

    def __init__(
            self,
            *args,
            self_share_prob: float = 0.8,
            foreign_vq_weight: float = 0.1,
            codebook_weight: float = 1.0,
            codebook_init_jitter: float = 0.01,
            usage_ema_decay: float = 0.99,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.self_share_prob = float(self_share_prob)
        self.foreign_vq_weight = float(foreign_vq_weight)
        self.codebook_weight = float(codebook_weight)
        self.codebook_init_jitter = float(codebook_init_jitter)
        self.usage_ema_decay = float(usage_ema_decay)

        initial_codebooks = F.normalize(
            torch.randn(self.num_domains, self.codebook_size, self.projection_dim),
            dim=-1,
        )
        self.domain_codebooks = nn.Parameter(initial_codebooks)
        self.register_buffer("domain_codebooks_initialized", torch.zeros(self.num_domains, dtype=torch.bool))
        self.register_buffer("domain_code_usage", torch.zeros(self.num_domains, self.codebook_size))
        self.register_buffer("domain_recent_code_usage", torch.zeros(self.num_domains, self.codebook_size))

        self._latest_codebook_loss: Optional[torch.Tensor] = None
        self._latest_encoder_commitment_loss: Optional[torch.Tensor] = None
        self._latest_borrow_domain_ids: Optional[torch.Tensor] = None

    def _is_codebook_ready(self) -> bool:
        return bool(self.domain_codebooks_initialized.all().item())

    def _initialize_codebook_from_warmup(self, device: torch.device) -> None:
        if self._is_codebook_ready():
            return
        if not self._warmup_vectors:
            return
        data = torch.cat(self._warmup_vectors, dim=0)
        data = F.normalize(data, dim=-1)
        centroids = self._run_kmeans(data, self.codebook_size, self.kmeans_iters)
        centroids = centroids.to(device=device, dtype=self.domain_codebooks.dtype)
        centroids = F.normalize(centroids, dim=-1)
        with torch.no_grad():
            for domain_idx in range(self.num_domains):
                domain_centroids = centroids
                if self.codebook_init_jitter > 0.0:
                    noise = self.codebook_init_jitter * torch.randn_like(domain_centroids)
                    domain_centroids = F.normalize(domain_centroids + noise, dim=-1)
                self.domain_codebooks[domain_idx].copy_(domain_centroids)
            self.domain_codebooks_initialized.fill_(True)
        self._warmup_vectors.clear()
        self._warmup_vector_count = 0

    def _sample_borrow_domains(self, domain_ids: torch.Tensor) -> torch.Tensor:
        if self.num_domains <= 1:
            return domain_ids

        if self.training:
            borrow_domain_ids = domain_ids.clone()
            choose_foreign = torch.rand_like(domain_ids.float()) > self.self_share_prob
            if choose_foreign.any():
                foreign_offsets = torch.randint(
                    low=1,
                    high=self.num_domains,
                    size=(int(choose_foreign.sum().item()),),
                    device=domain_ids.device,
                )
                borrow_domain_ids[choose_foreign] = (
                    domain_ids[choose_foreign] + foreign_offsets
                ) % self.num_domains
            return borrow_domain_ids

        # Keep evaluation deterministic while preserving the sparse-routing shape.
        return domain_ids

    def _quantize_selected_domains(
            self,
            z: torch.Tensor,
            selected_domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codebooks = F.normalize(self.domain_codebooks, dim=-1)
        selected_codebooks = codebooks[selected_domain_ids]
        similarity = torch.einsum("bp,bkp->bk", z, selected_codebooks)
        code_indices = similarity.argmax(dim=-1)
        batch_indices = torch.arange(z.size(0), device=z.device)
        quantized = selected_codebooks[batch_indices, code_indices]
        matched_similarity = similarity[batch_indices, code_indices]
        return quantized, code_indices, matched_similarity

    @torch.no_grad()
    def _update_domain_usage(
            self,
            target_domain_ids: torch.Tensor,
            target_code_indices: torch.Tensor,
            borrow_domain_ids: torch.Tensor,
            borrow_code_indices: torch.Tensor,
    ) -> None:
        for domain_idx in range(self.num_domains):
            target_mask = target_domain_ids == domain_idx
            borrow_mask = borrow_domain_ids == domain_idx

            hit_counts = torch.zeros(self.codebook_size, device=target_domain_ids.device, dtype=self.domain_code_usage.dtype)
            if target_mask.any():
                target_hits = torch.bincount(
                    target_code_indices[target_mask],
                    minlength=self.codebook_size,
                ).to(hit_counts.dtype)
                hit_counts.add_(target_hits)
            if borrow_mask.any():
                borrow_hits = torch.bincount(
                    borrow_code_indices[borrow_mask],
                    minlength=self.codebook_size,
                ).to(hit_counts.dtype)
                hit_counts.add_(borrow_hits)

            self.domain_code_usage[domain_idx].add_(hit_counts)
            self.domain_recent_code_usage[domain_idx].mul_(self.usage_ema_decay).add_(
                hit_counts,
                alpha=1.0 - self.usage_ema_decay,
            )

    def _vq_pair_loss(
            self,
            z: torch.Tensor,
            quantized: torch.Tensor,
            sample_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_commit = (z - quantized.detach()).pow(2).mean(dim=-1)
        codebook_pull = (quantized - z.detach()).pow(2).mean(dim=-1)
        weight_sum = sample_weight.sum().clamp_min(1e-12)
        commitment_term = self.commitment_weight * (encoder_commit * sample_weight).sum() / weight_sum
        codebook_term = self.codebook_weight * (codebook_pull * sample_weight).sum() / weight_sum
        return commitment_term, codebook_term

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
            specific_z = z - target_quantized

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
            commitment_loss = encoder_commitment_loss + codebook_loss
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

    def _record_debug_tensors(self, recorder) -> None:
        if recorder is None or not self._latest_debug:
            return

        code_indices = self._latest_debug["code_indices"]
        code_hist = F.one_hot(code_indices, num_classes=self.codebook_size).float()
        recorder.record("shavq_md_ste_v2_code_usage", code_hist)
        recorder.record("shavq_md_ste_v2_shared_hidden", self._latest_debug["shared_hidden"])
        recorder.record("shavq_md_ste_v2_shared_logits", self._latest_debug["shared_logits"].unsqueeze(-1))
        recorder.record("shavq_md_ste_v2_specific_logits", self._latest_debug["specific_logits"].unsqueeze(-1))
        recorder.record("shavq_md_ste_v2_shared_norm", self._latest_debug["shared_norm"].unsqueeze(-1))
        recorder.record("shavq_md_ste_v2_residual_norm", self._latest_debug["residual_norm"].unsqueeze(-1))
        recorder.record("shavq_md_ste_v2_quantized_cos", self._latest_debug["quantized_cos"].unsqueeze(-1))

    def debug_state(self) -> Dict[str, float]:
        lifetime_usage = self.domain_code_usage.float()
        recent_usage = self.domain_recent_code_usage.float()
        used_codes_per_domain = (recent_usage > 1e-8).sum(dim=-1).float()
        total_usage_per_domain = recent_usage.sum(dim=-1)
        probs = recent_usage / recent_usage.sum(dim=-1, keepdim=True).clamp_min(1.0)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        lifetime_used_codes_per_domain = (lifetime_usage > 0).sum(dim=-1).float()
        borrow_self_ratio = 0.0
        if self._latest_borrow_domain_ids is not None and self._latest_debug:
            borrow_self_ratio = float((self._latest_borrow_domain_ids == self._latest_debug["domain_ids"]).float().mean().item())

        return {
            "codebook_initialized": float(self.domain_codebooks_initialized.all().item()),
            "mean_used_codes": float(used_codes_per_domain.mean().item()),
            "min_used_codes": float(used_codes_per_domain.min().item()),
            "max_used_codes": float(used_codes_per_domain.max().item()),
            "mean_used_code_ratio": float((used_codes_per_domain / max(1, self.codebook_size)).mean().item()),
            "mean_code_usage_entropy": float(entropy.mean().item()),
            "total_code_hits": float(total_usage_per_domain.sum().item()),
            "lifetime_mean_used_codes": float(lifetime_used_codes_per_domain.mean().item()),
            "borrow_self_ratio": borrow_self_ratio,
            "latest_commitment_loss": float(self._latest_commitment_loss.item()) if self._latest_commitment_loss is not None else 0.0,
            "latest_encoder_commitment_loss": float(self._latest_encoder_commitment_loss.item()) if self._latest_encoder_commitment_loss is not None else 0.0,
            "latest_codebook_loss": float(self._latest_codebook_loss.item()) if self._latest_codebook_loss is not None else 0.0,
            "latest_contrastive_loss": float(self._latest_contrastive_loss.item()) if self._latest_contrastive_loss is not None else 0.0,
        }

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
                "[SHAVQ-MD-STE-v2 Recorder] "
                f"step={ctx.global_step + 1} "
                f"used(mean/min/max)={debug['mean_used_codes']:.1f}/{debug['min_used_codes']:.1f}/{debug['max_used_codes']:.1f} "
                f"usage_ratio={debug['mean_used_code_ratio']:.3f} "
                f"entropy={debug['mean_code_usage_entropy']:.3f} "
                f"borrow_self={debug['borrow_self_ratio']:.3f} "
                f"vq={debug['latest_commitment_loss']:.5f} "
                f"enc_commit={debug['latest_encoder_commitment_loss']:.5f} "
                f"codebook={debug['latest_codebook_loss']:.5f} "
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
            "[SHAVQ-MD-STE-v2 Debug] "
            f"ready={int(debug['codebook_initialized'])} "
            f"used(mean/min/max)={debug['mean_used_codes']:.1f}/{debug['min_used_codes']:.1f}/{debug['max_used_codes']:.1f} "
            f"usage_ratio={debug['mean_used_code_ratio']:.3f} "
            f"entropy={debug['mean_code_usage_entropy']:.3f} "
            f"borrow_self={debug['borrow_self_ratio']:.3f} "
            f"vq={debug['latest_commitment_loss']:.5f} "
            f"enc_commit={debug['latest_encoder_commitment_loss']:.5f} "
            f"codebook={debug['latest_codebook_loss']:.5f} "
            f"contrast={debug['latest_contrastive_loss']:.5f} "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_ratio={contrib['logit_var_shared_ratio']:.3f}"
        )
