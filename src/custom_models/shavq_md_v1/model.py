from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from betterbole.core.train.context import TrainContext
from custom_models.shavq_v1.model import SHAVQV1Model


class SHAVQMDV1Model(SHAVQV1Model):
    """
    Multi-domain codebook variant.

    - each domain owns a dedicated codebook
    - the shared expert consumes the mean quantized prototype across all codebooks
    - the specific expert consumes the target-domain residual only
    - non-target samples weakly update foreign codebooks through tunable weights
    """

    def __init__(
            self,
            *args,
            cross_domain_commitment_weight: float = 0.10,
            cross_domain_ema_weight: float = 0.10,
            codebook_init_jitter: float = 0.01,
            usage_ema_decay: float = 0.99,
            ema_eps: float = 1e-5,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cross_domain_commitment_weight = float(cross_domain_commitment_weight)
        self.cross_domain_ema_weight = float(cross_domain_ema_weight)
        self.codebook_init_jitter = float(codebook_init_jitter)
        self.usage_ema_decay = float(usage_ema_decay)
        self.ema_eps = float(ema_eps)

        initial_codebooks = F.normalize(
            torch.randn(self.num_domains, self.codebook_size, self.projection_dim),
            dim=-1,
        )
        self.register_buffer("domain_codebooks", initial_codebooks)
        self.register_buffer("domain_codebooks_initialized", torch.zeros(self.num_domains, dtype=torch.bool))
        self.register_buffer("domain_code_usage", torch.zeros(self.num_domains, self.codebook_size))
        self.register_buffer("domain_recent_code_usage", torch.zeros(self.num_domains, self.codebook_size))
        self.register_buffer("domain_ema_cluster_size", torch.ones(self.num_domains, self.codebook_size))
        self.register_buffer("domain_ema_cluster_sum", initial_codebooks.clone())
        self.register_buffer("domain_steps_since_hit", torch.zeros(self.num_domains, self.codebook_size, dtype=torch.long))

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
        for domain_idx in range(self.num_domains):
            domain_centroids = centroids
            if self.codebook_init_jitter > 0.0:
                noise = self.codebook_init_jitter * torch.randn_like(domain_centroids)
                domain_centroids = F.normalize(domain_centroids + noise, dim=-1)
            self.domain_codebooks[domain_idx].copy_(domain_centroids)
            self.domain_ema_cluster_size[domain_idx].fill_(1.0)
            self.domain_ema_cluster_sum[domain_idx].copy_(domain_centroids)
        self.domain_codebooks_initialized.fill_(True)
        self._warmup_vectors.clear()
        self._warmup_vector_count = 0

    def _build_cross_domain_weights(
            self,
            domain_ids: torch.Tensor,
            off_domain_weight: float,
    ) -> torch.Tensor:
        weights = torch.full(
            (domain_ids.size(0), self.num_domains),
            float(off_domain_weight),
            device=domain_ids.device,
            dtype=torch.float32,
        )
        weights.scatter_(1, domain_ids.view(-1, 1), 1.0)
        return weights

    def _quantize_all_domains(
            self,
            z: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        codebooks = F.normalize(self.domain_codebooks, dim=-1)
        similarity = torch.einsum("bp,dkp->bdk", z, codebooks)
        code_indices = similarity.argmax(dim=-1)
        quantized_per_domain = []
        for domain_idx in range(self.num_domains):
            quantized_per_domain.append(codebooks[domain_idx][code_indices[:, domain_idx]])
        quantized = torch.stack(quantized_per_domain, dim=1)
        matched_similarity = similarity.gather(-1, code_indices.unsqueeze(-1)).squeeze(-1)

        batch_indices = torch.arange(z.size(0), device=z.device)
        target_quantized = quantized[batch_indices, domain_ids]
        target_code_indices = code_indices[batch_indices, domain_ids]
        target_similarity = matched_similarity[batch_indices, domain_ids]
        return quantized, code_indices, matched_similarity, target_quantized, target_code_indices, target_similarity

    @torch.no_grad()
    def _ema_update_domain_codebooks(
            self,
            z: torch.Tensor,
            domain_ids: torch.Tensor,
            all_code_indices: torch.Tensor,
            sample_weights: torch.Tensor,
    ) -> None:
        if not self.training or not self._is_codebook_ready():
            return

        for domain_idx in range(self.num_domains):
            code_indices = all_code_indices[:, domain_idx]
            weights = sample_weights[:, domain_idx]
            hit_counts = torch.zeros(self.codebook_size, device=z.device, dtype=z.dtype)
            hit_counts.index_add_(0, code_indices, weights)
            embedding_sums = torch.zeros_like(self.domain_codebooks[domain_idx])
            embedding_sums.index_add_(0, code_indices, z * weights.unsqueeze(-1))

            self.domain_code_usage[domain_idx].add_(hit_counts.detach())
            self.domain_recent_code_usage[domain_idx].mul_(self.usage_ema_decay).add_(
                hit_counts.detach(),
                alpha=1.0 - self.usage_ema_decay,
            )
            self.domain_steps_since_hit[domain_idx].add_(1)
            used_mask = hit_counts > 0
            if used_mask.any():
                self.domain_steps_since_hit[domain_idx][used_mask] = 0

            self.domain_ema_cluster_size[domain_idx].mul_(self.ema_decay).add_(
                hit_counts.detach(),
                alpha=1.0 - self.ema_decay,
            )
            self.domain_ema_cluster_sum[domain_idx].mul_(self.ema_decay).add_(
                embedding_sums.detach(),
                alpha=1.0 - self.ema_decay,
            )
            cluster_size = self.domain_ema_cluster_size[domain_idx]
            total_cluster = cluster_size.sum()
            smoothed_cluster_size = (
                (cluster_size + self.ema_eps)
                / (total_cluster + self.codebook_size * self.ema_eps)
                * total_cluster.clamp_min(1.0)
            )
            updated = self.domain_ema_cluster_sum[domain_idx] / smoothed_cluster_size.clamp_min(self.ema_eps).unsqueeze(-1)
            self.domain_codebooks[domain_idx].copy_(F.normalize(updated, dim=-1))

            self._revive_dead_codes_for_domain(domain_idx, z, domain_ids)

    @torch.no_grad()
    def _revive_dead_codes_for_domain(
            self,
            domain_idx: int,
            z: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> None:
        if self.dead_code_threshold_steps <= 0 or self.max_revived_codes_per_step <= 0:
            return

        dead_mask = self.domain_steps_since_hit[domain_idx] >= self.dead_code_threshold_steps
        if not dead_mask.any():
            return

        dead_indices = torch.nonzero(dead_mask, as_tuple=False).view(-1)
        if dead_indices.numel() == 0:
            return

        num_replace = min(
            int(dead_indices.numel()),
            int(z.size(0)),
            self.max_revived_codes_per_step,
        )
        if num_replace <= 0:
            return

        domain_mask = domain_ids == domain_idx
        candidate_z = z[domain_mask] if domain_mask.any() else z
        if candidate_z.size(0) == 0:
            return

        num_replace = min(num_replace, int(candidate_z.size(0)))
        if num_replace <= 0:
            return

        codebook = F.normalize(self.domain_codebooks[domain_idx], dim=-1)
        candidate_z_norm = F.normalize(candidate_z, dim=-1)
        hardest_scores = 1.0 - (candidate_z_norm @ codebook.t()).amax(dim=1)
        hard_sample_indices = hardest_scores.topk(k=num_replace).indices
        oldest_dead_indices = self.domain_steps_since_hit[domain_idx][dead_indices].topk(k=num_replace).indices
        chosen_dead_codes = dead_indices[oldest_dead_indices]
        revived_raw = candidate_z[hard_sample_indices]
        revived = F.normalize(revived_raw, dim=-1)
        self.domain_codebooks[domain_idx][chosen_dead_codes] = revived
        self.domain_ema_cluster_sum[domain_idx][chosen_dead_codes] = revived_raw
        self.domain_ema_cluster_size[domain_idx][chosen_dead_codes] = 1.0
        self.domain_recent_code_usage[domain_idx][chosen_dead_codes] = 1.0
        self.domain_steps_since_hit[domain_idx][chosen_dead_codes] = 0

    def _multi_domain_commitment_loss(
            self,
            z: torch.Tensor,
            quantized_all: torch.Tensor,
            sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        squared_error = (z.unsqueeze(1) - quantized_all.detach()).pow(2).mean(dim=-1)
        weighted = squared_error * sample_weights
        return self.commitment_weight * weighted.sum() / sample_weights.sum().clamp_min(1e-12)

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
            quantized_all, all_code_indices, all_similarity, target_quantized, target_code_indices, target_similarity = (
                self._quantize_all_domains(z, domain_ids)
            )
            shared_quantized = quantized_all.mean(dim=1)
            shared_z = z + (shared_quantized - z).detach()
            specific_z = z - target_quantized.detach()
            sample_weights = self._build_cross_domain_weights(
                domain_ids=domain_ids,
                off_domain_weight=self.cross_domain_commitment_weight,
            )
            commitment_loss = self._multi_domain_commitment_loss(z, quantized_all, sample_weights)
            if self.training:
                ema_weights = self._build_cross_domain_weights(
                    domain_ids=domain_ids,
                    off_domain_weight=self.cross_domain_ema_weight,
                )
                self._ema_update_domain_codebooks(
                    z.detach(),
                    domain_ids.detach(),
                    all_code_indices.detach(),
                    ema_weights.detach(),
                )
        else:
            quantized_all = z.detach().unsqueeze(1).expand(-1, self.num_domains, -1)
            shared_quantized = z.detach()
            target_quantized = z.detach()
            target_code_indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            target_similarity = torch.ones(z.size(0), device=z.device)
            all_code_indices = torch.zeros(z.size(0), self.num_domains, dtype=torch.long, device=z.device)
            all_similarity = torch.ones(z.size(0), self.num_domains, device=z.device)
            shared_z = z
            specific_z = torch.zeros_like(z)
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
        self._latest_contrastive_loss = contrastive_loss.detach()
        self._latest_code_indices = target_code_indices.detach()
        self._latest_quantized_share = shared_quantized.detach()
        self._latest_debug = {
            "domain_ids": domain_ids.detach(),
            "code_indices": target_code_indices.detach(),
            "all_code_indices": all_code_indices.detach(),
            "all_quantized": quantized_all.detach(),
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
            "all_quantized_cos": all_similarity.detach(),
            "target_quantized": target_quantized.detach(),
            "shared_quantized_mean": shared_quantized.detach(),
        }
        return logits, commitment_loss, contrastive_loss

    def _record_debug_tensors(self, recorder) -> None:
        if recorder is None or not self._latest_debug:
            return

        code_indices = self._latest_debug["code_indices"]
        code_hist = F.one_hot(code_indices, num_classes=self.codebook_size).float()
        recorder.record("shavq_md_v1_code_usage", code_hist)
        recorder.record("shavq_md_v1_shared_hidden", self._latest_debug["shared_hidden"])
        recorder.record("shavq_md_v1_shared_logits", self._latest_debug["shared_logits"].unsqueeze(-1))
        recorder.record("shavq_md_v1_specific_logits", self._latest_debug["specific_logits"].unsqueeze(-1))
        recorder.record("shavq_md_v1_shared_norm", self._latest_debug["shared_norm"].unsqueeze(-1))
        recorder.record("shavq_md_v1_residual_norm", self._latest_debug["residual_norm"].unsqueeze(-1))
        recorder.record("shavq_md_v1_quantized_cos", self._latest_debug["quantized_cos"].unsqueeze(-1))

    def debug_state(self) -> Dict[str, float]:
        lifetime_usage = self.domain_code_usage.float()
        recent_usage = self.domain_recent_code_usage.float()
        used_codes_per_domain = (recent_usage > 1e-8).sum(dim=-1).float()
        total_usage_per_domain = recent_usage.sum(dim=-1)
        probs = recent_usage / recent_usage.sum(dim=-1, keepdim=True).clamp_min(1.0)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        lifetime_used_codes_per_domain = (lifetime_usage > 0).sum(dim=-1).float()

        return {
            "codebook_initialized": float(self.domain_codebooks_initialized.all().item()),
            "mean_used_codes": float(used_codes_per_domain.mean().item()),
            "min_used_codes": float(used_codes_per_domain.min().item()),
            "max_used_codes": float(used_codes_per_domain.max().item()),
            "mean_used_code_ratio": float((used_codes_per_domain / max(1, self.codebook_size)).mean().item()),
            "mean_code_usage_entropy": float(entropy.mean().item()),
            "total_code_hits": float(total_usage_per_domain.sum().item()),
            "lifetime_mean_used_codes": float(lifetime_used_codes_per_domain.mean().item()),
            "latest_commitment_loss": float(self._latest_commitment_loss.item()) if self._latest_commitment_loss is not None else 0.0,
            "latest_contrastive_loss": float(self._latest_contrastive_loss.item()) if self._latest_contrastive_loss is not None else 0.0,
        }

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx) -> None:
        del metrics, ctx
        debug = self.debug_state()
        contrib = self.contribution_state()
        print(
            "[SHAVQ-MD-v1 Debug] "
            f"ready={int(debug['codebook_initialized'])} "
            f"used(mean/min/max)={debug['mean_used_codes']:.1f}/{debug['min_used_codes']:.1f}/{debug['max_used_codes']:.1f} "
            f"usage_ratio={debug['mean_used_code_ratio']:.3f} "
            f"entropy={debug['mean_code_usage_entropy']:.3f} "
            f"commit={debug['latest_commitment_loss']:.5f} "
            f"contrast={debug['latest_contrastive_loss']:.5f} "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_ratio={contrib['logit_var_shared_ratio']:.3f}"
        )

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        loss = self.calculate_loss(batch_interaction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        self._record_debug_tensors(ctx.recorder)
        if ctx.recorder is not None and (ctx.global_step + 1) % 200 == 0:
            debug = self.debug_state()
            contrib = self.contribution_state()
            print(
                "[SHAVQ-MD-v1 Recorder] "
                f"step={ctx.global_step + 1} "
                f"used(mean/min/max)={debug['mean_used_codes']:.1f}/{debug['min_used_codes']:.1f}/{debug['max_used_codes']:.1f} "
                f"usage_ratio={debug['mean_used_code_ratio']:.3f} "
                f"entropy={debug['mean_code_usage_entropy']:.3f} "
                f"commit={debug['latest_commitment_loss']:.5f} "
                f"contrast={debug['latest_contrastive_loss']:.5f} "
                f"feat_var(shared/specific)={contrib['shared_feature_var']:.4f}/{contrib['specific_feature_var']:.4f} "
                f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
                f"logit_var(shared/specific)={contrib['shared_logit_var']:.4f}/{contrib['specific_logit_var']:.4f} "
                f"logit_ratio={contrib['logit_var_shared_ratio']:.3f}"
            )
            print(ctx.recorder.get_window_stats())
        return float(loss.item())
