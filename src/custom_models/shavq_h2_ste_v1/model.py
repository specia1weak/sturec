from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.train.context import TrainContext
from custom_models.shavq_md_ste_v2.model import SHAVQMDSTEV2Model


class SHAVQH2STEV1Model(SHAVQMDSTEV2Model):
    """
    Hierarchical gradient codebook:
    - stage 1 uses one global shared codebook on base features
    - stage 2 uses domain-specific codebooks on stage-1 residuals
    - shared expert consumes stage-1 quantized features
    - specific expert consumes the remaining residual after stage 2
    """

    def __init__(
            self,
            *args,
            shared_codebook_size: Optional[int] = None,
            shared_warmup_samples: Optional[int] = None,
            specific_warmup_samples: Optional[int] = None,
            residual_diversity_weight: float = 0.0,
            residual_diversity_margin: float = 0.0,
            specific_residual_scale: float = 1.0,
            specific_quantized_fusion: bool = False,
            specific_codebook_weight: float = 1.0,
            specific_codebook_init_jitter: float = 0.01,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shared_codebook_size = int(shared_codebook_size or self.codebook_size)
        self.shared_warmup_samples = max(int(shared_warmup_samples or self.warmup_samples), self.shared_codebook_size)
        self.specific_warmup_samples = max(int(specific_warmup_samples or self.warmup_samples), self.codebook_size)
        self.residual_diversity_weight = float(residual_diversity_weight)
        self.residual_diversity_margin = float(residual_diversity_margin)
        self.specific_residual_scale = float(specific_residual_scale)
        self.specific_quantized_fusion = bool(specific_quantized_fusion)
        self.specific_codebook_weight = float(specific_codebook_weight)
        self.specific_codebook_init_jitter = float(specific_codebook_init_jitter)

        initial_shared_codebook = F.normalize(
            torch.randn(self.shared_codebook_size, self.projection_dim),
            dim=-1,
        )
        self.shared_codebook = nn.Parameter(initial_shared_codebook)
        self.register_buffer("shared_codebook_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("shared_code_usage", torch.zeros(self.shared_codebook_size))
        self.register_buffer("shared_recent_code_usage", torch.zeros(self.shared_codebook_size))

        self._shared_warmup_vectors: List[torch.Tensor] = []
        self._shared_warmup_vector_count = 0
        self._specific_warmup_vectors: List[torch.Tensor] = []
        self._specific_warmup_domains: List[torch.Tensor] = []
        self._specific_warmup_vector_count = 0

        self._latest_shared_codebook_loss: Optional[torch.Tensor] = None
        self._latest_specific_codebook_loss: Optional[torch.Tensor] = None
        self._latest_shared_commitment_loss: Optional[torch.Tensor] = None
        self._latest_specific_commitment_loss: Optional[torch.Tensor] = None
        self._latest_residual_diversity_loss: Optional[torch.Tensor] = None

        if self.specific_quantized_fusion:
            self.specific_vq_fusion = nn.Linear(self.projection_dim * 2, self.projection_dim)
            nn.init.xavier_uniform_(self.specific_vq_fusion.weight)
            nn.init.zeros_(self.specific_vq_fusion.bias)
        else:
            self.specific_vq_fusion = None

    def component_logits(self) -> Dict[str, torch.Tensor]:
        if not self._latest_debug:
            raise RuntimeError("No debug state available. Run a forward pass first.")

        shared_logits = self._latest_debug["shared_logits"]
        specific_logits = self._latest_debug["specific_logits"]
        return {
            "full": shared_logits + specific_logits,
            "shared_only": shared_logits,
            "specific_only": specific_logits,
        }

    def predict_with_mode(self, interaction, mode: str = "full") -> torch.Tensor:
        x, domain_ids, domain_context = self.encode_features(interaction)
        logits, _, _ = self._forward_impl(
            x,
            domain_ids,
            domain_context,
            compute_aux_losses=False,
        )
        if mode == "full":
            return logits
        components = self.component_logits()
        if mode not in components:
            raise ValueError(f"Unsupported prediction mode: {mode}")
        return components[mode]

    def _is_shared_ready(self) -> bool:
        return bool(self.shared_codebook_initialized.item())

    def _is_specific_ready(self) -> bool:
        return bool(self.domain_codebooks_initialized.all().item())

    def _is_codebook_ready(self) -> bool:
        return self._is_shared_ready() and self._is_specific_ready()

    def _collect_shared_warmup_vectors(self, z: torch.Tensor) -> None:
        if self._is_shared_ready():
            return
        cached = z.detach().cpu()
        self._shared_warmup_vectors.append(cached)
        self._shared_warmup_vector_count += int(cached.size(0))
        if self._shared_warmup_vector_count >= self.shared_warmup_samples:
            self._initialize_shared_codebook_from_warmup(device=z.device)

    def _collect_specific_warmup_vectors(
            self,
            residual_dir: torch.Tensor,
            domain_ids: torch.Tensor,
            valid_mask: torch.Tensor,
    ) -> None:
        if self._is_specific_ready() or not valid_mask.any():
            return
        cached_vectors = residual_dir[valid_mask].detach().cpu()
        cached_domains = domain_ids[valid_mask].detach().cpu()
        self._specific_warmup_vectors.append(cached_vectors)
        self._specific_warmup_domains.append(cached_domains)
        self._specific_warmup_vector_count += int(cached_vectors.size(0))
        if self._specific_warmup_vector_count >= self.specific_warmup_samples:
            self._initialize_specific_codebooks_from_warmup(device=residual_dir.device)

    def _initialize_shared_codebook_from_warmup(self, device: torch.device) -> None:
        if self._is_shared_ready() or not self._shared_warmup_vectors:
            return
        data = torch.cat(self._shared_warmup_vectors, dim=0)
        data = F.normalize(data, dim=-1)
        centroids = self._run_kmeans(data, self.shared_codebook_size, self.kmeans_iters)
        with torch.no_grad():
            self.shared_codebook.copy_(F.normalize(centroids.to(device=device, dtype=self.shared_codebook.dtype), dim=-1))
            self.shared_codebook_initialized.fill_(True)
        self._shared_warmup_vectors.clear()
        self._shared_warmup_vector_count = 0

    def _initialize_specific_codebooks_from_warmup(self, device: torch.device) -> None:
        if self._is_specific_ready() or not self._specific_warmup_vectors:
            return
        data = torch.cat(self._specific_warmup_vectors, dim=0)
        domain_ids = torch.cat(self._specific_warmup_domains, dim=0).long()
        data = F.normalize(data, dim=-1)
        global_centroids = self._run_kmeans(data, self.codebook_size, self.kmeans_iters)
        global_centroids = F.normalize(global_centroids.to(device=device, dtype=self.domain_codebooks.dtype), dim=-1)

        with torch.no_grad():
            for domain_idx in range(self.num_domains):
                domain_mask = domain_ids == domain_idx
                if int(domain_mask.sum().item()) >= self.codebook_size:
                    domain_data = data[domain_mask]
                    centroids = self._run_kmeans(domain_data, self.codebook_size, self.kmeans_iters)
                    domain_centroids = F.normalize(
                        centroids.to(device=device, dtype=self.domain_codebooks.dtype),
                        dim=-1,
                    )
                else:
                    domain_centroids = global_centroids

                if self.specific_codebook_init_jitter > 0.0:
                    noise = self.specific_codebook_init_jitter * torch.randn_like(domain_centroids)
                    domain_centroids = F.normalize(domain_centroids + noise, dim=-1)
                self.domain_codebooks[domain_idx].copy_(domain_centroids)
            self.domain_codebooks_initialized.fill_(True)

        self._specific_warmup_vectors.clear()
        self._specific_warmup_domains.clear()
        self._specific_warmup_vector_count = 0

    def _quantize_shared(
            self,
            z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codebook = F.normalize(self.shared_codebook, dim=-1)
        similarity = z @ codebook.t()
        code_indices = similarity.argmax(dim=-1)
        quantized = codebook[code_indices]
        matched_similarity = similarity.gather(1, code_indices.unsqueeze(1)).squeeze(1)
        return quantized, code_indices, matched_similarity

    def _prepare_specific_residual(
            self,
            shared_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual_norm = shared_residual.norm(dim=-1, keepdim=True)
        valid_mask = residual_norm.squeeze(-1) > 1e-6
        residual_dir = torch.zeros_like(shared_residual)
        if valid_mask.any():
            residual_dir[valid_mask] = shared_residual[valid_mask] / residual_norm[valid_mask].clamp_min(1e-8)
        return residual_dir, valid_mask

    def _quantize_specific_residual(
            self,
            shared_residual: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_dir, valid_mask = self._prepare_specific_residual(shared_residual)
        selected_codebooks = F.normalize(self.domain_codebooks, dim=-1)[domain_ids]
        similarity = torch.einsum("bp,bkp->bk", residual_dir, selected_codebooks)
        code_indices = similarity.argmax(dim=-1)
        batch_indices = torch.arange(shared_residual.size(0), device=shared_residual.device)
        quantized_dir = selected_codebooks[batch_indices, code_indices]
        residual_norm = shared_residual.norm(dim=-1, keepdim=True)
        quantized = residual_norm * quantized_dir
        matched_similarity = similarity[batch_indices, code_indices]
        matched_similarity = torch.where(
            valid_mask,
            matched_similarity,
            torch.ones_like(matched_similarity),
        )
        return quantized, code_indices, matched_similarity, residual_dir, valid_mask

    @torch.no_grad()
    def _update_shared_usage(
            self,
            shared_code_indices: torch.Tensor,
    ) -> None:
        hit_counts = torch.bincount(
            shared_code_indices,
            minlength=self.shared_codebook_size,
        ).to(self.shared_code_usage.dtype)
        self.shared_code_usage.add_(hit_counts)
        self.shared_recent_code_usage.mul_(self.usage_ema_decay).add_(
            hit_counts,
            alpha=1.0 - self.usage_ema_decay,
        )

    def _specific_vq_pair_loss(
            self,
            shared_residual: torch.Tensor,
            specific_quantized: torch.Tensor,
            valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_weight = valid_mask.float()
        if self.specific_codebook_weight != 1.0:
            sample_weight = sample_weight * self.specific_codebook_weight
        return self._vq_pair_loss(shared_residual, specific_quantized, sample_weight)

    def _residual_diversity_loss(
            self,
            shared_residual: torch.Tensor,
            final_residual: torch.Tensor,
            valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.residual_diversity_weight <= 0.0 or not valid_mask.any():
            return shared_residual.new_zeros(())

        shared_norm = F.normalize(shared_residual[valid_mask], dim=-1)
        final_norm = F.normalize(final_residual[valid_mask], dim=-1)
        cosine = (shared_norm * final_norm).sum(dim=-1)
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

        if self.training and not self._is_shared_ready():
            self._collect_shared_warmup_vectors(z)

        if (not self.training) and (not self._is_shared_ready()) and self._shared_warmup_vector_count >= self.shared_warmup_samples:
            self._initialize_shared_codebook_from_warmup(device=z.device)

        shared_code_indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        specific_code_indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        shared_similarity = torch.ones(z.size(0), device=z.device, dtype=z.dtype)
        specific_similarity = torch.ones(z.size(0), device=z.device, dtype=z.dtype)
        shared_quantized = z.detach()
        specific_quantized = torch.zeros_like(z)
        shared_residual = torch.zeros_like(z)
        final_residual = torch.zeros_like(z)
        residual_dir = torch.zeros_like(z)
        valid_mask = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
        shared_commitment_loss = z.new_zeros(())
        shared_codebook_loss = z.new_zeros(())
        specific_commitment_loss = z.new_zeros(())
        specific_codebook_loss = z.new_zeros(())
        residual_diversity_loss = z.new_zeros(())

        if self._is_shared_ready():
            shared_quantized, shared_code_indices, shared_similarity = self._quantize_shared(z)
            shared_residual = z - shared_quantized
            residual_dir, valid_mask = self._prepare_specific_residual(shared_residual)

            if self.training and not self._is_specific_ready():
                self._collect_specific_warmup_vectors(residual_dir, domain_ids, valid_mask)

            if (not self.training) and (not self._is_specific_ready()) and self._specific_warmup_vector_count >= self.specific_warmup_samples:
                self._initialize_specific_codebooks_from_warmup(device=z.device)

            shared_commitment_loss, shared_codebook_loss = self._vq_pair_loss(
                z,
                shared_quantized,
                torch.ones(z.size(0), device=z.device, dtype=z.dtype),
            )

            if self._is_specific_ready():
                specific_quantized, specific_code_indices, specific_similarity, residual_dir, valid_mask = (
                    self._quantize_specific_residual(shared_residual, domain_ids)
                )
                final_residual = shared_residual - self.specific_residual_scale * specific_quantized
                specific_commitment_loss, specific_codebook_loss = self._specific_vq_pair_loss(
                    shared_residual,
                    specific_quantized,
                    valid_mask,
                )
                residual_diversity_loss = self._residual_diversity_loss(
                    shared_residual,
                    final_residual,
                    valid_mask,
                )
                if self.training:
                    self._update_domain_usage(
                        domain_ids.detach(),
                        specific_code_indices.detach(),
                        domain_ids.detach(),
                        specific_code_indices.detach(),
                    )
            else:
                final_residual = shared_residual

            if self.training:
                self._update_shared_usage(shared_code_indices.detach())
        else:
            final_residual = torch.zeros_like(z)

        shared_z = shared_quantized + (z - z.detach()) if self._is_shared_ready() else z
        specific_z = final_residual
        if self.specific_vq_fusion is not None:
            fused_specific = torch.cat([specific_quantized, final_residual], dim=-1)
            specific_z = final_residual + self.specific_vq_fusion(fused_specific)

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

        combined_quantized = shared_quantized + self.specific_residual_scale * specific_quantized
        combined_quantized_norm = combined_quantized.norm(dim=-1, keepdim=True)
        combined_quantized_dir = torch.where(
            combined_quantized_norm > 1e-6,
            combined_quantized / combined_quantized_norm.clamp_min(1e-8),
            shared_quantized,
        )
        total_similarity = (combined_quantized_dir * z).sum(dim=-1)

        contrastive_loss = self._contrastive_loss(x) if compute_aux_losses else logits.new_zeros(())
        encoder_commitment_loss = shared_commitment_loss + specific_commitment_loss
        codebook_loss = shared_codebook_loss + specific_codebook_loss
        commitment_loss = encoder_commitment_loss + codebook_loss + residual_diversity_loss

        self._latest_commitment_loss = commitment_loss.detach()
        self._latest_encoder_commitment_loss = encoder_commitment_loss.detach()
        self._latest_codebook_loss = codebook_loss.detach()
        self._latest_shared_commitment_loss = shared_commitment_loss.detach()
        self._latest_specific_commitment_loss = specific_commitment_loss.detach()
        self._latest_shared_codebook_loss = shared_codebook_loss.detach()
        self._latest_specific_codebook_loss = specific_codebook_loss.detach()
        self._latest_residual_diversity_loss = residual_diversity_loss.detach()
        self._latest_contrastive_loss = contrastive_loss.detach()
        self._latest_code_indices = specific_code_indices.detach()
        self._latest_quantized_share = shared_quantized.detach()
        self._latest_borrow_domain_ids = domain_ids.detach()
        self._latest_debug = {
            "domain_ids": domain_ids.detach(),
            "shared_code_indices": shared_code_indices.detach(),
            "code_indices": specific_code_indices.detach(),
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
            "quantized_cos": total_similarity.detach(),
            "shared_quantized_cos": shared_similarity.detach(),
            "specific_quantized_cos": specific_similarity.detach(),
            "shared_quantized": shared_quantized.detach(),
            "specific_quantized": specific_quantized.detach(),
            "shared_residual_norm": shared_residual.detach().norm(dim=-1),
            "final_residual_norm": final_residual.detach().norm(dim=-1),
            "valid_mask": valid_mask.detach(),
        }
        return logits, commitment_loss, contrastive_loss

    def _record_debug_tensors(self, recorder) -> None:
        if recorder is None or not self._latest_debug:
            return

        shared_hist = F.one_hot(
            self._latest_debug["shared_code_indices"],
            num_classes=self.shared_codebook_size,
        ).float()
        recorder.record("shavq_h2_ste_v1_shared_code_usage", shared_hist)

        specific_hist = F.one_hot(
            self._latest_debug["code_indices"],
            num_classes=self.codebook_size,
        ).float()
        specific_hist = specific_hist * self._latest_debug["valid_mask"].unsqueeze(-1).float()
        recorder.record("shavq_h2_ste_v1_specific_code_usage", specific_hist)
        recorder.record("shavq_h2_ste_v1_shared_hidden", self._latest_debug["shared_hidden"])
        recorder.record("shavq_h2_ste_v1_shared_logits", self._latest_debug["shared_logits"].unsqueeze(-1))
        recorder.record("shavq_h2_ste_v1_specific_logits", self._latest_debug["specific_logits"].unsqueeze(-1))
        recorder.record("shavq_h2_ste_v1_shared_norm", self._latest_debug["shared_norm"].unsqueeze(-1))
        recorder.record("shavq_h2_ste_v1_residual_norm", self._latest_debug["residual_norm"].unsqueeze(-1))
        recorder.record("shavq_h2_ste_v1_shared_quantized_cos", self._latest_debug["shared_quantized_cos"].unsqueeze(-1))
        recorder.record("shavq_h2_ste_v1_specific_quantized_cos", self._latest_debug["specific_quantized_cos"].unsqueeze(-1))

    def debug_state(self) -> Dict[str, float]:
        shared_usage = self.shared_recent_code_usage.float()
        shared_used = float((shared_usage > 1e-8).sum().item())
        shared_probs = shared_usage / shared_usage.sum().clamp_min(1.0)
        shared_entropy = float(
            (-(shared_probs.clamp_min(1e-12) * shared_probs.clamp_min(1e-12).log()).sum()).item()
        ) if shared_used > 0 else 0.0

        specific_usage = self.domain_recent_code_usage.float()
        specific_used = (specific_usage > 1e-8).sum(dim=-1).float()
        specific_probs = specific_usage / specific_usage.sum(dim=-1, keepdim=True).clamp_min(1.0)
        specific_entropy = -(specific_probs.clamp_min(1e-12) * specific_probs.clamp_min(1e-12).log()).sum(dim=-1)

        return {
            "shared_codebook_initialized": float(self.shared_codebook_initialized.item()),
            "specific_codebook_initialized": float(self.domain_codebooks_initialized.all().item()),
            "shared_used_codes": shared_used,
            "shared_used_code_ratio": shared_used / max(1, self.shared_codebook_size),
            "shared_code_usage_entropy": shared_entropy,
            "specific_mean_used_codes": float(specific_used.mean().item()),
            "specific_min_used_codes": float(specific_used.min().item()),
            "specific_max_used_codes": float(specific_used.max().item()),
            "specific_mean_used_code_ratio": float((specific_used / max(1, self.codebook_size)).mean().item()),
            "specific_mean_code_usage_entropy": float(specific_entropy.mean().item()),
            "latest_commitment_loss": float(self._latest_commitment_loss.item()) if self._latest_commitment_loss is not None else 0.0,
            "latest_encoder_commitment_loss": float(self._latest_encoder_commitment_loss.item()) if self._latest_encoder_commitment_loss is not None else 0.0,
            "latest_codebook_loss": float(self._latest_codebook_loss.item()) if self._latest_codebook_loss is not None else 0.0,
            "latest_shared_commitment_loss": float(self._latest_shared_commitment_loss.item()) if self._latest_shared_commitment_loss is not None else 0.0,
            "latest_specific_commitment_loss": float(self._latest_specific_commitment_loss.item()) if self._latest_specific_commitment_loss is not None else 0.0,
            "latest_shared_codebook_loss": float(self._latest_shared_codebook_loss.item()) if self._latest_shared_codebook_loss is not None else 0.0,
            "latest_specific_codebook_loss": float(self._latest_specific_codebook_loss.item()) if self._latest_specific_codebook_loss is not None else 0.0,
            "latest_residual_diversity_loss": float(self._latest_residual_diversity_loss.item()) if self._latest_residual_diversity_loss is not None else 0.0,
            "latest_contrastive_loss": float(self._latest_contrastive_loss.item()) if self._latest_contrastive_loss is not None else 0.0,
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
                "shared_residual_norm_mean": 0.0,
                "final_residual_norm_mean": 0.0,
                "shared_quantized_cos_mean": 0.0,
                "specific_quantized_cos_mean": 0.0,
                "total_quantized_cos_mean": 0.0,
            }

        shared_hidden = self._latest_debug["shared_hidden"].float()
        specific_feature = self._latest_debug["specific_feature_fluctuation"].float()
        shared_logits = self._latest_debug["shared_logits"].float()
        specific_logits = self._latest_debug["specific_logits"].float()

        shared_feature_var = float(shared_hidden.var(unbiased=False).item())
        specific_feature_var = float(specific_feature.var(unbiased=False).item())
        shared_logit_var = float(shared_logits.var(unbiased=False).item())
        specific_logit_var = float(specific_logits.var(unbiased=False).item())
        feature_total = shared_feature_var + specific_feature_var + 1e-12
        logit_total = shared_logit_var + specific_logit_var + 1e-12

        return {
            "shared_feature_var": shared_feature_var,
            "specific_feature_var": specific_feature_var,
            "shared_logit_var": shared_logit_var,
            "specific_logit_var": specific_logit_var,
            "feature_var_shared_ratio": shared_feature_var / feature_total,
            "logit_var_shared_ratio": shared_logit_var / logit_total,
            "shared_residual_norm_mean": float(self._latest_debug["shared_residual_norm"].float().mean().item()),
            "final_residual_norm_mean": float(self._latest_debug["final_residual_norm"].float().mean().item()),
            "shared_quantized_cos_mean": float(self._latest_debug["shared_quantized_cos"].float().mean().item()),
            "specific_quantized_cos_mean": float(self._latest_debug["specific_quantized_cos"].float().mean().item()),
            "total_quantized_cos_mean": float(self._latest_debug["quantized_cos"].float().mean().item()),
        }

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        loss = self.calculate_loss(batch_interaction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        with torch.no_grad():
            self.shared_codebook.copy_(F.normalize(self.shared_codebook, dim=-1))
            self.domain_codebooks.copy_(F.normalize(self.domain_codebooks, dim=-1))
        self._record_debug_tensors(ctx.recorder)
        if ctx.recorder is not None and (ctx.global_step + 1) % 200 == 0:
            debug = self.debug_state()
            contrib = self.contribution_state()
            print(
                "[SHAVQ-H2-STE-v1 Recorder] "
                f"step={ctx.global_step + 1} "
                f"shared_used={debug['shared_used_codes']:.1f}/{self.shared_codebook_size} "
                f"shared_ratio={debug['shared_used_code_ratio']:.3f} "
                f"specific(mean/min/max)={debug['specific_mean_used_codes']:.1f}/{debug['specific_min_used_codes']:.1f}/{debug['specific_max_used_codes']:.1f} "
                f"specific_ratio={debug['specific_mean_used_code_ratio']:.3f} "
                f"shared_ent={debug['shared_code_usage_entropy']:.3f} "
                f"specific_ent={debug['specific_mean_code_usage_entropy']:.3f} "
                f"vq={debug['latest_commitment_loss']:.5f} "
                f"shared_commit={debug['latest_shared_commitment_loss']:.5f} "
                f"specific_commit={debug['latest_specific_commitment_loss']:.5f} "
                f"shared_codebook={debug['latest_shared_codebook_loss']:.5f} "
                f"specific_codebook={debug['latest_specific_codebook_loss']:.5f} "
                f"res_div={debug['latest_residual_diversity_loss']:.5f} "
                f"contrast={debug['latest_contrastive_loss']:.5f} "
                f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
                f"logit_ratio={contrib['logit_var_shared_ratio']:.3f} "
                f"r1={contrib['shared_residual_norm_mean']:.3f} "
                f"r2={contrib['final_residual_norm_mean']:.3f}"
            )
            print(ctx.recorder.get_window_stats())
        return float(loss.item())

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx) -> None:
        del metrics, ctx
        debug = self.debug_state()
        contrib = self.contribution_state()
        print(
            "[SHAVQ-H2-STE-v1 Debug] "
            f"ready_shared={int(debug['shared_codebook_initialized'])} "
            f"ready_specific={int(debug['specific_codebook_initialized'])} "
            f"shared_used={debug['shared_used_codes']:.1f}/{self.shared_codebook_size} "
            f"specific(mean/min/max)={debug['specific_mean_used_codes']:.1f}/{debug['specific_min_used_codes']:.1f}/{debug['specific_max_used_codes']:.1f} "
            f"shared_ratio={debug['shared_used_code_ratio']:.3f} "
            f"specific_ratio={debug['specific_mean_used_code_ratio']:.3f} "
            f"shared_ent={debug['shared_code_usage_entropy']:.3f} "
            f"specific_ent={debug['specific_mean_code_usage_entropy']:.3f} "
            f"vq={debug['latest_commitment_loss']:.5f} "
            f"shared_commit={debug['latest_shared_commitment_loss']:.5f} "
            f"specific_commit={debug['latest_specific_commitment_loss']:.5f} "
            f"shared_codebook={debug['latest_shared_codebook_loss']:.5f} "
            f"specific_codebook={debug['latest_specific_codebook_loss']:.5f} "
            f"res_div={debug['latest_residual_diversity_loss']:.5f} "
            f"contrast={debug['latest_contrastive_loss']:.5f} "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_ratio={contrib['logit_var_shared_ratio']:.3f} "
            f"r1={contrib['shared_residual_norm_mean']:.3f} "
            f"r2={contrib['final_residual_norm_mean']:.3f} "
            f"total_cos={contrib['total_quantized_cos_mean']:.3f}"
        )
