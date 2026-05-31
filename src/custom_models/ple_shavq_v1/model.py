from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from custom_models.ple_balanced_v3.model import (
    PLEBalancedV3Model,
    gradient_reversal,
)


class PLESHAVQV1Model(PLEBalancedV3Model):
    """
    Minimal fusion:
    - keep `ple_balanced_v3` trunk, common branch, specific branch and gate
    - replace only the balanced shared branch with a VQ branch
    - optionally use domain-balanced EMA to update the codebook
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
        )
        self.codebook_size = int(codebook_size)
        self.ema_decay = float(ema_decay)
        self.commitment_weight = float(commitment_weight)
        self.warmup_samples = max(int(warmup_samples), self.codebook_size)
        self.kmeans_iters = max(1, int(kmeans_iters))
        self.dead_code_threshold_steps = max(0, int(dead_code_threshold_steps))
        self.max_revived_codes_per_step = max(0, int(max_revived_codes_per_step))
        self.domain_balanced_ema = bool(domain_balanced_ema)

        code_dim = self.encoder.output_dim
        initial_codebook = F.normalize(torch.randn(self.codebook_size, code_dim), dim=-1)
        self.register_buffer("codebook", initial_codebook)
        self.register_buffer("codebook_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("code_usage", torch.zeros(self.codebook_size, dtype=torch.long))
        self.register_buffer("steps_since_hit", torch.zeros(self.codebook_size, dtype=torch.long))
        self._warmup_vectors: List[torch.Tensor] = []
        self._warmup_vector_count = 0
        self._latest_commitment_loss = torch.tensor(0.0)

    def _is_codebook_ready(self) -> bool:
        return bool(self.codebook_initialized.item())

    def _collect_warmup_vectors(self, z: torch.Tensor) -> None:
        if self._is_codebook_ready():
            return
        cached = z.detach().cpu()
        self._warmup_vectors.append(cached)
        self._warmup_vector_count += int(cached.size(0))
        if self._warmup_vector_count >= self.warmup_samples:
            self._initialize_codebook_from_warmup(device=z.device)

    def _initialize_codebook_from_warmup(self, device: torch.device) -> None:
        if self._is_codebook_ready() or not self._warmup_vectors:
            return
        data = torch.cat(self._warmup_vectors, dim=0)
        data = F.normalize(data, dim=-1)
        centroids = self._run_kmeans(data, self.codebook_size, self.kmeans_iters)
        self.codebook.copy_(F.normalize(centroids.to(device=device, dtype=self.codebook.dtype), dim=-1))
        self.codebook_initialized.fill_(True)
        self._warmup_vectors.clear()
        self._warmup_vector_count = 0

    @staticmethod
    def _run_kmeans(data: torch.Tensor, num_clusters: int, num_iters: int) -> torch.Tensor:
        num_samples = int(data.size(0))
        if num_samples == 0:
            raise ValueError("K-Means warmup requires at least one sample.")
        if num_samples >= num_clusters:
            perm = torch.randperm(num_samples, device=data.device)[:num_clusters]
            centroids = data[perm].clone()
        else:
            extra_idx = torch.randint(0, num_samples, (num_clusters - num_samples,), device=data.device)
            centroids = torch.cat([data.clone(), data[extra_idx]], dim=0)

        for _ in range(num_iters):
            similarity = data @ centroids.t()
            assignments = similarity.argmax(dim=1)
            new_centroids = []
            for code_idx in range(num_clusters):
                mask = assignments == code_idx
                if mask.any():
                    centroid = data[mask].mean(dim=0)
                else:
                    centroid = data[torch.randint(0, num_samples, (1,), device=data.device)].squeeze(0)
                new_centroids.append(F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0))
            updated = torch.stack(new_centroids, dim=0)
            if torch.allclose(updated, centroids, atol=1e-4, rtol=1e-4):
                centroids = updated
                break
            centroids = updated
        return centroids

    def _quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_codebook = F.normalize(self.codebook, dim=-1)
        similarity = z @ normalized_codebook.t()
        code_indices = similarity.argmax(dim=1)
        quantized = normalized_codebook[code_indices]
        matched_similarity = similarity.gather(1, code_indices.unsqueeze(1)).squeeze(1)
        return quantized, code_indices, matched_similarity

    @torch.no_grad()
    def _ema_update_codebook(self, z: torch.Tensor, code_indices: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if not self.training or not self._is_codebook_ready():
            return

        hit_counts = torch.bincount(code_indices, minlength=self.codebook_size).to(self.code_usage.device)
        self.code_usage.add_(hit_counts.long())
        self.steps_since_hit.add_(1)
        used_mask = hit_counts > 0
        self.steps_since_hit[used_mask] = 0

        if used_mask.any():
            means = torch.zeros_like(self.codebook)
            for code_idx in torch.nonzero(used_mask, as_tuple=False).view(-1).tolist():
                mask = code_indices == code_idx
                z_k = z[mask]
                if self.domain_balanced_ema:
                    domain_k = domain_ids[mask]
                    domain_means = []
                    for domain_idx in range(self.num_domains):
                        domain_mask = domain_k == domain_idx
                        if domain_mask.any():
                            domain_means.append(z_k[domain_mask].mean(dim=0))
                    if domain_means:
                        means[code_idx] = torch.stack(domain_means, dim=0).mean(dim=0)
                else:
                    means[code_idx] = z_k.mean(dim=0)
            updated = self.ema_decay * self.codebook[used_mask] + (1.0 - self.ema_decay) * means[used_mask]
            self.codebook[used_mask] = F.normalize(updated, dim=-1)

        self._revive_dead_codes(z)

    @torch.no_grad()
    def _revive_dead_codes(self, z: torch.Tensor) -> None:
        if self.dead_code_threshold_steps <= 0 or self.max_revived_codes_per_step <= 0:
            return
        dead_mask = self.steps_since_hit >= self.dead_code_threshold_steps
        if not dead_mask.any():
            return
        dead_indices = torch.nonzero(dead_mask, as_tuple=False).view(-1)
        num_replace = min(int(dead_indices.numel()), int(z.size(0)), self.max_revived_codes_per_step)
        if num_replace <= 0:
            return
        codebook = F.normalize(self.codebook, dim=-1)
        hardest_scores = 1.0 - (z @ codebook.t()).amax(dim=1)
        hard_sample_indices = hardest_scores.topk(k=num_replace).indices
        oldest_dead_indices = self.steps_since_hit[dead_indices].topk(k=num_replace).indices
        chosen_dead_codes = dead_indices[oldest_dead_indices]
        self.codebook[chosen_dead_codes] = z[hard_sample_indices]
        self.steps_since_hit[chosen_dead_codes] = 0

    def _forward_dict(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        domain_embed = self.domain_embedding(domain_ids)
        gate_inputs = torch.cat([specific_hidden, common_hidden, balanced_hidden, domain_embed], dim=-1)
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
        }

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
            "balanced_domain_losses": balanced_domain_losses.detach(),
            "balanced_present_mask": balanced_present_mask.detach(),
            "balanced_domain_adv_losses": balanced_domain_adv_losses.detach(),
            "balanced_domain_adv_present_mask": balanced_domain_adv_present_mask.detach(),
            "common_probe_losses": common_probe_losses.detach(),
            "common_probe_present_mask": common_probe_present_mask.detach(),
        }
        return total_loss

    def debug_state(self) -> Dict[str, float]:
        state = super().debug_state()
        usage = self.code_usage.float()
        used_codes = int((usage > 0).sum().item())
        probs = usage / usage.sum().clamp_min(1.0)
        entropy = float((-(probs[probs > 0] * probs[probs > 0].log()).sum()).item()) if used_codes > 0 else 0.0
        state.update({
            "codebook_initialized": float(self.codebook_initialized.item()),
            "used_codes": float(used_codes),
            "used_code_ratio": used_codes / max(1, self.codebook_size),
            "code_usage_entropy": entropy,
            "commitment_loss": float(self._latest_commitment_loss.item()),
        })
        return state

    def contribution_state(self) -> Dict[str, float]:
        state = super().contribution_state()
        if not self._latest_debug:
            state.update({
                "balanced_quantized_var": 0.0,
                "balanced_latent_var": 0.0,
                "balanced_quantized_cos_mean": 0.0,
            })
            for domain_idx in range(self.num_domains):
                prefix = "d%d_" % domain_idx
                state[prefix + "code_used_ratio"] = 0.0
                state[prefix + "code_entropy"] = 0.0
            return state
        state.update({
            "balanced_quantized_var": float(self._latest_debug["balanced_quantized"].float().var(unbiased=False).item()),
            "balanced_latent_var": float(self._latest_debug["balanced_latent"].float().var(unbiased=False).item()),
            "balanced_quantized_cos_mean": float(self._latest_debug["balanced_quantized_cos"].float().mean().item()),
        })
        domain_ids = self._latest_debug["domain_ids"].long()
        code_indices = self._latest_debug["balanced_code_indices"].long()
        for domain_idx in range(self.num_domains):
            prefix = "d%d_" % domain_idx
            mask = domain_ids == domain_idx
            if not mask.any():
                state[prefix + "code_used_ratio"] = 0.0
                state[prefix + "code_entropy"] = 0.0
                continue
            counts = torch.bincount(code_indices[mask], minlength=self.codebook_size).float()
            used_ratio = float((counts > 0).float().mean().item())
            probs = counts / counts.sum().clamp_min(1.0)
            entropy = float((-(probs[probs > 0] * probs[probs > 0].log()).sum()).item()) if (counts > 0).any() else 0.0
            state[prefix + "code_used_ratio"] = used_ratio
            state[prefix + "code_entropy"] = entropy
        return state

    def _record_debug_tensors(self, recorder) -> None:
        super()._record_debug_tensors(recorder)
        if recorder is None or not self._latest_debug:
            return
        code_indices = self._latest_debug["balanced_code_indices"]
        code_hist = F.one_hot(code_indices, num_classes=self.codebook_size).float()
        recorder.record("ple_shavq_v1_code_usage", code_hist)
        recorder.record("ple_shavq_v1_balanced_latent", self._latest_debug["balanced_latent"])
        recorder.record("ple_shavq_v1_balanced_quantized", self._latest_debug["balanced_quantized"])
        recorder.record("ple_shavq_v1_balanced_quantized_cos", self._latest_debug["balanced_quantized_cos"].unsqueeze(-1))

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx) -> None:
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
            "[PLE-SHAVQ-v1 Debug] "
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
            f"gate_mean(s/c/v)={contrib['gate_specific_mean']:.3f}/{contrib['gate_common_mean']:.3f}/{contrib['gate_balanced_mean']:.3f} "
            f"domain_acc(common/vq)={contrib['common_domain_acc']:.3f}/{contrib['balanced_domain_acc']:.3f} "
            f"domain_ent(common/vq)={contrib['common_domain_entropy']:.3f}/{contrib['balanced_domain_entropy']:.3f} "
            f"gate_by_domain={' '.join(gate_domain_parts)}"
        )
