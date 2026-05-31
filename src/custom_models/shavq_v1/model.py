from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.train.context import TrainContext
from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.utils.common import to_dims
from betterbole.models.utils.general import FeatureBifurcator, MLP


class ProjectionEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: Iterable[int] = (256, 128),
            output_dim: int = 64,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        hidden_dims = to_dims(hidden_dims, (256, 128))
        dims = [int(input_dim), *hidden_dims, int(output_dim)]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
        self.net = nn.Sequential(*layers)
        self.output_dim = int(output_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class SHAVQV1Model(MSRModel):
    """
    SHAVQ-V1: isolated baseline that mirrors the current single-prototype shared VQ design.
    """

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            projection_hidden_dims: Iterable[int] = (256, 128),
            projection_dim: int = 64,
            projection_dropout_rate: float = 0.0,
            shared_hidden_dims: Iterable[int] = (128, 64),
            shared_dropout_rate: float = 0.0,
            shared_activation: str = "relu",
            shared_batch_norm: bool = False,
            specific_hidden_dims: Iterable[int] = (128, 64),
            specific_dropout_rate: float = 0.0,
            specific_activation: str = "relu",
            specific_batch_norm: bool = False,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
            codebook_size: int = 64,
            ema_decay: float = 0.99,
            commitment_weight: float = 0.25,
            contrastive_weight: float = 0.05,
            contrastive_temperature: float = 0.2,
            contrastive_dropout_rate: float = 0.1,
            warmup_samples: int = 4096,
            kmeans_iters: int = 15,
            dead_code_threshold_steps: int = 2000,
            max_revived_codes_per_step: int = 4,
            contrastive_warmup_only: bool = True,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field

        self.input_view = self.omni_embedding.whole_without_domain
        self.domain_view = self.omni_embedding.domain
        self.input_dim = self.input_view.embedding_dim
        self.domain_context_dim = self.domain_view.embedding_dim

        if self.input_dim <= 0:
            raise ValueError("SHAVQV1Model requires non-empty input embeddings.")

        self.projection = ProjectionEncoder(
            input_dim=self.input_dim,
            hidden_dims=projection_hidden_dims,
            output_dim=projection_dim,
            dropout_rate=projection_dropout_rate,
        )
        self.projection_dim = int(projection_dim)

        shared_hidden_dims = to_dims(shared_hidden_dims, (128, 64))
        self.shared_expert = MLP(
            self.projection_dim,
            *shared_hidden_dims,
            dropout_rate=shared_dropout_rate,
            activation=shared_activation,
            batch_norm=shared_batch_norm,
        )
        self.shared_output_dim = int(shared_hidden_dims[-1])
        self.shared_head = nn.Linear(self.shared_output_dim, 1)

        specific_hidden_dims = to_dims(specific_hidden_dims, (128, 64))
        specific_input_dim = self.projection_dim + int(self.domain_context_dim)
        self.specific_expert = MLP(
            specific_input_dim,
            *specific_hidden_dims,
            dropout_rate=specific_dropout_rate,
            activation=specific_activation,
            batch_norm=specific_batch_norm,
        )
        self.specific_output_dim = int(specific_hidden_dims[-1])
        self.specific_head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.specific_output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )
        self.specific_bifurcator = FeatureBifurcator(
            self.specific_output_dim,
            feature_dim=-1,
            mean_dims=0,
            normalize_var=False,
        )

        self.codebook_size = int(codebook_size)
        self.ema_decay = float(ema_decay)
        self.commitment_weight = float(commitment_weight)
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_dropout_rate = float(contrastive_dropout_rate)
        self.contrastive_warmup_only = bool(contrastive_warmup_only)
        self.warmup_samples = max(int(warmup_samples), self.codebook_size)
        self.kmeans_iters = max(1, int(kmeans_iters))
        self.dead_code_threshold_steps = max(0, int(dead_code_threshold_steps))
        self.max_revived_codes_per_step = max(0, int(max_revived_codes_per_step))

        initial_codebook = F.normalize(torch.randn(self.codebook_size, self.projection_dim), dim=-1)
        self.register_buffer("codebook", initial_codebook)
        self.register_buffer("codebook_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("code_usage", torch.zeros(self.codebook_size, dtype=torch.long))
        self.register_buffer("steps_since_hit", torch.zeros(self.codebook_size, dtype=torch.long))

        self._warmup_vectors: List[torch.Tensor] = []
        self._warmup_vector_count = 0

        self._latest_commitment_loss: Optional[torch.Tensor] = None
        self._latest_contrastive_loss: Optional[torch.Tensor] = None
        self._latest_code_indices: Optional[torch.Tensor] = None
        self._latest_quantized_share: Optional[torch.Tensor] = None
        self._latest_debug: Dict[str, torch.Tensor] = {}

    def _is_codebook_ready(self) -> bool:
        return bool(self.codebook_initialized.item())

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        x = torch.flatten(x, start_dim=1)
        domain_ids = interaction[self.DOMAIN].long().view(-1)
        domain_context = None
        if self.domain_context_dim > 0:
            domain_context = self.domain_view(interaction)
            domain_context = torch.flatten(domain_context, start_dim=1)
        return x, domain_ids, domain_context

    def _project(self, x: torch.Tensor, *, apply_feature_dropout: bool = False) -> torch.Tensor:
        if apply_feature_dropout and self.contrastive_dropout_rate > 0.0:
            x = F.dropout(x, p=self.contrastive_dropout_rate, training=self.training)
        return self.projection(x)

    def _collect_warmup_vectors(self, z: torch.Tensor) -> None:
        if self._is_codebook_ready():
            return
        cached = z.detach().cpu()
        self._warmup_vectors.append(cached)
        self._warmup_vector_count += int(cached.size(0))
        if self._warmup_vector_count >= self.warmup_samples:
            self._initialize_codebook_from_warmup(device=z.device)

    def _initialize_codebook_from_warmup(self, device: torch.device) -> None:
        if self._is_codebook_ready():
            return
        if not self._warmup_vectors:
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
    def _ema_update_codebook(self, z: torch.Tensor, code_indices: torch.Tensor) -> None:
        if not self.training or not self._is_codebook_ready():
            return

        hit_counts = torch.bincount(code_indices, minlength=self.codebook_size).to(self.code_usage.device)
        self.code_usage.add_(hit_counts.long())
        self.steps_since_hit.add_(1)
        used_mask = hit_counts > 0
        self.steps_since_hit[used_mask] = 0

        if used_mask.any():
            means = torch.zeros_like(self.codebook)
            means.index_add_(0, code_indices, z)
            means = means / hit_counts.clamp_min(1).unsqueeze(-1).to(means.dtype)
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
        if dead_indices.numel() == 0:
            return

        num_replace = min(
            int(dead_indices.numel()),
            int(z.size(0)),
            self.max_revived_codes_per_step,
        )
        if num_replace <= 0:
            return

        codebook = F.normalize(self.codebook, dim=-1)
        hardest_scores = 1.0 - (z @ codebook.t()).amax(dim=1)
        hard_sample_indices = hardest_scores.topk(k=num_replace).indices
        oldest_dead_indices = self.steps_since_hit[dead_indices].topk(k=num_replace).indices
        chosen_dead_codes = dead_indices[oldest_dead_indices]
        self.codebook[chosen_dead_codes] = z[hard_sample_indices]
        self.steps_since_hit[chosen_dead_codes] = 0

    def _contrastive_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self.contrastive_weight <= 0.0 or x.size(0) <= 1:
            return x.new_zeros(())
        if self.contrastive_warmup_only and self._is_codebook_ready():
            return x.new_zeros(())

        view_a = self._project(x, apply_feature_dropout=True)
        view_b = self._project(x, apply_feature_dropout=True)
        logits = (view_a @ view_b.t()) / max(self.contrastive_temperature, 1e-6)
        labels = torch.arange(x.size(0), device=x.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)
        )

    def _forward_impl(
            self,
            x: torch.Tensor,
            domain_ids: torch.Tensor,
            domain_context: Optional[torch.Tensor],
            *,
            compute_aux_losses: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self._project(x)
        matched_similarity = None

        if self.training and not self._is_codebook_ready():
            self._collect_warmup_vectors(z)

        if (not self.training) and (not self._is_codebook_ready()) and self._warmup_vector_count >= self.codebook_size:
            self._initialize_codebook_from_warmup(device=z.device)

        if self._is_codebook_ready():
            quantized, code_indices, matched_similarity = self._quantize(z)
            shared_z = z + (quantized - z).detach()
            specific_z = z - quantized.detach()
            commitment_loss = self.commitment_weight * F.mse_loss(z, quantized.detach())
            if self.training:
                self._ema_update_codebook(z.detach(), code_indices.detach())
        else:
            quantized = z.detach()
            code_indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
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
        self._latest_code_indices = code_indices.detach()
        self._latest_quantized_share = quantized.detach()
        self._latest_debug = {
            "code_indices": code_indices.detach(),
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
            "quantized_cos": (
                matched_similarity.detach()
                if matched_similarity is not None
                else torch.ones_like(shared_logits.detach())
            ),
        }
        return logits, commitment_loss, contrastive_loss

    def _record_debug_tensors(self, recorder) -> None:
        if recorder is None or not self._latest_debug:
            return

        code_indices = self._latest_debug["code_indices"]
        code_hist = F.one_hot(code_indices, num_classes=self.codebook_size).float()
        recorder.record("shavq_v1_code_usage", code_hist)
        recorder.record("shavq_v1_shared_hidden", self._latest_debug["shared_hidden"])
        recorder.record("shavq_v1_shared_feature_importance", self._latest_debug["shared_feature_importance"])
        recorder.record("shavq_v1_shared_logits", self._latest_debug["shared_logits"].unsqueeze(-1))
        recorder.record("shavq_v1_specific_logits_raw", self._latest_debug["specific_logits_raw"].unsqueeze(-1))
        recorder.record("shavq_v1_specific_logits", self._latest_debug["specific_logits"].unsqueeze(-1))
        recorder.record("shavq_v1_specific_hidden", self._latest_debug["specific_hidden"])
        recorder.record("shavq_v1_specific_feature_bias", self._latest_debug["specific_feature_bias"])
        recorder.record("shavq_v1_specific_feature_fluctuation", self._latest_debug["specific_feature_fluctuation"])
        recorder.record("shavq_v1_specific_feature_importance", self._latest_debug["specific_feature_importance"])
        recorder.record("shavq_v1_shared_norm", self._latest_debug["shared_norm"].unsqueeze(-1))
        recorder.record("shavq_v1_residual_norm", self._latest_debug["residual_norm"].unsqueeze(-1))
        recorder.record("shavq_v1_quantized_cos", self._latest_debug["quantized_cos"].unsqueeze(-1))

    def debug_state(self) -> Dict[str, float]:
        usage = self.code_usage.float()
        total_usage = float(usage.sum().item())
        used_codes = int((usage > 0).sum().item())
        probs = usage / usage.sum().clamp_min(1.0)
        entropy = float((-(probs[probs > 0] * probs[probs > 0].log()).sum()).item()) if used_codes > 0 else 0.0

        return {
            "codebook_initialized": float(self.codebook_initialized.item()),
            "used_codes": float(used_codes),
            "used_code_ratio": used_codes / max(1, self.codebook_size),
            "code_usage_entropy": entropy,
            "total_code_hits": total_usage,
            "latest_commitment_loss": float(self._latest_commitment_loss.item()) if self._latest_commitment_loss is not None else 0.0,
            "latest_contrastive_loss": float(self._latest_contrastive_loss.item()) if self._latest_contrastive_loss is not None else 0.0,
        }

    def contribution_state(self) -> Dict[str, float]:
        if not self._latest_debug:
            return {
                "shared_feature_var": 0.0,
                "specific_feature_var": 0.0,
                "shared_feature_abs_mean": 0.0,
                "specific_feature_abs_mean": 0.0,
                "shared_logit_var": 0.0,
                "specific_logit_var": 0.0,
                "shared_logit_abs_mean": 0.0,
                "specific_logit_abs_mean": 0.0,
                "feature_var_shared_ratio": 0.0,
                "logit_var_shared_ratio": 0.0,
            }

        shared_hidden = self._latest_debug["shared_hidden"].float()
        specific_feature = self._latest_debug["specific_feature_fluctuation"].float()
        shared_logits = self._latest_debug["shared_logits"].float()
        specific_logits = self._latest_debug["specific_logits"].float()

        shared_feature_var = float(shared_hidden.var(unbiased=False).item())
        specific_feature_var = float(specific_feature.var(unbiased=False).item())
        shared_feature_abs_mean = float(shared_hidden.abs().mean().item())
        specific_feature_abs_mean = float(specific_feature.abs().mean().item())

        shared_logit_var = float(shared_logits.var(unbiased=False).item())
        specific_logit_var = float(specific_logits.var(unbiased=False).item())
        shared_logit_abs_mean = float(shared_logits.abs().mean().item())
        specific_logit_abs_mean = float(specific_logits.abs().mean().item())

        feature_var_total = shared_feature_var + specific_feature_var + 1e-12
        logit_var_total = shared_logit_var + specific_logit_var + 1e-12

        return {
            "shared_feature_var": shared_feature_var,
            "specific_feature_var": specific_feature_var,
            "shared_feature_abs_mean": shared_feature_abs_mean,
            "specific_feature_abs_mean": specific_feature_abs_mean,
            "shared_logit_var": shared_logit_var,
            "specific_logit_var": specific_logit_var,
            "shared_logit_abs_mean": shared_logit_abs_mean,
            "specific_logit_abs_mean": specific_logit_abs_mean,
            "feature_var_shared_ratio": shared_feature_var / feature_var_total,
            "logit_var_shared_ratio": shared_logit_var / logit_var_total,
        }

    def predict(self, interaction):
        x, domain_ids, domain_context = self.encode_features(interaction)
        logits, _, _ = self._forward_impl(
            x,
            domain_ids,
            domain_context,
            compute_aux_losses=False,
        )
        return logits

    def calculate_loss(self, interaction):
        x, domain_ids, domain_context = self.encode_features(interaction)
        labels = interaction[self.LABEL].float().view(-1)
        logits, commitment_loss, contrastive_loss = self._forward_impl(
            x,
            domain_ids,
            domain_context,
            compute_aux_losses=self.training,
        )
        logits = logits.view(-1)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        return bce_loss + commitment_loss + self.contrastive_weight * contrastive_loss

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
                "[SHAVQ-v1 Recorder] "
                f"step={ctx.global_step + 1} "
                f"used={int(debug['used_codes'])}/{self.codebook_size} "
                f"usage_ratio={debug['used_code_ratio']:.3f} "
                f"entropy={debug['code_usage_entropy']:.3f} "
                f"commit={debug['latest_commitment_loss']:.5f} "
                f"contrast={debug['latest_contrastive_loss']:.5f} "
                f"feat_var(shared/specific)={contrib['shared_feature_var']:.4f}/{contrib['specific_feature_var']:.4f} "
                f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
                f"logit_var(shared/specific)={contrib['shared_logit_var']:.4f}/{contrib['specific_logit_var']:.4f} "
                f"logit_ratio={contrib['logit_var_shared_ratio']:.3f}"
            )
            print(ctx.recorder.get_window_stats())
        return float(loss.item())

    def on_train_epoch_start(self, ctx: TrainContext) -> None:
        del ctx

    def on_train_epoch_end(self, ctx: TrainContext) -> None:
        del ctx

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx: TrainContext) -> None:
        del metrics, ctx
        debug = self.debug_state()
        contrib = self.contribution_state()
        print(
            "[SHAVQ-v1 Debug] "
            f"ready={int(debug['codebook_initialized'])} "
            f"used={int(debug['used_codes'])}/{self.codebook_size} "
            f"usage_ratio={debug['used_code_ratio']:.3f} "
            f"entropy={debug['code_usage_entropy']:.3f} "
            f"commit={debug['latest_commitment_loss']:.5f} "
            f"contrast={debug['latest_contrastive_loss']:.5f} "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_ratio={contrib['logit_var_shared_ratio']:.3f}"
        )
