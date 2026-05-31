from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.train.context import TrainContext
from betterbole.experiment import WORKSPACE
from betterbole.models.msr.components import DomainTowerHead
from custom_models.shavq_v3.model import ProjectionEncoder, SHAVQV3Model
from betterbole.models.utils.common import to_dims
from betterbole.models.utils.general import FeatureBifurcator, MLP
from betterbole.utils.observatory import RelationOptions, TensorDisplayConfig, TensorMonitorOptions
from betterbole.utils.observatory.metrics import compute_spectrum
from betterbole.utils.observatory.plots import plot_multi_series, plot_ranked_profile, plot_step_dim_heatmap, plot_topk_bar, PlotSeries


class SHAVQV4Model(SHAVQV3Model):
    """
    SHAVQ-V4: integrated shared-prior + innovation-correction architecture.

    Core ideas:
    - shared branch and specific branch have separate encoders
    - shared branch keeps two-stage quantization and uses coarse/fine shared sub-experts
    - specific branch has two input channels:
      1) innovation channel over continuous specific representation and its deviation from shared prior
      2) context channel over detached shared contexts
    - final specific correction is confidence-modulated before adding to shared logits
    """

    def __init__(
            self,
            manager,
            num_domains: int,
            projection_hidden_dims: Iterable[int] = (256, 128),
            projection_dim: int = 64,
            specific_projection_dim: Optional[int] = None,
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
            stage2_scale: float = 0.75,
            shared_code_embed_dim: int = 16,
            shared_gate_temperature: float = 1.0,
            specific_gate_temperature: float = 1.0,
            specific_projection_init_gain: float = 1.0,
            specific_delta_init_gain: float = 2.0,
            specific_stage1_init_gain: float = 0.75,
            specific_stage2_init_gain: float = 0.5,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            projection_hidden_dims=projection_hidden_dims,
            projection_dim=projection_dim,
            projection_dropout_rate=projection_dropout_rate,
            shared_hidden_dims=shared_hidden_dims,
            shared_dropout_rate=shared_dropout_rate,
            shared_activation=shared_activation,
            shared_batch_norm=shared_batch_norm,
            specific_hidden_dims=specific_hidden_dims,
            specific_dropout_rate=specific_dropout_rate,
            specific_activation=specific_activation,
            specific_batch_norm=specific_batch_norm,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
            codebook_size=codebook_size,
            ema_decay=ema_decay,
            commitment_weight=commitment_weight,
            contrastive_weight=contrastive_weight,
            contrastive_temperature=contrastive_temperature,
            contrastive_dropout_rate=contrastive_dropout_rate,
            warmup_samples=warmup_samples,
            kmeans_iters=kmeans_iters,
            dead_code_threshold_steps=dead_code_threshold_steps,
            max_revived_codes_per_step=max_revived_codes_per_step,
            contrastive_warmup_only=contrastive_warmup_only,
            stage2_scale=stage2_scale,
        )

        self.specific_projection_dim = int(specific_projection_dim or projection_dim)
        self.specific_projection = ProjectionEncoder(
            input_dim=self.input_dim,
            hidden_dims=projection_hidden_dims,
            output_dim=self.specific_projection_dim,
            dropout_rate=projection_dropout_rate,
        )

        shared_hidden_dims = to_dims(shared_hidden_dims, (128, 64))
        self.shared_output_dim = int(shared_hidden_dims[-1])
        self.shared_code_embed_dim = int(shared_code_embed_dim)
        self.shared_gate_temperature = max(float(shared_gate_temperature), 1e-4)
        self.specific_gate_temperature = max(float(specific_gate_temperature), 1e-4)

        self.stage1_code_embedding = nn.Embedding(self.codebook_size, self.shared_code_embed_dim)
        self.stage2_code_embedding = nn.Embedding(self.codebook_size, self.shared_code_embed_dim)

        shared_stage1_input_dim = self.projection_dim + self.shared_code_embed_dim
        shared_stage2_input_dim = self.projection_dim + self.shared_code_embed_dim + 1
        self.shared_stage1_expert = MLP(
            shared_stage1_input_dim,
            *shared_hidden_dims,
            dropout_rate=shared_dropout_rate,
            activation=shared_activation,
            batch_norm=shared_batch_norm,
        )
        self.shared_stage2_expert = MLP(
            shared_stage2_input_dim,
            *shared_hidden_dims,
            dropout_rate=shared_dropout_rate,
            activation=shared_activation,
            batch_norm=shared_batch_norm,
        )
        self.shared_stage1_head = nn.Linear(self.shared_output_dim, 1)
        self.shared_stage2_head = nn.Linear(self.shared_output_dim, 1)

        shared_gate_input_dim = self.projection_dim * 2 + self.shared_code_embed_dim * 2 + 2
        shared_gate_hidden_dim = max(32, self.projection_dim)
        self.shared_gate = nn.Sequential(
            nn.Linear(shared_gate_input_dim, shared_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(shared_gate_hidden_dim, 2),
        )

        self.stage1_to_specific = nn.Linear(self.projection_dim, self.specific_projection_dim, bias=False)
        self.stage2_to_specific = nn.Linear(self.projection_dim, self.specific_projection_dim, bias=False)

        specific_hidden_dims = to_dims(specific_hidden_dims, (128, 64))
        self.specific_output_dim = int(specific_hidden_dims[-1])
        relation_feature_dim = 7

        innovation_input_dim = self.specific_projection_dim * 2 + relation_feature_dim
        context_input_dim = self.specific_projection_dim * 2 + relation_feature_dim + int(self.domain_context_dim)
        self.specific_innovation_expert = MLP(
            innovation_input_dim,
            *specific_hidden_dims,
            dropout_rate=specific_dropout_rate,
            activation=specific_activation,
            batch_norm=specific_batch_norm,
        )
        self.specific_context_expert = MLP(
            context_input_dim,
            *specific_hidden_dims,
            dropout_rate=specific_dropout_rate,
            activation=specific_activation,
            batch_norm=specific_batch_norm,
        )

        specific_gate_input_dim = relation_feature_dim + int(self.domain_context_dim)
        specific_gate_hidden_dim = max(32, self.specific_projection_dim)
        self.specific_gate = nn.Sequential(
            nn.Linear(specific_gate_input_dim, specific_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(specific_gate_hidden_dim, 2),
        )
        self.specific_confidence_head = nn.Sequential(
            nn.Linear(specific_gate_input_dim, specific_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(specific_gate_hidden_dim, 1),
        )
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

        self.specific_projection_gain = nn.Parameter(torch.tensor(float(specific_projection_init_gain)))
        self.specific_delta_gain = nn.Parameter(torch.tensor(float(specific_delta_init_gain)))
        self.specific_stage1_gain = nn.Parameter(torch.tensor(float(specific_stage1_init_gain)))
        self.specific_stage2_gain = nn.Parameter(torch.tensor(float(specific_stage2_init_gain)))
        self._observatory_initialized = False
        self._observatory_steps = []
        self._observatory_scalar_history = {}
        self.apply_xavier_initialization()

    def _observatory_output_dir(self, ctx: TrainContext) -> Path:
        model_name = getattr(ctx.cfg, "model", None)
        if model_name:
            output_name = str(model_name)
        else:
            output_name = str(getattr(ctx.cfg, "experiment_name", "shavq_v4"))
        return WORKSPACE / ctx.cfg.dataset_name / "observatory" / output_name

    def _setup_observatory(self, recorder) -> None:
        if recorder is None or self._observatory_initialized:
            return
        if not hasattr(recorder, "register"):
            return

        vector_options = TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=12,
                topk_display_dims=8,
                rank_by="variance",
            )
        )
        scalar_options = TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=False,
                max_display_dims=1,
                topk_display_dims=1,
            )
        )
        for name in [
            "shavq_v4_shared_hidden",
            "shavq_v4_specific_feature_fluctuation",
            "shavq_v4_specific_innovation_hidden_weighted",
            "shavq_v4_specific_context_hidden_weighted",
            "shavq_v4_shared_stage1_hidden_weighted",
            "shavq_v4_shared_stage2_hidden_weighted",
            "shavq_v4_shared_gate_weights",
            "shavq_v4_specific_gate_weights",
        ]:
            recorder.register(name, vector_options)
        for name in [
            "shavq_v4_specific_confidence",
            "shavq_v4_residual2_norm",
            "shavq_v4_total_cos",
        ]:
            recorder.register(name, scalar_options)
        if hasattr(recorder, "configure_relations"):
            recorder.configure_relations(
                RelationOptions(
                    enabled=True,
                    rank=8,
                    max_pairs=12,
                    names=(
                        "shavq_v4_shared_hidden",
                        "shavq_v4_specific_feature_fluctuation",
                        "shavq_v4_specific_innovation_hidden_weighted",
                        "shavq_v4_specific_context_hidden_weighted",
                        "shavq_v4_shared_stage1_hidden_weighted",
                        "shavq_v4_shared_stage2_hidden_weighted",
                    ),
                )
            )
        self._observatory_initialized = True

    def _append_observatory_scalars(self, step: int) -> None:
        if not self._latest_debug:
            return
        if self._observatory_steps and self._observatory_steps[-1] == int(step):
            return

        shared_spectrum = compute_spectrum(self._latest_debug["shared_hidden"])
        specific_spectrum = compute_spectrum(self._latest_debug["specific_feature_fluctuation"])
        innovation_spectrum = compute_spectrum(self._latest_debug["specific_innovation_hidden_weighted"])
        context_spectrum = compute_spectrum(self._latest_debug["specific_context_hidden_weighted"])
        contrib = self.contribution_state()

        scalar_values = {
            "shared_eff_rank": shared_spectrum.effective_rank,
            "specific_eff_rank": specific_spectrum.effective_rank,
            "innovation_eff_rank": innovation_spectrum.effective_rank,
            "context_eff_rank": context_spectrum.effective_rank,
            "shared_top1_energy": float((shared_spectrum.energy[:1].sum() / shared_spectrum.total_energy.clamp_min(1e-12)).item()) if shared_spectrum.energy.numel() > 0 else 0.0,
            "specific_top1_energy": float((specific_spectrum.energy[:1].sum() / specific_spectrum.total_energy.clamp_min(1e-12)).item()) if specific_spectrum.energy.numel() > 0 else 0.0,
            "shared_gate_entropy": float(contrib["shared_gate_entropy"]),
            "specific_gate_entropy": float(contrib["specific_gate_entropy"]),
            "specific_confidence_mean": float(contrib["specific_confidence_mean"]),
            "shared_hidden_var": float(contrib["shared_feature_var"]),
            "specific_fluctuation_var": float(contrib["specific_feature_var"]),
            "innovation_weighted_var": float(contrib["specific_innovation_hidden_weighted_var"]),
            "context_weighted_var": float(contrib["specific_context_hidden_weighted_var"]),
        }
        self._observatory_steps.append(int(step))
        for key, value in scalar_values.items():
            self._observatory_scalar_history.setdefault(key, []).append(float(value))

    @staticmethod
    def _topk_dim_var(feature_tensor: torch.Tensor, topk: int = 12):
        flat = feature_tensor.detach().float().reshape(feature_tensor.shape[0], -1)
        dim_var = flat.var(dim=0, unbiased=False).cpu()
        topk = min(int(topk), int(dim_var.numel()))
        if topk <= 0:
            return [], []
        top_indices = torch.topk(dim_var, k=topk).indices.sort().values
        top_values = dim_var[top_indices]
        return top_indices.tolist(), top_values.tolist()

    def _export_observatory_artifacts(self, ctx: TrainContext) -> None:
        recorder = ctx.recorder
        if recorder is None or not self._latest_debug:
            return

        output_dir = self._observatory_output_dir(ctx)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._observatory_steps:
            plot_multi_series(
                [
                    PlotSeries("shared", self._observatory_steps, self._observatory_scalar_history.get("shared_eff_rank", [])),
                    PlotSeries("specific", self._observatory_steps, self._observatory_scalar_history.get("specific_eff_rank", [])),
                    PlotSeries("innovation", self._observatory_steps, self._observatory_scalar_history.get("innovation_eff_rank", [])),
                    PlotSeries("context", self._observatory_steps, self._observatory_scalar_history.get("context_eff_rank", [])),
                ],
                title="shavq_v4_eff_rank_series",
                xlabel="step",
                ylabel="eff_rank",
                save_path=output_dir / "shavq_v4_eff_rank_series.png",
            )
            plot_multi_series(
                [
                    PlotSeries("shared_gate_entropy", self._observatory_steps, self._observatory_scalar_history.get("shared_gate_entropy", [])),
                    PlotSeries("specific_gate_entropy", self._observatory_steps, self._observatory_scalar_history.get("specific_gate_entropy", [])),
                    PlotSeries("specific_conf", self._observatory_steps, self._observatory_scalar_history.get("specific_confidence_mean", [])),
                ],
                title="shavq_v4_gate_series",
                xlabel="step",
                ylabel="value",
                save_path=output_dir / "shavq_v4_gate_series.png",
            )
            plot_multi_series(
                [
                    PlotSeries("shared_var", self._observatory_steps, self._observatory_scalar_history.get("shared_hidden_var", [])),
                    PlotSeries("specific_var", self._observatory_steps, self._observatory_scalar_history.get("specific_fluctuation_var", [])),
                    PlotSeries("innovation_var", self._observatory_steps, self._observatory_scalar_history.get("innovation_weighted_var", [])),
                    PlotSeries("context_var", self._observatory_steps, self._observatory_scalar_history.get("context_weighted_var", [])),
                ],
                title="shavq_v4_branch_var_series",
                xlabel="step",
                ylabel="feature_var",
                save_path=output_dir / "shavq_v4_branch_var_series.png",
            )
            plot_multi_series(
                [
                    PlotSeries("shared_top1", self._observatory_steps, self._observatory_scalar_history.get("shared_top1_energy", [])),
                    PlotSeries("specific_top1", self._observatory_steps, self._observatory_scalar_history.get("specific_top1_energy", [])),
                ],
                title="shavq_v4_collapse_series",
                xlabel="step",
                ylabel="top1_energy",
                save_path=output_dir / "shavq_v4_collapse_series.png",
            )

        feature_names = [
            ("shared_hidden", self._latest_debug["shared_hidden"]),
            ("specific_fluctuation", self._latest_debug["specific_feature_fluctuation"]),
            ("innovation_weighted", self._latest_debug["specific_innovation_hidden_weighted"]),
            ("context_weighted", self._latest_debug["specific_context_hidden_weighted"]),
        ]
        for short_name, tensor in feature_names:
            dim_idx, top_values = self._topk_dim_var(tensor, topk=12)
            if dim_idx:
                plot_topk_bar(
                    dim_idx=dim_idx,
                    values=top_values,
                    title="shavq_v4_%s_top_dim_var" % short_name,
                    ylabel="batch_var",
                    save_path=output_dir / ("shavq_v4_%s_top_dim_var.png" % short_name),
                )
            spectrum = compute_spectrum(tensor)
            if spectrum.singular_values.numel() > 0:
                plot_ranked_profile(
                    values=spectrum.singular_values.detach().cpu().numpy(),
                    title="shavq_v4_%s_singular_values" % short_name,
                    ylabel="singular_value",
                    save_path=output_dir / ("shavq_v4_%s_singular_values.png" % short_name),
                    descending=True,
                )

        _, shared_step_dim = recorder.get_step_dim_matrix("shavq_v4_shared_hidden", "batch_var")
        if shared_step_dim.size > 0:
            plot_step_dim_heatmap(
                values=shared_step_dim,
                title="shavq_v4_shared_hidden_step_dim_var",
                save_path=output_dir / "shavq_v4_shared_hidden_step_dim_var.png",
            )
        _, specific_step_dim = recorder.get_step_dim_matrix("shavq_v4_specific_feature_fluctuation", "batch_var")
        if specific_step_dim.size > 0:
            plot_step_dim_heatmap(
                values=specific_step_dim,
                title="shavq_v4_specific_fluctuation_step_dim_var",
                save_path=output_dir / "shavq_v4_specific_fluctuation_step_dim_var.png",
            )


    def _forward_impl(
            self,
            x: torch.Tensor,
            domain_ids: torch.Tensor,
            domain_context: Optional[torch.Tensor],
            *,
            compute_aux_losses: bool,
    ):
        z_shared = self._project(x)
        z_specific = self.specific_projection(x)

        if self.training and not self._is_stage1_ready():
            self._collect_stage1_warmup_vectors(z_shared)
        if (not self.training) and (not self._is_stage1_ready()) and self._stage1_warmup_vector_count >= self.codebook_size:
            self._initialize_stage1_codebook(device=z_shared.device)

        stage1_indices = torch.zeros(z_shared.size(0), dtype=torch.long, device=z_shared.device)
        stage2_indices = torch.zeros(z_shared.size(0), dtype=torch.long, device=z_shared.device)
        stage1_cos = torch.ones(z_shared.size(0), dtype=z_shared.dtype, device=z_shared.device)
        stage2_cos = torch.zeros(z_shared.size(0), dtype=z_shared.dtype, device=z_shared.device)
        quantized_stage1 = z_shared.detach()
        quantized_stage2 = torch.zeros_like(z_shared)
        residual1 = torch.zeros_like(z_shared)
        residual2 = torch.zeros_like(z_shared)
        residual1_norm = torch.zeros(z_shared.size(0), dtype=z_shared.dtype, device=z_shared.device)
        residual2_norm = torch.zeros(z_shared.size(0), dtype=z_shared.dtype, device=z_shared.device)
        residual_dir = torch.zeros_like(z_shared)
        shared_quantized = z_shared.detach()
        commitment_loss = z_shared.new_zeros(())

        if self._is_stage1_ready():
            quantized_stage1, stage1_indices, stage1_cos = self._quantize_stage1(z_shared)
            residual1 = z_shared - quantized_stage1.detach()
            residual1_norm = residual1.norm(dim=-1)

            if self.training and not self._is_stage2_ready():
                self._collect_stage2_warmup_vectors(residual1)
            if (not self.training) and (not self._is_stage2_ready()) and self._stage2_warmup_vector_count >= self.codebook_size:
                self._initialize_stage2_codebook(device=z_shared.device)

            shared_quantized = quantized_stage1
            residual2 = residual1
            residual2_norm = residual2.norm(dim=-1)

            if self._is_stage2_ready():
                quantized_stage2, stage2_indices, stage2_cos, residual1_norm, residual_dir = self._quantize_stage2(residual1)
                shared_quantized = quantized_stage1 + quantized_stage2
                residual2 = z_shared - shared_quantized.detach()
                residual2_norm = residual2.norm(dim=-1)
                valid_mask = residual1_norm > 1e-6
                if self.training:
                    self._ema_update_stage2_codebook(residual_dir.detach(), stage2_indices.detach(), valid_mask.detach())

            commitment_loss = self.commitment_weight * F.mse_loss(z_shared, shared_quantized.detach())
            if self.training:
                self._ema_update_stage1_codebook(z_shared.detach(), stage1_indices.detach())

        stage1_code_embed = self.stage1_code_embedding(stage1_indices)
        stage2_code_embed = self.stage2_code_embedding(stage2_indices)
        residual_norm_features = torch.stack([residual1_norm, residual2_norm], dim=-1)

        shared_stage1_input = torch.cat([quantized_stage1, stage1_code_embed], dim=-1)
        shared_stage2_input = torch.cat([quantized_stage2, stage2_code_embed, residual1_norm.unsqueeze(-1)], dim=-1)
        shared_stage1_hidden = self.shared_stage1_expert(shared_stage1_input)
        shared_stage2_hidden = self.shared_stage2_expert(shared_stage2_input)

        shared_gate_input = torch.cat(
            [quantized_stage1, quantized_stage2, stage1_code_embed, stage2_code_embed, residual_norm_features],
            dim=-1,
        )
        shared_gate_logits = self.shared_gate(shared_gate_input)
        shared_gate_weights = F.softmax(shared_gate_logits / self.shared_gate_temperature, dim=-1)

        shared_stage1_hidden_weighted = shared_gate_weights[:, 0:1] * shared_stage1_hidden
        shared_stage2_hidden_weighted = shared_gate_weights[:, 1:2] * shared_stage2_hidden
        shared_hidden = shared_stage1_hidden_weighted + shared_stage2_hidden_weighted

        shared_stage1_logits_raw = self.shared_stage1_head(shared_stage1_hidden).squeeze(-1)
        shared_stage2_logits_raw = self.shared_stage2_head(shared_stage2_hidden).squeeze(-1)
        shared_stage1_logits = shared_gate_weights[:, 0] * shared_stage1_logits_raw
        shared_stage2_logits = shared_gate_weights[:, 1] * shared_stage2_logits_raw
        shared_logits = shared_stage1_logits + shared_stage2_logits

        aligned_stage1 = self.stage1_to_specific(quantized_stage1.detach())
        aligned_stage2 = self.stage2_to_specific(quantized_stage2.detach())
        shared_prior_specific = aligned_stage1 + aligned_stage2
        innovation_delta = z_specific - shared_prior_specific

        specific_stage1_cos = (F.normalize(z_specific, dim=-1) * F.normalize(aligned_stage1, dim=-1)).sum(dim=-1)
        valid_stage2 = aligned_stage2.norm(dim=-1) > 1e-6
        specific_stage2_cos = torch.zeros_like(specific_stage1_cos)
        if valid_stage2.any():
            specific_stage2_cos[valid_stage2] = (
                F.normalize(z_specific[valid_stage2], dim=-1) * F.normalize(aligned_stage2[valid_stage2], dim=-1)
            ).sum(dim=-1)

        total_cos = (F.normalize(shared_quantized, dim=-1) * z_shared).sum(dim=-1) if self._is_stage1_ready() else torch.ones_like(shared_logits)
        relation_features = torch.stack(
            [
                residual1_norm,
                residual2_norm,
                total_cos,
                specific_stage1_cos,
                specific_stage2_cos,
                shared_gate_weights[:, 0].detach(),
                shared_gate_weights[:, 1].detach(),
            ],
            dim=-1,
        )

        scaled_specific_projection = self.specific_projection_gain * z_specific
        scaled_innovation_delta = self.specific_delta_gain * innovation_delta
        scaled_stage1_context = self.specific_stage1_gain * aligned_stage1
        scaled_stage2_context = self.specific_stage2_gain * aligned_stage2

        innovation_input = torch.cat(
            [scaled_specific_projection, scaled_innovation_delta, relation_features],
            dim=-1,
        )
        context_inputs = [scaled_stage1_context, scaled_stage2_context, relation_features]
        if domain_context is not None:
            context_inputs.append(domain_context)
        context_input = torch.cat(context_inputs, dim=-1)

        specific_gate_inputs = [relation_features]
        if domain_context is not None:
            specific_gate_inputs.append(domain_context)
        specific_gate_input = torch.cat(specific_gate_inputs, dim=-1)

        specific_innovation_hidden = self.specific_innovation_expert(innovation_input)
        specific_context_hidden = self.specific_context_expert(context_input)
        specific_gate_logits = self.specific_gate(specific_gate_input)
        specific_gate_weights = F.softmax(specific_gate_logits / self.specific_gate_temperature, dim=-1)
        specific_confidence = torch.sigmoid(self.specific_confidence_head(specific_gate_input)).squeeze(-1)

        specific_innovation_hidden_weighted = specific_gate_weights[:, 0:1] * specific_innovation_hidden
        specific_context_hidden_weighted = specific_gate_weights[:, 1:2] * specific_context_hidden
        specific_hidden = specific_innovation_hidden_weighted + specific_context_hidden_weighted

        specific_bias_feature, specific_fluctuation_feature = self.specific_bifurcator(specific_hidden)
        specific_logits_raw = self.specific_head(specific_hidden, domain_ids)
        specific_logits_base = self.specific_head(specific_fluctuation_feature, domain_ids)
        specific_logits = specific_confidence * specific_logits_base
        logits = shared_logits + specific_logits

        contrastive_loss = self._contrastive_loss(x) if compute_aux_losses else logits.new_zeros(())

        self._latest_commitment_loss = commitment_loss.detach()
        self._latest_contrastive_loss = contrastive_loss.detach()
        self._latest_debug = {
            "domain_ids": domain_ids.detach(),
            "stage1_indices": stage1_indices.detach(),
            "stage2_indices": stage2_indices.detach(),
            "stage2_valid_mask": (residual1_norm > 1e-6).detach(),
            "shared_hidden": shared_hidden.detach(),
            "shared_logits": shared_logits.detach(),
            "shared_feature_importance": shared_hidden.detach().abs(),
            "shared_stage1_hidden": shared_stage1_hidden.detach(),
            "shared_stage2_hidden": shared_stage2_hidden.detach(),
            "shared_stage1_hidden_weighted": shared_stage1_hidden_weighted.detach(),
            "shared_stage2_hidden_weighted": shared_stage2_hidden_weighted.detach(),
            "shared_gate_weights": shared_gate_weights.detach(),
            "shared_stage1_logits_raw": shared_stage1_logits_raw.detach(),
            "shared_stage2_logits_raw": shared_stage2_logits_raw.detach(),
            "shared_stage1_logits": shared_stage1_logits.detach(),
            "shared_stage2_logits": shared_stage2_logits.detach(),
            "specific_logits_raw": specific_logits_raw.detach(),
            "specific_logits_base": specific_logits_base.detach(),
            "specific_logits": specific_logits.detach(),
            "specific_hidden": specific_hidden.detach(),
            "specific_feature_bias": specific_bias_feature.detach(),
            "specific_feature_fluctuation": specific_fluctuation_feature.detach(),
            "specific_feature_importance": specific_fluctuation_feature.detach().abs(),
            "specific_innovation_hidden": specific_innovation_hidden.detach(),
            "specific_context_hidden": specific_context_hidden.detach(),
            "specific_innovation_hidden_weighted": specific_innovation_hidden_weighted.detach(),
            "specific_context_hidden_weighted": specific_context_hidden_weighted.detach(),
            "specific_gate_weights": specific_gate_weights.detach(),
            "specific_confidence": specific_confidence.detach(),
            "specific_gate_input": specific_gate_input.detach(),
            "innovation_input": innovation_input.detach(),
            "context_input": context_input.detach(),
            "z_specific": z_specific.detach(),
            "innovation_delta": innovation_delta.detach(),
            "shared_prior_specific": shared_prior_specific.detach(),
            "aligned_stage1": aligned_stage1.detach(),
            "aligned_stage2": aligned_stage2.detach(),
            "specific_relation_features": relation_features.detach(),
            "quantized_stage1": quantized_stage1.detach(),
            "quantized_stage2": quantized_stage2.detach(),
            "shared_norm": shared_quantized.detach().norm(dim=-1),
            "residual1_norm": residual1_norm.detach(),
            "residual2_norm": residual2_norm.detach(),
            "stage1_cos": stage1_cos.detach(),
            "stage2_cos": stage2_cos.detach(),
            "total_cos": total_cos.detach(),
        }
        return logits, commitment_loss, contrastive_loss

    def _record_debug_tensors(self, recorder, step: Optional[int] = None) -> None:
        if recorder is None or not self._latest_debug:
            return
        self._setup_observatory(recorder)

        stage1_indices = self._latest_debug["stage1_indices"]
        stage2_indices = self._latest_debug["stage2_indices"]
        stage2_valid_mask = self._latest_debug["stage2_valid_mask"]

        recorder.record("shavq_v4_stage1_code_usage", F.one_hot(stage1_indices, num_classes=self.codebook_size).float(), step=step)
        stage2_hist = F.one_hot(stage2_indices, num_classes=self.codebook_size).float()
        stage2_hist = stage2_hist * stage2_valid_mask.unsqueeze(-1).float()
        recorder.record("shavq_v4_stage2_code_usage", stage2_hist, step=step)

        for key in [
            "quantized_stage1",
            "quantized_stage2",
            "shared_stage1_hidden",
            "shared_stage2_hidden",
            "shared_stage1_hidden_weighted",
            "shared_stage2_hidden_weighted",
            "shared_hidden",
            "shared_feature_importance",
            "shared_gate_weights",
            "specific_hidden",
            "specific_feature_bias",
            "specific_feature_fluctuation",
            "specific_feature_importance",
            "specific_innovation_hidden",
            "specific_context_hidden",
            "specific_innovation_hidden_weighted",
            "specific_context_hidden_weighted",
            "specific_gate_weights",
            "specific_gate_input",
            "innovation_input",
            "context_input",
            "z_specific",
            "innovation_delta",
            "shared_prior_specific",
            "aligned_stage1",
            "aligned_stage2",
            "specific_relation_features",
        ]:
            recorder.record("shavq_v4_" + key, self._latest_debug[key], step=step)

        for key in [
            "shared_stage1_logits_raw",
            "shared_stage2_logits_raw",
            "shared_stage1_logits",
            "shared_stage2_logits",
            "shared_logits",
            "specific_logits_raw",
            "specific_logits_base",
            "specific_logits",
            "specific_confidence",
            "shared_norm",
            "residual1_norm",
            "residual2_norm",
            "stage1_cos",
            "stage2_cos",
            "total_cos",
        ]:
            recorder.record("shavq_v4_" + key, self._latest_debug[key].unsqueeze(-1), step=step)

    def contribution_state(self) -> Dict[str, float]:
        state = super().contribution_state()
        if not self._latest_debug:
            state.update({
                "shared_gate_entropy": 0.0,
                "specific_gate_entropy": 0.0,
                "shared_gate_stage1_mean": 0.0,
                "shared_gate_stage2_mean": 0.0,
                "specific_gate_innovation_mean": 0.0,
                "specific_gate_context_mean": 0.0,
                "specific_confidence_mean": 0.0,
                "specific_confidence_var": 0.0,
                "specific_projection_var": 0.0,
                "innovation_delta_var": 0.0,
                "shared_prior_specific_var": 0.0,
                "shared_stage1_hidden_weighted_var": 0.0,
                "shared_stage2_hidden_weighted_var": 0.0,
                "specific_innovation_hidden_weighted_var": 0.0,
                "specific_context_hidden_weighted_var": 0.0,
                "specific_projection_gain": float(self.specific_projection_gain.detach().item()),
                "specific_delta_gain": float(self.specific_delta_gain.detach().item()),
                "specific_stage1_gain": float(self.specific_stage1_gain.detach().item()),
                "specific_stage2_gain": float(self.specific_stage2_gain.detach().item()),
            })
            return state

        shared_gate_weights = self._latest_debug["shared_gate_weights"].float()
        specific_gate_weights = self._latest_debug["specific_gate_weights"].float()
        specific_confidence = self._latest_debug["specific_confidence"].float()
        z_specific = self._latest_debug["z_specific"].float()
        innovation_delta = self._latest_debug["innovation_delta"].float()
        shared_prior_specific = self._latest_debug["shared_prior_specific"].float()
        shared_stage1_hidden_weighted = self._latest_debug["shared_stage1_hidden_weighted"].float()
        shared_stage2_hidden_weighted = self._latest_debug["shared_stage2_hidden_weighted"].float()
        specific_innovation_hidden_weighted = self._latest_debug["specific_innovation_hidden_weighted"].float()
        specific_context_hidden_weighted = self._latest_debug["specific_context_hidden_weighted"].float()
        specific_relation_features = self._latest_debug["specific_relation_features"].float()
        specific_logits_base = self._latest_debug["specific_logits_base"].float()

        shared_gate_probs = shared_gate_weights.clamp_min(1e-12)
        specific_gate_probs = specific_gate_weights.clamp_min(1e-12)
        shared_gate_entropy = float((-(shared_gate_probs * shared_gate_probs.log()).sum(dim=-1).mean()).item())
        specific_gate_entropy = float((-(specific_gate_probs * specific_gate_probs.log()).sum(dim=-1).mean()).item())

        state.update({
            "shared_gate_entropy": shared_gate_entropy,
            "specific_gate_entropy": specific_gate_entropy,
            "shared_gate_stage1_mean": float(shared_gate_weights[:, 0].mean().item()),
            "shared_gate_stage2_mean": float(shared_gate_weights[:, 1].mean().item()),
            "specific_gate_innovation_mean": float(specific_gate_weights[:, 0].mean().item()),
            "specific_gate_context_mean": float(specific_gate_weights[:, 1].mean().item()),
            "specific_confidence_mean": float(specific_confidence.mean().item()),
            "specific_confidence_var": float(specific_confidence.var(unbiased=False).item()),
            "specific_projection_var": float(z_specific.var(unbiased=False).item()),
            "innovation_delta_var": float(innovation_delta.var(unbiased=False).item()),
            "shared_prior_specific_var": float(shared_prior_specific.var(unbiased=False).item()),
            "shared_stage1_hidden_weighted_var": float(shared_stage1_hidden_weighted.var(unbiased=False).item()),
            "shared_stage2_hidden_weighted_var": float(shared_stage2_hidden_weighted.var(unbiased=False).item()),
            "specific_innovation_hidden_weighted_var": float(specific_innovation_hidden_weighted.var(unbiased=False).item()),
            "specific_context_hidden_weighted_var": float(specific_context_hidden_weighted.var(unbiased=False).item()),
            "specific_relation_var": float(specific_relation_features.var(unbiased=False).item()),
            "specific_logits_base_var": float(specific_logits_base.var(unbiased=False).item()),
            "specific_projection_gain": float(self.specific_projection_gain.detach().item()),
            "specific_delta_gain": float(self.specific_delta_gain.detach().item()),
            "specific_stage1_gain": float(self.specific_stage1_gain.detach().item()),
            "specific_stage2_gain": float(self.specific_stage2_gain.detach().item()),
        })
        return state

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        loss = self.calculate_loss(batch_interaction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        self._record_debug_tensors(ctx.recorder, step=ctx.global_step + 1)
        self._append_observatory_scalars(step=ctx.global_step + 1)
        if ctx.recorder is not None and (ctx.global_step + 1) % 200 == 0:
            debug = self.debug_state()
            contrib = self.contribution_state()
            print(
                "[SHAVQ-v4 Recorder] "
                f"step={ctx.global_step + 1} "
                f"s1={int(debug['stage1_used_codes'])}/{self.codebook_size} "
                f"s2={int(debug['stage2_used_codes'])}/{self.codebook_size} "
                f"ent1={debug['stage1_entropy']:.3f} "
                f"ent2={debug['stage2_entropy']:.3f} "
                f"sg=({contrib['shared_gate_stage1_mean']:.3f},{contrib['shared_gate_stage2_mean']:.3f}) "
                f"cg=({contrib['specific_gate_innovation_mean']:.3f},{contrib['specific_gate_context_mean']:.3f}) "
                f"alpha={contrib['specific_confidence_mean']:.3f} "
                f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
                f"logit_ratio={contrib['logit_var_shared_ratio']:.3f} "
                f"delta_var={contrib['innovation_delta_var']:.4f} "
                f"r2={contrib['residual2_norm_mean']:.3f} "
                f"total_cos={contrib['total_cos_mean']:.3f}"
            )
            print(
                ctx.recorder.get_window_stats(
                    names=[
                        "shavq_v4_shared_hidden",
                        "shavq_v4_specific_feature_fluctuation",
                        "shavq_v4_specific_innovation_hidden_weighted",
                        "shavq_v4_specific_context_hidden_weighted",
                        "shavq_v4_shared_gate_weights",
                        "shavq_v4_specific_gate_weights",
                    ],
                    include_relations=True,
                    relation_names=[
                        "shavq_v4_shared_hidden",
                        "shavq_v4_specific_feature_fluctuation",
                        "shavq_v4_specific_innovation_hidden_weighted",
                        "shavq_v4_specific_context_hidden_weighted",
                        "shavq_v4_shared_stage1_hidden_weighted",
                        "shavq_v4_shared_stage2_hidden_weighted",
                    ],
                )
            )
        return float(loss.item())

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx: TrainContext) -> None:
        del metrics
        debug = self.debug_state()
        contrib = self.contribution_state()
        print(
            "[SHAVQ-v4 Debug] "
            f"ready1={int(debug['stage1_initialized'])} "
            f"ready2={int(debug['stage2_initialized'])} "
            f"s1={int(debug['stage1_used_codes'])}/{self.codebook_size} "
            f"s2={int(debug['stage2_used_codes'])}/{self.codebook_size} "
            f"ent1={debug['stage1_entropy']:.3f} "
            f"ent2={debug['stage2_entropy']:.3f} "
            f"sg=({contrib['shared_gate_stage1_mean']:.3f},{contrib['shared_gate_stage2_mean']:.3f}) "
            f"cg=({contrib['specific_gate_innovation_mean']:.3f},{contrib['specific_gate_context_mean']:.3f}) "
            f"alpha={contrib['specific_confidence_mean']:.3f} "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_ratio={contrib['logit_var_shared_ratio']:.3f} "
            f"delta_var={contrib['innovation_delta_var']:.4f} "
            f"r2={contrib['residual2_norm_mean']:.3f} "
            f"total_cos={contrib['total_cos_mean']:.3f}"
        )
        if ctx.recorder is not None:
            print(
                ctx.recorder.get_window_stats(
                    names=[
                        "shavq_v4_shared_hidden",
                        "shavq_v4_specific_feature_fluctuation",
                        "shavq_v4_specific_innovation_hidden_weighted",
                        "shavq_v4_specific_context_hidden_weighted",
                        "shavq_v4_shared_gate_weights",
                        "shavq_v4_specific_gate_weights",
                    ],
                    include_relations=True,
                    relation_names=[
                        "shavq_v4_shared_hidden",
                        "shavq_v4_specific_feature_fluctuation",
                        "shavq_v4_specific_innovation_hidden_weighted",
                        "shavq_v4_specific_context_hidden_weighted",
                        "shavq_v4_shared_stage1_hidden_weighted",
                        "shavq_v4_shared_stage2_hidden_weighted",
                    ],
                )
            )
            self._export_observatory_artifacts(ctx)
