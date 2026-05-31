from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn

from betterbole.core.train.context import TrainContext
from custom_models.shavq_v3.model import SHAVQV3Model
from betterbole.models.utils.common import to_dims
from betterbole.models.utils.general import FeatureBifurcator, MLP


class SHAVQV3BModel(SHAVQV3Model):
    """
    SHAVQ-V3B: stage-wise shared residual quantization with conditioned specific branch.

    Specific branch consumes:
    - scaled final residual
    - detached shared quantized context
    - residual norm summary
    - domain context
    """

    def __init__(
            self,
            manager,
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
            stage2_scale: float = 1.0,
            specific_residual_init_gain: float = 2.0,
            specific_shared_init_gain: float = 1.0,
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

        specific_hidden_dims = to_dims(specific_hidden_dims, (128, 64))
        specific_input_dim = self.projection_dim * 2 + int(self.domain_context_dim) + 2
        self.specific_expert = MLP(
            specific_input_dim,
            *specific_hidden_dims,
            dropout_rate=specific_dropout_rate,
            activation=specific_activation,
            batch_norm=specific_batch_norm,
        )
        self.specific_output_dim = int(specific_hidden_dims[-1])
        self.specific_bifurcator = FeatureBifurcator(
            self.specific_output_dim,
            feature_dim=-1,
            mean_dims=0,
            normalize_var=False,
        )

        self.specific_residual_gain = nn.Parameter(torch.tensor(float(specific_residual_init_gain)))
        self.specific_shared_gain = nn.Parameter(torch.tensor(float(specific_shared_init_gain)))

    def _forward_impl(
            self,
            x: torch.Tensor,
            domain_ids: torch.Tensor,
            domain_context: Optional[torch.Tensor],
            *,
            compute_aux_losses: bool,
    ):
        z = self._project(x)

        if self.training and not self._is_stage1_ready():
            self._collect_stage1_warmup_vectors(z)
        if (not self.training) and (not self._is_stage1_ready()) and self._stage1_warmup_vector_count >= self.codebook_size:
            self._initialize_stage1_codebook(device=z.device)

        stage1_indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        stage2_indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        stage1_cos = torch.ones(z.size(0), dtype=z.dtype, device=z.device)
        stage2_cos = torch.zeros(z.size(0), dtype=z.dtype, device=z.device)
        quantized_stage1 = z.detach()
        quantized_stage2 = torch.zeros_like(z)
        residual1 = torch.zeros_like(z)
        residual2 = torch.zeros_like(z)
        residual1_norm = torch.zeros(z.size(0), dtype=z.dtype, device=z.device)
        residual2_norm = torch.zeros(z.size(0), dtype=z.dtype, device=z.device)
        residual_dir = torch.zeros_like(z)
        shared_quantized = z.detach()
        commitment_loss = z.new_zeros(())

        if self._is_stage1_ready():
            quantized_stage1, stage1_indices, stage1_cos = self._quantize_stage1(z)
            residual1 = z - quantized_stage1.detach()
            residual1_norm = residual1.norm(dim=-1)

            if self.training and not self._is_stage2_ready():
                self._collect_stage2_warmup_vectors(residual1)
            if (not self.training) and (not self._is_stage2_ready()) and self._stage2_warmup_vector_count >= self.codebook_size:
                self._initialize_stage2_codebook(device=z.device)

            shared_quantized = quantized_stage1
            residual2 = residual1
            residual2_norm = residual2.norm(dim=-1)

            if self._is_stage2_ready():
                quantized_stage2, stage2_indices, stage2_cos, residual1_norm, residual_dir = self._quantize_stage2(residual1)
                shared_quantized = quantized_stage1 + quantized_stage2
                residual2 = z - shared_quantized.detach()
                residual2_norm = residual2.norm(dim=-1)
                valid_mask = residual1_norm > 1e-6
                if self.training:
                    self._ema_update_stage2_codebook(residual_dir.detach(), stage2_indices.detach(), valid_mask.detach())

            commitment_loss = self.commitment_weight * torch.nn.functional.mse_loss(z, shared_quantized.detach())
            if self.training:
                self._ema_update_stage1_codebook(z.detach(), stage1_indices.detach())

        shared_z = z + (shared_quantized - z).detach()
        specific_z = residual2

        shared_hidden = self.shared_expert(shared_z)
        shared_logits = self.shared_head(shared_hidden).squeeze(-1)

        shared_context = shared_quantized.detach()
        scaled_specific_residual = self.specific_residual_gain * specific_z
        scaled_shared_context = self.specific_shared_gain * shared_context
        residual_norm_features = torch.stack([residual1_norm, residual2_norm], dim=-1)

        specific_inputs = [scaled_specific_residual, scaled_shared_context, residual_norm_features]
        if domain_context is not None:
            specific_inputs.append(domain_context)
        specific_input = torch.cat(specific_inputs, dim=-1)

        specific_hidden = self.specific_expert(specific_input)
        specific_bias_feature, specific_fluctuation_feature = self.specific_bifurcator(specific_hidden)
        specific_logits_raw = self.specific_head(specific_hidden, domain_ids)
        specific_logits = self.specific_head(specific_fluctuation_feature, domain_ids)
        logits = shared_logits + specific_logits

        contrastive_loss = self._contrastive_loss(x) if compute_aux_losses else logits.new_zeros(())
        total_cos = (torch.nn.functional.normalize(shared_quantized, dim=-1) * z).sum(dim=-1) if self._is_stage1_ready() else torch.ones_like(shared_logits)

        self._latest_commitment_loss = commitment_loss.detach()
        self._latest_contrastive_loss = contrastive_loss.detach()
        self._latest_debug = {
            "stage1_indices": stage1_indices.detach(),
            "stage2_indices": stage2_indices.detach(),
            "stage2_valid_mask": (residual1_norm > 1e-6).detach(),
            "shared_hidden": shared_hidden.detach(),
            "shared_logits": shared_logits.detach(),
            "shared_feature_importance": shared_hidden.detach().abs(),
            "specific_logits_raw": specific_logits_raw.detach(),
            "specific_logits": specific_logits.detach(),
            "specific_hidden": specific_hidden.detach(),
            "specific_feature_bias": specific_bias_feature.detach(),
            "specific_feature_fluctuation": specific_fluctuation_feature.detach(),
            "specific_feature_importance": specific_fluctuation_feature.detach().abs(),
            "quantized_stage1": quantized_stage1.detach(),
            "quantized_stage2": quantized_stage2.detach(),
            "shared_norm": shared_z.detach().norm(dim=-1),
            "residual1_norm": residual1_norm.detach(),
            "residual2_norm": residual2_norm.detach(),
            "stage1_cos": stage1_cos.detach(),
            "stage2_cos": stage2_cos.detach(),
            "total_cos": total_cos.detach(),
            "specific_residual_raw": specific_z.detach(),
            "specific_residual_scaled": scaled_specific_residual.detach(),
            "specific_shared_context": shared_context.detach(),
            "specific_shared_context_scaled": scaled_shared_context.detach(),
            "specific_input": specific_input.detach(),
        }
        return logits, commitment_loss, contrastive_loss

    def _record_debug_tensors(self, recorder) -> None:
        if recorder is None or not self._latest_debug:
            return

        stage1_indices = self._latest_debug["stage1_indices"]
        stage2_indices = self._latest_debug["stage2_indices"]
        stage2_valid_mask = self._latest_debug["stage2_valid_mask"]

        recorder.record("shavq_v3b_stage1_code_usage", torch.nn.functional.one_hot(stage1_indices, num_classes=self.codebook_size).float())
        stage2_hist = torch.nn.functional.one_hot(stage2_indices, num_classes=self.codebook_size).float()
        stage2_hist = stage2_hist * stage2_valid_mask.unsqueeze(-1).float()
        recorder.record("shavq_v3b_stage2_code_usage", stage2_hist)

        recorder.record("shavq_v3b_quantized_stage1", self._latest_debug["quantized_stage1"])
        recorder.record("shavq_v3b_quantized_stage2", self._latest_debug["quantized_stage2"])
        recorder.record("shavq_v3b_shared_hidden", self._latest_debug["shared_hidden"])
        recorder.record("shavq_v3b_shared_feature_importance", self._latest_debug["shared_feature_importance"])
        recorder.record("shavq_v3b_shared_logits", self._latest_debug["shared_logits"].unsqueeze(-1))
        recorder.record("shavq_v3b_specific_logits_raw", self._latest_debug["specific_logits_raw"].unsqueeze(-1))
        recorder.record("shavq_v3b_specific_logits", self._latest_debug["specific_logits"].unsqueeze(-1))
        recorder.record("shavq_v3b_specific_hidden", self._latest_debug["specific_hidden"])
        recorder.record("shavq_v3b_specific_feature_bias", self._latest_debug["specific_feature_bias"])
        recorder.record("shavq_v3b_specific_feature_fluctuation", self._latest_debug["specific_feature_fluctuation"])
        recorder.record("shavq_v3b_specific_feature_importance", self._latest_debug["specific_feature_importance"])
        recorder.record("shavq_v3b_specific_residual_raw", self._latest_debug["specific_residual_raw"])
        recorder.record("shavq_v3b_specific_residual_scaled", self._latest_debug["specific_residual_scaled"])
        recorder.record("shavq_v3b_specific_shared_context", self._latest_debug["specific_shared_context"])
        recorder.record("shavq_v3b_specific_shared_context_scaled", self._latest_debug["specific_shared_context_scaled"])
        recorder.record("shavq_v3b_specific_input", self._latest_debug["specific_input"])
        recorder.record("shavq_v3b_shared_norm", self._latest_debug["shared_norm"].unsqueeze(-1))
        recorder.record("shavq_v3b_residual1_norm", self._latest_debug["residual1_norm"].unsqueeze(-1))
        recorder.record("shavq_v3b_residual2_norm", self._latest_debug["residual2_norm"].unsqueeze(-1))
        recorder.record("shavq_v3b_stage1_cos", self._latest_debug["stage1_cos"].unsqueeze(-1))
        recorder.record("shavq_v3b_stage2_cos", self._latest_debug["stage2_cos"].unsqueeze(-1))
        recorder.record("shavq_v3b_total_cos", self._latest_debug["total_cos"].unsqueeze(-1))

    def contribution_state(self) -> Dict[str, float]:
        state = super().contribution_state()
        if not self._latest_debug:
            state.update({
                "specific_residual_raw_var": 0.0,
                "specific_residual_scaled_var": 0.0,
                "specific_shared_context_var": 0.0,
                "specific_shared_context_scaled_var": 0.0,
                "specific_input_var": 0.0,
                "specific_residual_condition_ratio": 0.0,
                "specific_residual_gain": float(self.specific_residual_gain.detach().item()),
                "specific_shared_gain": float(self.specific_shared_gain.detach().item()),
            })
            return state

        residual_raw = self._latest_debug["specific_residual_raw"].float()
        residual_scaled = self._latest_debug["specific_residual_scaled"].float()
        shared_context = self._latest_debug["specific_shared_context"].float()
        shared_context_scaled = self._latest_debug["specific_shared_context_scaled"].float()
        specific_input = self._latest_debug["specific_input"].float()

        residual_raw_var = float(residual_raw.var(unbiased=False).item())
        residual_scaled_var = float(residual_scaled.var(unbiased=False).item())
        shared_context_var = float(shared_context.var(unbiased=False).item())
        shared_context_scaled_var = float(shared_context_scaled.var(unbiased=False).item())
        specific_input_var = float(specific_input.var(unbiased=False).item())
        specific_residual_condition_ratio = residual_scaled_var / (residual_scaled_var + shared_context_scaled_var + 1e-12)

        state.update({
            "specific_residual_raw_var": residual_raw_var,
            "specific_residual_scaled_var": residual_scaled_var,
            "specific_shared_context_var": shared_context_var,
            "specific_shared_context_scaled_var": shared_context_scaled_var,
            "specific_input_var": specific_input_var,
            "specific_residual_condition_ratio": specific_residual_condition_ratio,
            "specific_residual_gain": float(self.specific_residual_gain.detach().item()),
            "specific_shared_gain": float(self.specific_shared_gain.detach().item()),
        })
        return state

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
                "[SHAVQ-v3b Recorder] "
                f"step={ctx.global_step + 1} "
                f"s1={int(debug['stage1_used_codes'])}/{self.codebook_size} "
                f"s2={int(debug['stage2_used_codes'])}/{self.codebook_size} "
                f"ent1={debug['stage1_entropy']:.3f} "
                f"ent2={debug['stage2_entropy']:.3f} "
                f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
                f"logit_ratio={contrib['logit_var_shared_ratio']:.3f} "
                f"res_gain={contrib['specific_residual_gain']:.3f} "
                f"ctx_gain={contrib['specific_shared_gain']:.3f} "
                f"cond_res_ratio={contrib['specific_residual_condition_ratio']:.3f} "
                f"r2={contrib['residual2_norm_mean']:.3f} "
                f"total_cos={contrib['total_cos_mean']:.3f}"
            )
            print(ctx.recorder.get_window_stats())
        return float(loss.item())

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx: TrainContext) -> None:
        del metrics, ctx
        debug = self.debug_state()
        contrib = self.contribution_state()
        print(
            "[SHAVQ-v3b Debug] "
            f"ready1={int(debug['stage1_initialized'])} "
            f"ready2={int(debug['stage2_initialized'])} "
            f"s1={int(debug['stage1_used_codes'])}/{self.codebook_size} "
            f"s2={int(debug['stage2_used_codes'])}/{self.codebook_size} "
            f"ent1={debug['stage1_entropy']:.3f} "
            f"ent2={debug['stage2_entropy']:.3f} "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_ratio={contrib['logit_var_shared_ratio']:.3f} "
            f"res_gain={contrib['specific_residual_gain']:.3f} "
            f"ctx_gain={contrib['specific_shared_gain']:.3f} "
            f"cond_res_ratio={contrib['specific_residual_condition_ratio']:.3f} "
            f"r2={contrib['residual2_norm_mean']:.3f} "
            f"total_cos={contrib['total_cos_mean']:.3f}"
        )
