from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.train.context import TrainContext
from betterbole.utils.observatory import TensorDisplayConfig, TensorMonitorOptions
from betterbole.utils.observatory.plots import PlotSeries, plot_multi_series
from betterbole.models.utils.common import to_dims
from custom_models.vq_share.model import VQShareModel


class VQShareGateV1Model(VQShareModel):
    """
    Lightweight residual-gated extension over VQShare.

    Design goals:
    - keep the original VQ shared path intact
    - add a domain-aware gate that decides how strongly to trust shared vs residual
    - expose gate behaviour to observatory without adding a heavy new branch structure
    """

    def __init__(
            self,
            *args,
            gate_hidden_dims: Iterable[int] = (128,),
            gate_domain_embed_dim: int = 16,
            gate_temperature: float = 0.7,
            shared_logit_floor: float = 0.85,
            shared_logit_ceiling: float = 1.15,
            specific_logit_floor: float = 0.40,
            specific_logit_ceiling: float = 1.30,
            residual_gain_floor: float = 0.60,
            residual_gain_ceiling: float = 1.40,
            gate_dropout_rate: float = 0.0,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gate_domain_embed_dim = int(gate_domain_embed_dim)
        self.gate_temperature = max(float(gate_temperature), 1e-4)
        self.shared_logit_floor = float(shared_logit_floor)
        self.shared_logit_ceiling = float(shared_logit_ceiling)
        self.specific_logit_floor = float(specific_logit_floor)
        self.specific_logit_ceiling = float(specific_logit_ceiling)
        self.residual_gain_floor = float(residual_gain_floor)
        self.residual_gain_ceiling = float(residual_gain_ceiling)

        gate_hidden_dims = to_dims(gate_hidden_dims, (128,))
        gate_input_dim = (
            self.shared_output_dim
            + self.specific_output_dim
            + int(self.domain_context_dim)
            + self.gate_domain_embed_dim
            + 2
        )
        self.gate_domain_embedding = nn.Embedding(self.num_domains, self.gate_domain_embed_dim)
        self.branch_gate = self._build_gate_stack(
            input_dim=gate_input_dim,
            hidden_dims=gate_hidden_dims,
            output_dim=2,
            dropout_rate=gate_dropout_rate,
        )
        self.residual_gain_head = self._build_gate_stack(
            input_dim=gate_input_dim,
            hidden_dims=gate_hidden_dims,
            output_dim=1,
            dropout_rate=gate_dropout_rate,
        )
        self._reset_gate_parameters()

    @staticmethod
    def _build_gate_stack(
            input_dim: int,
            hidden_dims: Iterable[int],
            output_dim: int,
            dropout_rate: float,
    ) -> nn.Sequential:
        dims = [int(input_dim), *[int(dim) for dim in hidden_dims], int(output_dim)]
        layers = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.GELU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    @staticmethod
    def _last_linear(module: nn.Module) -> nn.Linear:
        for child in reversed(list(module.modules())):
            if isinstance(child, nn.Linear):
                return child
        raise ValueError("Expected module to contain at least one Linear layer.")

    def _reset_gate_parameters(self) -> None:
        nn.init.normal_(self.gate_domain_embedding.weight, mean=0.0, std=0.02)
        for module in [self.branch_gate, self.residual_gain_head]:
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)

        branch_gate_last = self._last_linear(self.branch_gate)
        residual_gain_last = self._last_linear(self.residual_gain_head)
        with torch.no_grad():
            branch_gate_last.bias.copy_(torch.tensor([0.35, -0.35], dtype=branch_gate_last.bias.dtype))
            residual_gain_last.bias.zero_()

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
        shared_logits_raw = self.shared_head(shared_hidden).squeeze(-1)

        if domain_context is not None:
            specific_input = torch.cat([specific_z, domain_context], dim=-1)
        else:
            specific_input = specific_z
        specific_hidden = self.specific_expert(specific_input)
        specific_bias_feature, specific_fluctuation_feature = self.specific_bifurcator(specific_hidden)
        specific_logits_raw = self.specific_head(specific_hidden, domain_ids)
        specific_logits_base = self.specific_head(specific_fluctuation_feature, domain_ids)

        quantized_cos = (
            matched_similarity
            if matched_similarity is not None
            else torch.ones_like(shared_logits_raw)
        )
        residual_norm = specific_z.norm(dim=-1)

        gate_inputs = [
            shared_hidden,
            specific_fluctuation_feature,
            self.gate_domain_embedding(domain_ids),
            residual_norm.unsqueeze(-1),
            quantized_cos.unsqueeze(-1),
        ]
        if domain_context is not None:
            gate_inputs.append(domain_context)
        gate_input = torch.cat(gate_inputs, dim=-1)

        branch_gate_logits = self.branch_gate(gate_input)
        branch_gate_weights = F.softmax(branch_gate_logits / self.gate_temperature, dim=-1)

        shared_logit_scale = self.shared_logit_floor + (
            self.shared_logit_ceiling - self.shared_logit_floor
        ) * branch_gate_weights[:, 0]
        specific_logit_scale = self.specific_logit_floor + (
            self.specific_logit_ceiling - self.specific_logit_floor
        ) * branch_gate_weights[:, 1]
        residual_gain = self.residual_gain_floor + (
            self.residual_gain_ceiling - self.residual_gain_floor
        ) * torch.sigmoid(self.residual_gain_head(gate_input).squeeze(-1))

        shared_logits = shared_logit_scale * shared_logits_raw
        specific_logits = specific_logit_scale * residual_gain * specific_logits_base
        logits = shared_logits + specific_logits

        contrastive_loss = self._contrastive_loss(x) if compute_aux_losses else logits.new_zeros(())
        self._latest_commitment_loss = commitment_loss.detach()
        self._latest_contrastive_loss = contrastive_loss.detach()
        self._latest_code_indices = code_indices.detach()
        self._latest_quantized_share = quantized.detach()
        self._latest_debug = {
            "domain_ids": domain_ids.detach(),
            "code_indices": code_indices.detach(),
            "shared_hidden": shared_hidden.detach(),
            "shared_logits": shared_logits.detach(),
            "shared_logits_raw": shared_logits_raw.detach(),
            "shared_feature_importance": shared_hidden.detach().abs(),
            "specific_logits_raw": specific_logits_raw.detach(),
            "specific_logits_base": specific_logits_base.detach(),
            "specific_logits": specific_logits.detach(),
            "specific_hidden": specific_hidden.detach(),
            "specific_feature_bias": specific_bias_feature.detach(),
            "specific_feature_fluctuation": specific_fluctuation_feature.detach(),
            "specific_feature_importance": specific_fluctuation_feature.detach().abs(),
            "shared_norm": shared_z.detach().norm(dim=-1),
            "residual_norm": residual_norm.detach(),
            "quantized_cos": quantized_cos.detach(),
            "branch_gate_weights": branch_gate_weights.detach(),
            "branch_gate_logits": branch_gate_logits.detach(),
            "shared_logit_scale": shared_logit_scale.detach(),
            "specific_logit_scale": specific_logit_scale.detach(),
            "residual_gain": residual_gain.detach(),
        }
        return logits, commitment_loss, contrastive_loss

    def _setup_observatory(self, recorder) -> None:
        super()._setup_observatory(recorder)
        if recorder is None or not hasattr(recorder, "register"):
            return

        gate_options = TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=4,
                topk_display_dims=4,
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
        for name in ["vq_gate_branch_weights", "vq_gate_logit_scales"]:
            recorder.register(name, gate_options)
        recorder.register("vq_gate_residual_gain", scalar_options)

    def _record_debug_tensors(self, recorder, step: Optional[int] = None) -> None:
        super()._record_debug_tensors(recorder, step=step)
        if recorder is None or not self._latest_debug:
            return

        recorder.record("vq_gate_branch_weights", self._latest_debug["branch_gate_weights"], step=step)
        recorder.record(
            "vq_gate_logit_scales",
            torch.stack(
                [
                    self._latest_debug["shared_logit_scale"],
                    self._latest_debug["specific_logit_scale"],
                ],
                dim=-1,
            ),
            step=step,
        )
        recorder.record("vq_gate_residual_gain", self._latest_debug["residual_gain"].unsqueeze(-1), step=step)

    def _append_observatory_scalars(self, step: int) -> None:
        super()._append_observatory_scalars(step)
        if not self._latest_debug:
            return

        branch_gate_weights = self._latest_debug["branch_gate_weights"].float().clamp_min(1e-12)
        gate_entropy = float((-(branch_gate_weights * branch_gate_weights.log()).sum(dim=-1).mean()).item())
        extra_values = {
            "gate_entropy": gate_entropy,
            "gate_shared_mean": float(branch_gate_weights[:, 0].mean().item()),
            "gate_specific_mean": float(branch_gate_weights[:, 1].mean().item()),
            "shared_logit_scale_mean": float(self._latest_debug["shared_logit_scale"].float().mean().item()),
            "specific_logit_scale_mean": float(self._latest_debug["specific_logit_scale"].float().mean().item()),
            "residual_gain_mean": float(self._latest_debug["residual_gain"].float().mean().item()),
            "raw_shared_logit_var": float(self._latest_debug["shared_logits_raw"].float().var(unbiased=False).item()),
            "base_specific_logit_var": float(self._latest_debug["specific_logits_base"].float().var(unbiased=False).item()),
        }
        for key, value in extra_values.items():
            self._observatory_scalar_history.setdefault(key, []).append(float(value))

    def _export_observatory_artifacts(self, ctx: TrainContext) -> None:
        super()._export_observatory_artifacts(ctx)
        if not self._observatory_steps:
            return

        output_dir = self._observatory_output_dir(ctx)
        plot_multi_series(
            [
                PlotSeries("gate_entropy", self._observatory_steps, self._observatory_scalar_history.get("gate_entropy", [])),
                PlotSeries("gate_shared_mean", self._observatory_steps, self._observatory_scalar_history.get("gate_shared_mean", [])),
                PlotSeries("gate_specific_mean", self._observatory_steps, self._observatory_scalar_history.get("gate_specific_mean", [])),
            ],
            title="vq_gate_series",
            xlabel="step",
            ylabel="value",
            save_path=output_dir / "vq_gate_series.png",
        )
        plot_multi_series(
            [
                PlotSeries("shared_scale", self._observatory_steps, self._observatory_scalar_history.get("shared_logit_scale_mean", [])),
                PlotSeries("specific_scale", self._observatory_steps, self._observatory_scalar_history.get("specific_logit_scale_mean", [])),
                PlotSeries("residual_gain", self._observatory_steps, self._observatory_scalar_history.get("residual_gain_mean", [])),
            ],
            title="vq_gate_scale_series",
            xlabel="step",
            ylabel="value",
            save_path=output_dir / "vq_gate_scale_series.png",
        )
        plot_multi_series(
            [
                PlotSeries("raw_shared_logit_var", self._observatory_steps, self._observatory_scalar_history.get("raw_shared_logit_var", [])),
                PlotSeries("base_specific_logit_var", self._observatory_steps, self._observatory_scalar_history.get("base_specific_logit_var", [])),
                PlotSeries("scaled_shared_logit_var", self._observatory_steps, self._observatory_scalar_history.get("shared_logit_var", [])),
                PlotSeries("scaled_specific_logit_var", self._observatory_steps, self._observatory_scalar_history.get("specific_logit_var", [])),
            ],
            title="vq_gate_logit_var_series",
            xlabel="step",
            ylabel="logit_var",
            save_path=output_dir / "vq_gate_logit_var_series.png",
        )

    def contribution_state(self) -> Dict[str, float]:
        state = super().contribution_state()
        if not self._latest_debug:
            state.update(
                {
                    "gate_entropy": 0.0,
                    "gate_shared_mean": 0.0,
                    "gate_specific_mean": 0.0,
                    "shared_logit_scale_mean": 0.0,
                    "specific_logit_scale_mean": 0.0,
                    "residual_gain_mean": 0.0,
                    "raw_shared_logit_var": 0.0,
                    "base_specific_logit_var": 0.0,
                }
            )
            return state

        branch_gate_weights = self._latest_debug["branch_gate_weights"].float().clamp_min(1e-12)
        state.update(
            {
                "gate_entropy": float((-(branch_gate_weights * branch_gate_weights.log()).sum(dim=-1).mean()).item()),
                "gate_shared_mean": float(branch_gate_weights[:, 0].mean().item()),
                "gate_specific_mean": float(branch_gate_weights[:, 1].mean().item()),
                "shared_logit_scale_mean": float(self._latest_debug["shared_logit_scale"].float().mean().item()),
                "specific_logit_scale_mean": float(self._latest_debug["specific_logit_scale"].float().mean().item()),
                "residual_gain_mean": float(self._latest_debug["residual_gain"].float().mean().item()),
                "raw_shared_logit_var": float(self._latest_debug["shared_logits_raw"].float().var(unbiased=False).item()),
                "base_specific_logit_var": float(self._latest_debug["specific_logits_base"].float().var(unbiased=False).item()),
            }
        )
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
                "[VQShareGateV1 Recorder] "
                f"step={ctx.global_step + 1} "
                f"used={int(debug['used_codes'])}/{self.codebook_size} "
                f"usage_ratio={debug['used_code_ratio']:.3f} "
                f"entropy={debug['code_usage_entropy']:.3f} "
                f"commit={debug['latest_commitment_loss']:.5f} "
                f"contrast={debug['latest_contrastive_loss']:.5f} "
                f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
                f"logit_ratio={contrib['logit_var_shared_ratio']:.3f} "
                f"gate=({contrib['gate_shared_mean']:.3f},{contrib['gate_specific_mean']:.3f}) "
                f"gate_ent={contrib['gate_entropy']:.3f} "
                f"scale=({contrib['shared_logit_scale_mean']:.3f},{contrib['specific_logit_scale_mean']:.3f}) "
                f"res_gain={contrib['residual_gain_mean']:.3f}"
            )
            print(
                ctx.recorder.get_window_stats(
                    names=[
                        "vq_shared_hidden",
                        "vq_specific_feature_fluctuation",
                        "vq_code_usage",
                        "vq_gate_branch_weights",
                        "vq_gate_residual_gain",
                    ],
                    include_relations=True,
                    relation_names=[
                        "vq_shared_hidden",
                        "vq_specific_feature_fluctuation",
                        "vq_gate_branch_weights",
                    ],
                )
            )
        return float(loss.item())

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx: TrainContext) -> None:
        del metrics
        debug = self.debug_state()
        contrib = self.contribution_state()
        print(
            "[VQShareGateV1 Debug] "
            f"ready={int(debug['codebook_initialized'])} "
            f"used={int(debug['used_codes'])}/{self.codebook_size} "
            f"usage_ratio={debug['used_code_ratio']:.3f} "
            f"entropy={debug['code_usage_entropy']:.3f} "
            f"commit={debug['latest_commitment_loss']:.5f} "
            f"contrast={debug['latest_contrastive_loss']:.5f} "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_ratio={contrib['logit_var_shared_ratio']:.3f} "
            f"gate=({contrib['gate_shared_mean']:.3f},{contrib['gate_specific_mean']:.3f}) "
            f"gate_ent={contrib['gate_entropy']:.3f} "
            f"scale=({contrib['shared_logit_scale_mean']:.3f},{contrib['specific_logit_scale_mean']:.3f}) "
            f"res_gain={contrib['residual_gain_mean']:.3f}"
        )
        if ctx.recorder is not None:
            print(
                ctx.recorder.get_window_stats(
                    names=[
                        "vq_shared_hidden",
                        "vq_specific_feature_fluctuation",
                        "vq_code_usage",
                        "vq_gate_branch_weights",
                        "vq_gate_residual_gain",
                    ],
                    include_relations=True,
                    relation_names=[
                        "vq_shared_hidden",
                        "vq_specific_feature_fluctuation",
                        "vq_gate_branch_weights",
                    ],
                )
            )
            self._export_observatory_artifacts(ctx)
