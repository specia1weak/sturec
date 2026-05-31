from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.train.context import TrainContext
from betterbole.emb import SchemaManager
from betterbole.experiment import WORKSPACE
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.msr.ple.layers import select_domain_output
from betterbole.models.utils.common import build_mlp, to_dims
from betterbole.utils.observatory import RelationOptions, TensorDisplayConfig, TensorMonitorOptions


def sharpened_softmax(logits: torch.Tensor, temperature: float = 1.0, power: float = 1.0) -> torch.Tensor:
    temperature = max(float(temperature), 1e-4)
    probs = F.softmax(logits / temperature, dim=-1)
    power = float(power)
    if abs(power - 1.0) > 1e-6:
        probs = probs.clamp_min(1e-12).pow(power)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


def _gather_domain_tensor(stacked: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
    gather_index = domain_ids.long().view(-1, 1, 1, 1).expand(-1, 1, stacked.size(2), stacked.size(3))
    return stacked.gather(dim=1, index=gather_index).squeeze(1)


class PLEObservatoryLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            task_gate_temperature: float = 1.0,
            task_gate_power: float = 1.0,
            shared_gate_temperature: float = 1.0,
            shared_gate_power: float = 1.0,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = int(num_domains)
        self.num_specific_experts = int(num_specific_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.output_dim = int(expert_dims[-1])
        self.task_gate_temperature = max(float(task_gate_temperature), 1e-4)
        self.task_gate_power = float(task_gate_power)
        self.shared_gate_temperature = max(float(shared_gate_temperature), 1e-4)
        self.shared_gate_power = float(shared_gate_power)

        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                build_mlp(
                    input_dim,
                    expert_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
                for _ in range(self.num_specific_experts)
            ])
            for _ in range(self.num_domains)
        ])
        self.shared_experts = nn.ModuleList([
            build_mlp(
                input_dim,
                expert_dims,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for _ in range(self.num_shared_experts)
        ])
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.num_specific_experts + self.num_shared_experts),
            )
            for _ in range(self.num_domains)
        ])
        self.shared_gate = nn.Sequential(
            nn.Linear(input_dim, self.num_domains * self.num_specific_experts + self.num_shared_experts),
        )

    def forward_with_debug(
            self,
            task_inputs: List[torch.Tensor],
            shared_input: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        task_outputs = []
        all_specific = []
        per_domain_specific_outputs = []
        per_domain_task_gates = []

        shared_outputs = torch.stack([expert(shared_input) for expert in self.shared_experts], dim=1)

        for domain_idx in range(self.num_domains):
            task_input = task_inputs[domain_idx]
            specific_outputs = torch.stack(
                [expert(task_input) for expert in self.specific_experts[domain_idx]],
                dim=1,
            )
            per_domain_specific_outputs.append(specific_outputs)
            all_specific.extend([specific_outputs[:, expert_idx, :] for expert_idx in range(self.num_specific_experts)])

            gate = sharpened_softmax(
                self.task_gates[domain_idx](task_input),
                temperature=self.task_gate_temperature,
                power=self.task_gate_power,
            )
            per_domain_task_gates.append(gate)
            mix_inputs = torch.cat([specific_outputs, shared_outputs], dim=1)
            task_outputs.append(torch.sum(gate.unsqueeze(-1) * mix_inputs, dim=1))

        shared_gate = sharpened_softmax(
            self.shared_gate(shared_input),
            temperature=self.shared_gate_temperature,
            power=self.shared_gate_power,
        )
        shared_mix = torch.stack(all_specific + [shared_outputs[:, expert_idx, :] for expert_idx in range(self.num_shared_experts)], dim=1)
        shared_output = torch.sum(shared_gate.unsqueeze(-1) * shared_mix, dim=1)

        stacked_task_outputs = torch.stack(task_outputs, dim=1)
        stacked_specific_outputs = torch.stack(per_domain_specific_outputs, dim=1)
        stacked_task_gates = torch.stack(per_domain_task_gates, dim=1)

        selected_task_output = select_domain_output(task_outputs=task_outputs, domain_ids=domain_ids)
        selected_specific_outputs = _gather_domain_tensor(stacked_specific_outputs, domain_ids)
        selected_task_gate = stacked_task_gates[
            torch.arange(domain_ids.size(0), device=domain_ids.device),
            domain_ids.long(),
        ]

        debug = {
            "task_outputs": stacked_task_outputs,
            "shared_output": shared_output,
            "shared_expert_outputs": shared_outputs,
            "specific_expert_outputs": stacked_specific_outputs,
            "task_gates": stacked_task_gates,
            "shared_gate": shared_gate,
            "selected_task_output": selected_task_output,
            "selected_specific_outputs": selected_specific_outputs,
            "selected_task_gate": selected_task_gate,
        }
        return task_outputs, shared_output, debug


class PLEObservatoryEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: int = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            task_gate_temperature: float = 1.0,
            task_gate_power: float = 1.0,
            shared_gate_temperature: float = 1.0,
            shared_gate_power: float = 1.0,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = int(num_domains)
        self.output_dim = int(expert_dims[-1])
        self.num_levels = int(num_levels)
        self.num_specific_experts = int(num_specific_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.layers = nn.ModuleList()

        layer_input_dim = int(input_dim)
        for _ in range(self.num_levels):
            layer = PLEObservatoryLayer(
                input_dim=layer_input_dim,
                num_domains=num_domains,
                expert_dims=expert_dims,
                num_specific_experts=num_specific_experts,
                num_shared_experts=num_shared_experts,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
                task_gate_temperature=task_gate_temperature,
                task_gate_power=task_gate_power,
                shared_gate_temperature=shared_gate_temperature,
                shared_gate_power=shared_gate_power,
            )
            self.layers.append(layer)
            layer_input_dim = layer.output_dim

    def forward_debug(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        task_inputs = [x for _ in range(self.num_domains)]
        shared_input = x
        layer_debugs = []

        for layer in self.layers:
            task_inputs, shared_input, layer_debug = layer.forward_with_debug(task_inputs, shared_input, domain_ids)
            layer_debugs.append(layer_debug)

        selected_hidden = select_domain_output(task_outputs=task_inputs, domain_ids=domain_ids)

        selected_task_outputs = torch.stack([layer_debug["selected_task_output"] for layer_debug in layer_debugs], dim=1)
        shared_layer_outputs = torch.stack([layer_debug["shared_output"] for layer_debug in layer_debugs], dim=1)
        selected_specific_experts = torch.stack([layer_debug["selected_specific_outputs"] for layer_debug in layer_debugs], dim=1)
        shared_expert_outputs = torch.stack([layer_debug["shared_expert_outputs"] for layer_debug in layer_debugs], dim=1)
        selected_task_gates = torch.stack([layer_debug["selected_task_gate"] for layer_debug in layer_debugs], dim=1)
        shared_gate_weights = torch.stack([layer_debug["shared_gate"] for layer_debug in layer_debugs], dim=1)
        all_task_gate_weights = torch.stack([layer_debug["task_gates"] for layer_debug in layer_debugs], dim=1)
        all_task_outputs = torch.stack([layer_debug["task_outputs"] for layer_debug in layer_debugs], dim=1)

        selected_specific_mean = selected_specific_experts.mean(dim=2)
        shared_expert_mean = shared_expert_outputs.mean(dim=2)
        task_shared_delta = selected_task_outputs - shared_layer_outputs

        debug = {
            "selected_task_outputs_by_layer": selected_task_outputs,
            "shared_outputs_by_layer": shared_layer_outputs,
            "selected_specific_experts_by_layer": selected_specific_experts,
            "shared_experts_by_layer": shared_expert_outputs,
            "selected_task_gates_by_layer": selected_task_gates,
            "shared_gate_weights_by_layer": shared_gate_weights,
            "all_task_gate_weights_by_layer": all_task_gate_weights,
            "all_task_outputs_by_layer": all_task_outputs,
            "selected_specific_mean_by_layer": selected_specific_mean,
            "shared_expert_mean_by_layer": shared_expert_mean,
            "task_shared_delta_by_layer": task_shared_delta,
        }
        return selected_hidden, shared_input, debug


class PLEObservatoryModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
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
            task_gate_temperature: float = 1.0,
            task_gate_power: float = 1.0,
            shared_gate_temperature: float = 1.0,
            shared_gate_power: float = 1.0,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim
        self.num_levels = int(num_levels)
        self.num_specific_experts = int(num_specific_experts)
        self.num_shared_experts = int(num_shared_experts)

        self.encoder = PLEObservatoryEncoder(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_levels=num_levels,
            num_specific_experts=num_specific_experts,
            num_shared_experts=num_shared_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            task_gate_temperature=task_gate_temperature,
            task_gate_power=task_gate_power,
            shared_gate_temperature=shared_gate_temperature,
            shared_gate_power=shared_gate_power,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.encoder.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )

        self._observatory_initialized = False
        self._latest_debug = {}

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long().view(-1)

    def _forward_dict(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        selected_hidden, shared_hidden, encoder_debug = self.encoder.forward_debug(x, domain_ids)
        logits = self.head(selected_hidden, domain_ids)

        debug = {
            "selected_hidden": selected_hidden,
            "shared_hidden": shared_hidden,
            "logits": logits,
            "domain_ids": domain_ids,
        }
        debug.update(encoder_debug)
        return debug

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_dict(x, domain_ids)["logits"]

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_dict(x, domain_ids)
        loss = F.binary_cross_entropy_with_logits(forward_dict["logits"], labels)
        self._latest_debug = {key: value.detach() for key, value in forward_dict.items()}
        return loss

    def _observatory_output_dir(self, ctx: TrainContext) -> Path:
        model_name = getattr(ctx.cfg, "model", None)
        output_name = str(model_name or getattr(ctx.cfg, "experiment_name", "ple_ob"))
        return WORKSPACE / ctx.cfg.dataset_name / "observatory" / output_name

    def _setup_observatory(self, recorder) -> None:
        if recorder is None or self._observatory_initialized:
            return
        if not hasattr(recorder, "register"):
            return

        hidden_options = TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=12,
                topk_display_dims=8,
                rank_by="variance",
            )
        )
        gate_options = TensorMonitorOptions(
            display=TensorDisplayConfig(
                show_global_summary=True,
                show_per_dim=True,
                max_display_dims=8,
                topk_display_dims=8,
                rank_by="variance",
            )
        )
        for name in [
            "ple_ob_selected_hidden",
            "ple_ob_shared_hidden",
            "ple_ob_selected_task_outputs_by_layer",
            "ple_ob_shared_outputs_by_layer",
            "ple_ob_selected_specific_mean_by_layer",
            "ple_ob_shared_expert_mean_by_layer",
            "ple_ob_task_shared_delta_by_layer",
        ]:
            recorder.register(name, hidden_options)
        for name in [
            "ple_ob_selected_task_gates_by_layer",
            "ple_ob_shared_gate_weights_by_layer",
            "ple_ob_gate_mass_by_layer",
            "ple_ob_shared_gate_mass_by_layer",
        ]:
            recorder.register(name, gate_options)

        if hasattr(recorder, "configure_relations"):
            recorder.configure_relations(
                RelationOptions(
                    enabled=True,
                    rank=8,
                    max_pairs=12,
                    names=(
                        "ple_ob_selected_hidden",
                        "ple_ob_shared_hidden",
                        "ple_ob_selected_specific_mean_by_layer",
                        "ple_ob_shared_expert_mean_by_layer",
                        "ple_ob_task_shared_delta_by_layer",
                    ),
                )
            )
        self._observatory_initialized = True

    def debug_state(self) -> Dict[str, float]:
        if not self._latest_debug:
            return {
                "task_gate_entropy": 0.0,
                "shared_gate_entropy": 0.0,
                "task_gate_shared_mass": 0.0,
                "shared_gate_shared_mass": 0.0,
                "task_shared_alignment": 0.0,
            }

        selected_task_gates = self._latest_debug["selected_task_gates_by_layer"].float().clamp_min(1e-12)
        shared_gate_weights = self._latest_debug["shared_gate_weights_by_layer"].float().clamp_min(1e-12)
        selected_hidden = self._latest_debug["selected_hidden"].float()
        shared_hidden = self._latest_debug["shared_hidden"].float()
        task_shared_alignment = F.cosine_similarity(selected_hidden, shared_hidden, dim=-1).mean()

        specific_width = self.num_specific_experts
        shared_width = self.num_shared_experts
        task_gate_shared_mass = selected_task_gates[:, :, specific_width:specific_width + shared_width].sum(dim=-1).mean()
        shared_gate_shared_mass = shared_gate_weights[:, :, self.num_domains * self.num_specific_experts:].sum(dim=-1).mean()

        return {
            "task_gate_entropy": float((-(selected_task_gates * selected_task_gates.log()).sum(dim=-1).mean()).item()),
            "shared_gate_entropy": float((-(shared_gate_weights * shared_gate_weights.log()).sum(dim=-1).mean()).item()),
            "task_gate_shared_mass": float(task_gate_shared_mass.item()),
            "shared_gate_shared_mass": float(shared_gate_shared_mass.item()),
            "task_shared_alignment": float(task_shared_alignment.item()),
        }

    def contribution_state(self) -> Dict[str, float]:
        if not self._latest_debug:
            return {
                "selected_feature_var": 0.0,
                "shared_feature_var": 0.0,
                "feature_var_shared_ratio": 0.0,
                "logit_var": 0.0,
                "task_shared_delta_var": 0.0,
                "selected_specific_mean_var": 0.0,
                "shared_expert_mean_var": 0.0,
                "selected_low_var_dim_ratio": 0.0,
                "shared_low_var_dim_ratio": 0.0,
                "task_gate_domain_dispersion": 0.0,
                "task_output_domain_dispersion": 0.0,
            }

        selected_hidden = self._latest_debug["selected_hidden"].float()
        shared_hidden = self._latest_debug["shared_hidden"].float()
        logits = self._latest_debug["logits"].float()
        task_shared_delta = self._latest_debug["task_shared_delta_by_layer"].float()
        selected_specific_mean = self._latest_debug["selected_specific_mean_by_layer"].float()
        shared_expert_mean = self._latest_debug["shared_expert_mean_by_layer"].float()
        all_task_gate_weights = self._latest_debug["all_task_gate_weights_by_layer"].float()
        all_task_outputs = self._latest_debug["all_task_outputs_by_layer"].float()

        selected_feature_var = float(selected_hidden.var(unbiased=False).item())
        shared_feature_var = float(shared_hidden.var(unbiased=False).item())
        feature_var_total = selected_feature_var + shared_feature_var + 1e-12

        selected_per_dim_var = selected_hidden.var(dim=0, unbiased=False)
        shared_per_dim_var = shared_hidden.var(dim=0, unbiased=False)

        task_gate_domain_mean = all_task_gate_weights.mean(dim=0)
        task_output_domain_mean = all_task_outputs.mean(dim=0)

        return {
            "selected_feature_var": selected_feature_var,
            "shared_feature_var": shared_feature_var,
            "feature_var_shared_ratio": shared_feature_var / feature_var_total,
            "logit_var": float(logits.var(unbiased=False).item()),
            "task_shared_delta_var": float(task_shared_delta.var(unbiased=False).item()),
            "selected_specific_mean_var": float(selected_specific_mean.var(unbiased=False).item()),
            "shared_expert_mean_var": float(shared_expert_mean.var(unbiased=False).item()),
            "selected_low_var_dim_ratio": float((selected_per_dim_var < 1e-6).float().mean().item()),
            "shared_low_var_dim_ratio": float((shared_per_dim_var < 1e-6).float().mean().item()),
            "task_gate_domain_dispersion": float(task_gate_domain_mean.var(unbiased=False).item()),
            "task_output_domain_dispersion": float(task_output_domain_mean.var(unbiased=False).item()),
        }

    def _record_debug_tensors(self, recorder, step: Optional[int] = None) -> None:
        if recorder is None or not self._latest_debug:
            return
        self._setup_observatory(recorder)

        selected_task_gates = self._latest_debug["selected_task_gates_by_layer"]
        shared_gate_weights = self._latest_debug["shared_gate_weights_by_layer"]

        gate_mass = torch.stack([
            selected_task_gates[:, :, :self.num_specific_experts].sum(dim=-1),
            selected_task_gates[:, :, self.num_specific_experts:].sum(dim=-1),
        ], dim=-1)
        shared_gate_mass = torch.stack([
            shared_gate_weights[:, :, :self.num_domains * self.num_specific_experts].sum(dim=-1),
            shared_gate_weights[:, :, self.num_domains * self.num_specific_experts:].sum(dim=-1),
        ], dim=-1)

        recorder.record("ple_ob_selected_hidden", self._latest_debug["selected_hidden"], step=step)
        recorder.record("ple_ob_shared_hidden", self._latest_debug["shared_hidden"], step=step)
        recorder.record("ple_ob_selected_task_outputs_by_layer", self._latest_debug["selected_task_outputs_by_layer"], step=step)
        recorder.record("ple_ob_shared_outputs_by_layer", self._latest_debug["shared_outputs_by_layer"], step=step)
        recorder.record("ple_ob_selected_specific_mean_by_layer", self._latest_debug["selected_specific_mean_by_layer"], step=step)
        recorder.record("ple_ob_shared_expert_mean_by_layer", self._latest_debug["shared_expert_mean_by_layer"], step=step)
        recorder.record("ple_ob_task_shared_delta_by_layer", self._latest_debug["task_shared_delta_by_layer"], step=step)
        recorder.record("ple_ob_selected_task_gates_by_layer", selected_task_gates, step=step)
        recorder.record("ple_ob_shared_gate_weights_by_layer", shared_gate_weights, step=step)
        recorder.record("ple_ob_gate_mass_by_layer", gate_mass, step=step)
        recorder.record("ple_ob_shared_gate_mass_by_layer", shared_gate_mass, step=step)
        recorder.record("ple_ob_logits", self._latest_debug["logits"].unsqueeze(-1), step=step)

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        loss = self.calculate_loss(batch_interaction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        self._record_debug_tensors(ctx.recorder, step=ctx.global_step + 1)
        if ctx.recorder is not None and (ctx.global_step + 1) % 200 == 0:
            debug = self.debug_state()
            contrib = self.contribution_state()
            print(
                "[PLE-OB Recorder] "
                f"step={ctx.global_step + 1} "
                f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
                f"logit_var={contrib['logit_var']:.4f} "
                f"task_gate_ent={debug['task_gate_entropy']:.3f} "
                f"shared_gate_ent={debug['shared_gate_entropy']:.3f} "
                f"task_shared_mass={debug['task_gate_shared_mass']:.3f} "
                f"shared_gate_shared_mass={debug['shared_gate_shared_mass']:.3f} "
                f"align={debug['task_shared_alignment']:.3f} "
                f"delta_var={contrib['task_shared_delta_var']:.4f} "
                f"low_var(selected/shared)={contrib['selected_low_var_dim_ratio']:.3f}/{contrib['shared_low_var_dim_ratio']:.3f}"
            )
            print(
                ctx.recorder.get_window_stats(
                    names=[
                        "ple_ob_selected_hidden",
                        "ple_ob_shared_hidden",
                        "ple_ob_selected_specific_mean_by_layer",
                        "ple_ob_shared_expert_mean_by_layer",
                        "ple_ob_selected_task_gates_by_layer",
                        "ple_ob_shared_gate_weights_by_layer",
                        "ple_ob_gate_mass_by_layer",
                        "ple_ob_shared_gate_mass_by_layer",
                    ],
                    include_relations=True,
                    relation_names=[
                        "ple_ob_selected_hidden",
                        "ple_ob_shared_hidden",
                        "ple_ob_selected_specific_mean_by_layer",
                        "ple_ob_shared_expert_mean_by_layer",
                        "ple_ob_task_shared_delta_by_layer",
                    ],
                )
            )
        return float(loss.item())

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx: TrainContext) -> None:
        del metrics
        debug = self.debug_state()
        contrib = self.contribution_state()
        print(
            "[PLE-OB Debug] "
            f"feat_ratio={contrib['feature_var_shared_ratio']:.3f} "
            f"logit_var={contrib['logit_var']:.4f} "
            f"task_gate_ent={debug['task_gate_entropy']:.3f} "
            f"shared_gate_ent={debug['shared_gate_entropy']:.3f} "
            f"task_shared_mass={debug['task_gate_shared_mass']:.3f} "
            f"shared_gate_shared_mass={debug['shared_gate_shared_mass']:.3f} "
            f"align={debug['task_shared_alignment']:.3f} "
            f"delta_var={contrib['task_shared_delta_var']:.4f} "
            f"gate_disp={contrib['task_gate_domain_dispersion']:.5f} "
            f"out_disp={contrib['task_output_domain_dispersion']:.5f}"
        )
        if ctx.recorder is not None:
            print(
                ctx.recorder.get_window_stats(
                    names=[
                        "ple_ob_selected_hidden",
                        "ple_ob_shared_hidden",
                        "ple_ob_selected_specific_mean_by_layer",
                        "ple_ob_shared_expert_mean_by_layer",
                        "ple_ob_selected_task_gates_by_layer",
                        "ple_ob_shared_gate_weights_by_layer",
                        "ple_ob_gate_mass_by_layer",
                        "ple_ob_shared_gate_mass_by_layer",
                    ],
                    include_relations=True,
                    relation_names=[
                        "ple_ob_selected_hidden",
                        "ple_ob_shared_hidden",
                        "ple_ob_selected_specific_mean_by_layer",
                        "ple_ob_shared_expert_mean_by_layer",
                        "ple_ob_task_shared_delta_by_layer",
                    ],
                )
            )


class PLEObservatorySharpModel(PLEObservatoryModel):
    def __init__(
            self,
            manager: SchemaManager,
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
            task_gate_temperature: float = 0.7,
            task_gate_power: float = 1.0,
            shared_gate_temperature: float = 0.7,
            shared_gate_power: float = 1.0,
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
            task_gate_temperature=task_gate_temperature,
            task_gate_power=task_gate_power,
            shared_gate_temperature=shared_gate_temperature,
            shared_gate_power=shared_gate_power,
        )
