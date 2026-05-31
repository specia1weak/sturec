from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.train.context import TrainContext
from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.msr.ple.layers import PLELayer, select_domain_output
from betterbole.models.utils.common import to_dims
from betterbole.models.utils.general import MLP


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(inputs: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversalFunction.apply(inputs, lambda_)


class PLEBalancedV3Encoder(nn.Module):
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
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = int(num_domains)
        self.output_dim = int(expert_dims[-1])
        self.layers = nn.ModuleList()

        layer_input_dim = int(input_dim)
        for _ in range(int(num_levels)):
            layer = PLELayer(
                input_dim=layer_input_dim,
                num_domains=num_domains,
                expert_dims=expert_dims,
                num_specific_experts=num_specific_experts,
                num_shared_experts=num_shared_experts,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            self.layers.append(layer)
            layer_input_dim = layer.output_dim

    def forward_all(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        task_inputs = [x for _ in range(self.num_domains)]
        shared_input = x
        for layer in self.layers:
            task_inputs, shared_input = layer(task_inputs, shared_input)
        return task_inputs, shared_input

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        task_outputs, shared_output = self.forward_all(x)
        return select_domain_output(task_outputs=task_outputs, domain_ids=domain_ids), shared_output


class PLEBalancedV3Model(MSRModel):
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
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim
        self.common_aux_weight = float(common_aux_weight)
        self.balanced_aux_weight = float(balanced_aux_weight)
        self.balanced_domain_adv_weight = float(balanced_domain_adv_weight)
        self.balanced_domain_adv_lambda = float(balanced_domain_adv_lambda)
        self.common_probe_weight = float(common_probe_weight)
        self.gate_temperature = max(float(gate_temperature), 1e-4)

        self.encoder = PLEBalancedV3Encoder(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_levels=num_levels,
            num_specific_experts=num_specific_experts,
            num_shared_experts=num_shared_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        output_dim = self.encoder.output_dim

        self.common_projection = nn.Linear(output_dim, output_dim)
        self.balanced_projection = nn.Linear(output_dim, output_dim)
        self.common_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.balanced_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.branch_dropout = nn.Dropout(branch_dropout_rate) if branch_dropout_rate > 0.0 else nn.Identity()

        self.common_head = nn.Linear(output_dim, 1)
        self.balanced_head = nn.Linear(output_dim, 1)

        domain_hidden_dims = to_dims(domain_head_hidden_dims, (max(1, output_dim // 2),))
        self.common_domain_probe = MLP(
            output_dim,
            *domain_hidden_dims,
            self.num_domains,
            dropout_rate=branch_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.balanced_domain_discriminator = MLP(
            output_dim,
            *domain_hidden_dims,
            self.num_domains,
            dropout_rate=branch_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

        self.domain_embedding = nn.Embedding(self.num_domains, gate_domain_embed_dim)
        gate_hidden_dims = to_dims(gate_hidden_dims, (max(32, output_dim),))
        self.gate_network = MLP(
            output_dim * 3 + gate_domain_embed_dim,
            *gate_hidden_dims,
            3,
            dropout_rate=branch_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )

        self._latest_task_loss = torch.tensor(0.0)
        self._latest_common_aux_loss = torch.tensor(0.0)
        self._latest_balanced_aux_loss = torch.tensor(0.0)
        self._latest_balanced_domain_adv_loss = torch.tensor(0.0)
        self._latest_common_probe_loss = torch.tensor(0.0)
        self._latest_debug = {}
        self.apply_xavier_initialization()

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long().view(-1)

    def _balanced_shared_bce(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_domain_losses = logits.new_zeros(self.num_domains)
        present_mask = torch.zeros(self.num_domains, dtype=torch.bool, device=logits.device)
        losses = []
        for domain_idx in range(self.num_domains):
            mask = domain_ids == domain_idx
            if not mask.any():
                continue
            domain_loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            per_domain_losses[domain_idx] = domain_loss
            present_mask[domain_idx] = True
            losses.append(domain_loss)
        if losses:
            return torch.stack(losses).mean(), per_domain_losses, present_mask
        return logits.new_zeros(()), per_domain_losses, present_mask

    def _balanced_domain_ce(
            self,
            logits: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_domain_losses = logits.new_zeros(self.num_domains)
        present_mask = torch.zeros(self.num_domains, dtype=torch.bool, device=logits.device)
        losses = []
        for domain_idx in range(self.num_domains):
            mask = domain_ids == domain_idx
            if not mask.any():
                continue
            domain_loss = F.cross_entropy(logits[mask], domain_ids[mask])
            per_domain_losses[domain_idx] = domain_loss
            present_mask[domain_idx] = True
            losses.append(domain_loss)
        if losses:
            return torch.stack(losses).mean(), per_domain_losses, present_mask
        return logits.new_zeros(()), per_domain_losses, present_mask

    def _forward_dict(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        specific_hidden, shared_hidden_raw = self.encoder(x, domain_ids)

        common_hidden_raw = self.common_projection(shared_hidden_raw)
        balanced_hidden_raw = self.balanced_projection(shared_hidden_raw)
        common_hidden = self.branch_dropout(self.common_norm(common_hidden_raw))
        balanced_hidden = self.branch_dropout(self.balanced_norm(balanced_hidden_raw))

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
            "balanced_hidden_raw": balanced_hidden_raw,
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
        }

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_dict(x, domain_ids)["fused_logits"]

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_dict(x, domain_ids)

        task_loss = F.binary_cross_entropy_with_logits(forward_dict["fused_logits"], labels)
        common_aux_loss = F.binary_cross_entropy_with_logits(forward_dict["common_logits"], labels)
        balanced_aux_loss, balanced_domain_losses, balanced_present_mask = self._balanced_shared_bce(
            forward_dict["balanced_logits"],
            labels,
            domain_ids,
        )
        balanced_domain_adv_loss, balanced_domain_adv_losses, balanced_domain_adv_present_mask = self._balanced_domain_ce(
            forward_dict["balanced_domain_logits"],
            domain_ids,
        )
        common_probe_loss, common_probe_losses, common_probe_present_mask = self._balanced_domain_ce(
            forward_dict["common_domain_logits"],
            domain_ids,
        )

        total_loss = (
            task_loss
            + self.common_aux_weight * common_aux_loss
            + self.balanced_aux_weight * balanced_aux_loss
            + self.balanced_domain_adv_weight * balanced_domain_adv_loss
            + self.common_probe_weight * common_probe_loss
        )

        self._latest_task_loss = task_loss.detach()
        self._latest_common_aux_loss = common_aux_loss.detach()
        self._latest_balanced_aux_loss = balanced_aux_loss.detach()
        self._latest_balanced_domain_adv_loss = balanced_domain_adv_loss.detach()
        self._latest_common_probe_loss = common_probe_loss.detach()
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
            "balanced_domain_losses": balanced_domain_losses.detach(),
            "balanced_present_mask": balanced_present_mask.detach(),
            "balanced_domain_adv_losses": balanced_domain_adv_losses.detach(),
            "balanced_domain_adv_present_mask": balanced_domain_adv_present_mask.detach(),
            "common_probe_losses": common_probe_losses.detach(),
            "common_probe_present_mask": common_probe_present_mask.detach(),
        }
        return total_loss

    def debug_state(self) -> Dict[str, float]:
        if not self._latest_debug:
            return {
                "task_loss": 0.0,
                "common_aux_loss": 0.0,
                "balanced_aux_loss": 0.0,
                "balanced_domain_adv_loss": 0.0,
                "common_probe_loss": 0.0,
            }
        return {
            "task_loss": float(self._latest_task_loss.item()),
            "common_aux_loss": float(self._latest_common_aux_loss.item()),
            "balanced_aux_loss": float(self._latest_balanced_aux_loss.item()),
            "balanced_domain_adv_loss": float(self._latest_balanced_domain_adv_loss.item()),
            "common_probe_loss": float(self._latest_common_probe_loss.item()),
        }

    def _domain_loss_stats(self, losses: torch.Tensor, present_mask: torch.Tensor) -> Tuple[float, float]:
        if present_mask.any():
            active_losses = losses[present_mask]
            return float(active_losses.mean().item()), float((active_losses.max() - active_losses.min()).item())
        return 0.0, 0.0

    def _domain_accuracy_entropy(self, logits: torch.Tensor, domain_ids: torch.Tensor) -> Tuple[float, float]:
        if logits.numel() == 0:
            return 0.0, 0.0
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        acc = float((preds == domain_ids).float().mean().item())
        entropy = float((-probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean().item())
        return acc, entropy

    def contribution_state(self) -> Dict[str, float]:
        if not self._latest_debug:
            return {
                "specific_var": 0.0,
                "common_var": 0.0,
                "balanced_var": 0.0,
                "fused_var": 0.0,
                "common_ratio": 0.0,
                "balanced_ratio": 0.0,
                "common_logit_var": 0.0,
                "balanced_logit_var": 0.0,
                "fused_logit_var": 0.0,
                "common_domain_acc": 0.0,
                "balanced_domain_acc": 0.0,
                "common_domain_entropy": 0.0,
                "balanced_domain_entropy": 0.0,
                "common_probe_gap": 0.0,
                "balanced_adv_gap": 0.0,
                "balanced_aux_gap": 0.0,
                "common_gated_var": 0.0,
                "balanced_gated_var": 0.0,
                "specific_gated_var": 0.0,
                "gate_specific_mean": 0.0,
                "gate_common_mean": 0.0,
                "gate_balanced_mean": 0.0,
                "gate_specific_var": 0.0,
                "gate_common_var": 0.0,
                "gate_balanced_var": 0.0,
                "common_low_var_ratio": 0.0,
                "balanced_low_var_ratio": 0.0,
                "fused_low_var_ratio": 0.0,
            }

        domain_ids = self._latest_debug["domain_ids"].long()
        specific_hidden = self._latest_debug["specific_hidden"].float()
        common_hidden = self._latest_debug["common_hidden"].float()
        balanced_hidden = self._latest_debug["balanced_hidden"].float()
        fused_hidden = self._latest_debug["fused_hidden"].float()
        common_logits = self._latest_debug["common_logits"].float()
        balanced_logits = self._latest_debug["balanced_logits"].float()
        fused_logits = self._latest_debug["fused_logits"].float()
        common_domain_logits = self._latest_debug["common_domain_logits"].float()
        balanced_domain_logits = self._latest_debug["balanced_domain_logits"].float()
        gate_weights = self._latest_debug["gate_weights"].float()
        specific_gated = self._latest_debug["specific_gated"].float()
        common_gated = self._latest_debug["common_gated"].float()
        balanced_gated = self._latest_debug["balanced_gated"].float()

        total_var = (
            float(specific_hidden.var(unbiased=False).item())
            + float(common_hidden.var(unbiased=False).item())
            + float(balanced_hidden.var(unbiased=False).item())
            + 1e-12
        )
        specific_var = float(specific_hidden.var(unbiased=False).item())
        common_var = float(common_hidden.var(unbiased=False).item())
        balanced_var = float(balanced_hidden.var(unbiased=False).item())
        fused_var = float(fused_hidden.var(unbiased=False).item())

        common_probe_mean, common_probe_gap = self._domain_loss_stats(
            self._latest_debug["common_probe_losses"].float(),
            self._latest_debug["common_probe_present_mask"],
        )
        balanced_adv_mean, balanced_adv_gap = self._domain_loss_stats(
            self._latest_debug["balanced_domain_adv_losses"].float(),
            self._latest_debug["balanced_domain_adv_present_mask"],
        )
        balanced_aux_mean, balanced_aux_gap = self._domain_loss_stats(
            self._latest_debug["balanced_domain_losses"].float(),
            self._latest_debug["balanced_present_mask"],
        )
        common_domain_acc, common_domain_entropy = self._domain_accuracy_entropy(common_domain_logits, domain_ids)
        balanced_domain_acc, balanced_domain_entropy = self._domain_accuracy_entropy(balanced_domain_logits, domain_ids)

        common_per_dim_var = common_hidden.var(dim=0, unbiased=False)
        balanced_per_dim_var = balanced_hidden.var(dim=0, unbiased=False)
        fused_per_dim_var = fused_hidden.var(dim=0, unbiased=False)

        result = {
            "specific_var": specific_var,
            "common_var": common_var,
            "balanced_var": balanced_var,
            "fused_var": fused_var,
            "common_ratio": common_var / total_var,
            "balanced_ratio": balanced_var / total_var,
            "common_logit_var": float(common_logits.var(unbiased=False).item()),
            "balanced_logit_var": float(balanced_logits.var(unbiased=False).item()),
            "fused_logit_var": float(fused_logits.var(unbiased=False).item()),
            "common_domain_acc": common_domain_acc,
            "balanced_domain_acc": balanced_domain_acc,
            "common_domain_entropy": common_domain_entropy,
            "balanced_domain_entropy": balanced_domain_entropy,
            "common_probe_mean": common_probe_mean,
            "common_probe_gap": common_probe_gap,
            "balanced_adv_mean": balanced_adv_mean,
            "balanced_adv_gap": balanced_adv_gap,
            "balanced_aux_mean": balanced_aux_mean,
            "balanced_aux_gap": balanced_aux_gap,
            "specific_gated_var": float(specific_gated.var(unbiased=False).item()),
            "common_gated_var": float(common_gated.var(unbiased=False).item()),
            "balanced_gated_var": float(balanced_gated.var(unbiased=False).item()),
            "gate_specific_mean": float(gate_weights[:, 0].mean().item()),
            "gate_common_mean": float(gate_weights[:, 1].mean().item()),
            "gate_balanced_mean": float(gate_weights[:, 2].mean().item()),
            "gate_specific_var": float(gate_weights[:, 0].var(unbiased=False).item()),
            "gate_common_var": float(gate_weights[:, 1].var(unbiased=False).item()),
            "gate_balanced_var": float(gate_weights[:, 2].var(unbiased=False).item()),
            "common_low_var_ratio": float((common_per_dim_var < 1e-6).float().mean().item()),
            "balanced_low_var_ratio": float((balanced_per_dim_var < 1e-6).float().mean().item()),
            "fused_low_var_ratio": float((fused_per_dim_var < 1e-6).float().mean().item()),
        }

        for domain_idx in range(self.num_domains):
            mask = domain_ids == domain_idx
            prefix = "d%d_" % domain_idx
            if mask.any():
                domain_gate = gate_weights[mask]
                result[prefix + "gate_specific_mean"] = float(domain_gate[:, 0].mean().item())
                result[prefix + "gate_common_mean"] = float(domain_gate[:, 1].mean().item())
                result[prefix + "gate_balanced_mean"] = float(domain_gate[:, 2].mean().item())
            else:
                result[prefix + "gate_specific_mean"] = 0.0
                result[prefix + "gate_common_mean"] = 0.0
                result[prefix + "gate_balanced_mean"] = 0.0
        return result

    def _record_debug_tensors(self, recorder) -> None:
        if recorder is None or not self._latest_debug:
            return
        recorder.record("ple_balanced_v3_specific_hidden", self._latest_debug["specific_hidden"])
        recorder.record("ple_balanced_v3_common_hidden", self._latest_debug["common_hidden"])
        recorder.record("ple_balanced_v3_balanced_hidden", self._latest_debug["balanced_hidden"])
        recorder.record("ple_balanced_v3_fused_hidden", self._latest_debug["fused_hidden"])
        recorder.record("ple_balanced_v3_gate_weights", self._latest_debug["gate_weights"])
        recorder.record("ple_balanced_v3_common_logits", self._latest_debug["common_logits"].unsqueeze(-1))
        recorder.record("ple_balanced_v3_balanced_logits", self._latest_debug["balanced_logits"].unsqueeze(-1))
        recorder.record("ple_balanced_v3_fused_logits", self._latest_debug["fused_logits"].unsqueeze(-1))

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        loss = self.calculate_loss(batch_interaction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        self._record_debug_tensors(ctx.recorder)
        return float(loss.item())

    def on_train_epoch_start(self, ctx: TrainContext) -> None:
        del ctx

    def on_train_epoch_end(self, ctx: TrainContext) -> None:
        del ctx

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx: TrainContext) -> None:
        del metrics, ctx
        debug = self.debug_state()
        contrib = self.contribution_state()
        gate_domain_parts = []
        for domain_idx in range(self.num_domains):
            gate_domain_parts.append(
                "d%d(%.2f/%.2f/%.2f)" % (
                    domain_idx,
                    contrib["d%d_gate_specific_mean" % domain_idx],
                    contrib["d%d_gate_common_mean" % domain_idx],
                    contrib["d%d_gate_balanced_mean" % domain_idx],
                )
            )
        print(
            "[PLE-Balanced-v3 Debug] "
            f"task={debug['task_loss']:.5f} "
            f"common_aux={debug['common_aux_loss']:.5f} "
            f"balanced_aux={debug['balanced_aux_loss']:.5f} "
            f"balanced_adv={debug['balanced_domain_adv_loss']:.5f} "
            f"common_probe={debug['common_probe_loss']:.5f} "
            f"var(s/c/b/f)={contrib['specific_var']:.4f}/{contrib['common_var']:.4f}/{contrib['balanced_var']:.4f}/{contrib['fused_var']:.4f} "
            f"logit(c/b/f)={contrib['common_logit_var']:.4f}/{contrib['balanced_logit_var']:.4f}/{contrib['fused_logit_var']:.4f} "
            f"gate_mean(s/c/b)={contrib['gate_specific_mean']:.3f}/{contrib['gate_common_mean']:.3f}/{contrib['gate_balanced_mean']:.3f} "
            f"gate_var(s/c/b)={contrib['gate_specific_var']:.4f}/{contrib['gate_common_var']:.4f}/{contrib['gate_balanced_var']:.4f} "
            f"gated_var(s/c/b)={contrib['specific_gated_var']:.4f}/{contrib['common_gated_var']:.4f}/{contrib['balanced_gated_var']:.4f} "
            f"domain_acc(common/balanced)={contrib['common_domain_acc']:.3f}/{contrib['balanced_domain_acc']:.3f} "
            f"domain_ent(common/balanced)={contrib['common_domain_entropy']:.3f}/{contrib['balanced_domain_entropy']:.3f} "
            f"gap(common_probe/bal_aux/bal_adv)={contrib['common_probe_gap']:.4f}/{contrib['balanced_aux_gap']:.4f}/{contrib['balanced_adv_gap']:.4f} "
            f"low_var(common/bal/fused)={contrib['common_low_var_ratio']:.3f}/{contrib['balanced_low_var_ratio']:.3f}/{contrib['fused_low_var_ratio']:.3f} "
            f"gate_by_domain={' '.join(gate_domain_parts)}"
        )
