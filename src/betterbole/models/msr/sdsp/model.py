from __future__ import annotations

import copy
import heapq
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.train.context import TrainContext
from betterbole.core.train.hooks import CustomTrainStepProtocol, TrainerHooksProtocol
from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.msr.ple.layers import select_domain_output
from betterbole.models.utils.common import build_mlp, to_dims


def _expand_domain_values(
        value: Union[int, Sequence[int], None],
        num_domains: int,
        default: int = 1,
) -> List[int]:
    if value is None:
        return [int(default) for _ in range(num_domains)]
    if isinstance(value, int):
        return [int(value) for _ in range(num_domains)]
    values = [int(v) for v in value]
    if len(values) == 1:
        return values * num_domains
    if len(values) != num_domains:
        raise ValueError(f"Expected {num_domains} domain values, got {len(values)}")
    return values


def _resize_feature_axis(x: torch.Tensor, target_width: int) -> torch.Tensor:
    if x.size(1) == target_width:
        return x
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    if x.size(1) > target_width:
        return x[:, :target_width]
    pad_width = target_width - x.size(1)
    padding = x.new_zeros(x.size(0), pad_width)
    return torch.cat([x, padding], dim=1)


class EpsilonGreedyStateSelector:
    def __init__(
            self,
            num_states: int,
            initial_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            decay_rate: float = 0.9,
    ):
        self.num_states = int(num_states)
        self.epsilon = float(initial_epsilon)
        self.min_epsilon = float(min_epsilon)
        self.decay_rate = float(decay_rate)
        self.value_fn = [[None for _ in range(self.num_states)] for _ in range(self.num_states)]
        self.state_list = [[[] for _ in range(self.num_states)] for _ in range(self.num_states)]

    def update_value_function(self, states: List[List[int]], val_auc: Sequence[float]) -> None:
        state_num_list = [max(0, len(states[i]) - 1) for i in range(self.num_states)]
        for i, state_num in enumerate(state_num_list):
            score = val_auc[i]
            if score is None:
                continue
            if self.value_fn[i][state_num] is None or score > self.value_fn[i][state_num]:
                self.value_fn[i][state_num] = score
                self.state_list[i][state_num] = list(states[i])

    def get_next_state(self, model) -> List[List[int]]:
        next_state: List[List[int]] = [None] * self.num_states  # type: ignore[assignment]
        for i in range(self.num_states):
            if random.random() < self.epsilon:
                next_state_num = random.randint(0, self.num_states - 1)
                next_state[i] = model.domain_distance(model.proto_emb, next_state_num, i)
            else:
                candidate_scores = [v if v is not None else float("-inf") for v in self.value_fn[i]]
                next_state_num = int(np.nanargmax(candidate_scores))
                next_state[i] = list(self.state_list[i][next_state_num])
        return next_state

    def decay_temperature(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


class SDSPBaseModel(MSRModel, TrainerHooksProtocol):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            scaled_batch: Sequence[int],
            proto_num: int = 10,
            proto_gamma: float = 1e-4,
            init_iter: int = 4,
            selector_initial_epsilon: float = 1.0,
            selector_min_epsilon: float = 0.1,
            selector_decay_rate: float = 0.9,
            selector_update_every: int = 2,
    ):
        super().__init__(manager, num_domains)
        self.proto_num = int(proto_num)
        self.proto_gamma = float(proto_gamma)
        self.init_iter = int(init_iter)
        self.selector_update_every = int(selector_update_every)
        self.scaled_batch = [int(v) for v in scaled_batch]

        self.dis_fn = nn.PairwiseDistance(p=2)
        self.sim_domain: List[List[int]] = [list(range(self.num_domains)) for _ in range(self.num_domains)]
        self.sim_domain_best: List[List[int]] = copy.deepcopy(self.sim_domain)
        self.selector = EpsilonGreedyStateSelector(
            num_states=self.num_domains,
            initial_epsilon=selector_initial_epsilon,
            min_epsilon=selector_min_epsilon,
            decay_rate=selector_decay_rate,
        )
        self.proto_emb: List[Optional[torch.Tensor]] = [None for _ in range(self.num_domains)]
        self._best_overall_auc = float("-inf")
        self._latest_debug: Dict[str, torch.Tensor] = {}
        self._latest_proto_loss = torch.tensor(0.0)
        self._latest_domain_auc: List[Optional[float]] = [None for _ in range(self.num_domains)]
        self._latest_overall_auc = None

        self.proto_encoders = nn.ModuleList([
            build_mlp(self.scaled_batch[d], self.proto_num, dropout_rate=0.0, activation="relu", batch_norm=False)
            for d in range(self.num_domains)
        ])
        self.proto_decoders = nn.ModuleList([
            build_mlp(self.proto_num, self.scaled_batch[d], dropout_rate=0.0, activation="relu", batch_norm=False)
            for d in range(self.num_domains)
        ])

    def _proto_encode_decode(
            self,
            domain_hidden: Sequence[Optional[torch.Tensor]],
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], torch.Tensor]:
        ori_sample_emb: List[Optional[torch.Tensor]] = [None for _ in range(self.num_domains)]
        new_sample_emb: List[Optional[torch.Tensor]] = [None for _ in range(self.num_domains)]
        proto_loss = None

        for d in range(self.num_domains):
            hidden = domain_hidden[d]
            if hidden is None or hidden.numel() == 0:
                continue

            hidden = hidden.detach()
            ori_matrix = hidden.transpose(0, 1)
            ori_matrix = _resize_feature_axis(ori_matrix, self.scaled_batch[d])
            proto_latent = self.proto_encoders[d](ori_matrix)
            recon_matrix = self.proto_decoders[d](proto_latent)
            ori_sample_emb[d] = ori_matrix
            new_sample_emb[d] = recon_matrix

            cur_loss = torch.norm(recon_matrix - ori_matrix, p=2)
            proto_loss = cur_loss if proto_loss is None else proto_loss + cur_loss
            self.proto_emb[d] = proto_latent.transpose(0, 1).detach()

        if proto_loss is None:
            proto_loss = torch.zeros((), device=self.proto_encoders[0].net[0].weight.device)
        return ori_sample_emb, new_sample_emb, proto_loss

    def domain_distance(
            self,
            proto_emb: Sequence[Optional[torch.Tensor]],
            sim_num: int,
            target_dom: int,
    ) -> List[int]:
        sim_domain: List[int] = []
        dom_dis: List[Optional[float]] = [None for _ in range(self.num_domains)]
        target_proto = proto_emb[target_dom]
        if target_proto is None:
            return [target_dom]

        for d2 in range(self.num_domains):
            if d2 == target_dom:
                dom_dis[d2] = 0.0
                continue
            proto = proto_emb[d2]
            if proto is None:
                dom_dis[d2] = float("inf")
                continue

            final_dis = 0.0
            for i in range(self.proto_num):
                min_distance = float("inf")
                for j in range(self.proto_num):
                    distance = self.dis_fn(target_proto[i], proto[j])
                    min_distance = min(min_distance, float(distance.item()))
                final_dis += min_distance
            final_dis /= max(1, self.proto_num)
            dom_dis[d2] = final_dis

        valid_scores = [value for value in dom_dis if value is not None]
        if not valid_scores:
            return [target_dom]
        temp_list = heapq.nsmallest(min(sim_num + 1, len(valid_scores)), valid_scores)
        for d2 in range(self.num_domains):
            if dom_dis[d2] in temp_list:
                sim_domain.append(d2)
        if target_dom not in sim_domain:
            sim_domain.insert(0, target_dom)
        return sim_domain

    def _extract_domain_aucs(self, metrics: Optional[Dict]) -> List[Optional[float]]:
        domain_aucs: List[Optional[float]] = [None for _ in range(self.num_domains)]
        if not metrics:
            return domain_aucs
        for d in range(self.num_domains):
            domain_key = f"domain{d}"
            domain_metrics = metrics.get(domain_key)
            if isinstance(domain_metrics, dict) and "auc" in domain_metrics:
                try:
                    domain_aucs[d] = float(domain_metrics["auc"])
                except (TypeError, ValueError):
                    domain_aucs[d] = None
        return domain_aucs

    def _update_domain_state(self, metrics: Optional[Dict], ctx: TrainContext) -> None:
        domain_aucs = self._extract_domain_aucs(metrics)
        self._latest_domain_auc = domain_aucs
        overall_auc = None
        if metrics and isinstance(metrics.get("overall"), dict) and "auc" in metrics["overall"]:
            overall_auc = float(metrics["overall"]["auc"])
        self._latest_overall_auc = overall_auc

        if overall_auc is not None and overall_auc > self._best_overall_auc:
            self._best_overall_auc = overall_auc
            self.sim_domain_best = copy.deepcopy(self.sim_domain)

        if ctx.epoch < self.init_iter:
            return
        if (ctx.epoch % self.selector_update_every) != 0:
            return
        if any(score is None for score in domain_aucs):
            return

        self.selector.update_value_function(self.sim_domain, domain_aucs)  # type: ignore[arg-type]
        self.selector.decay_temperature()
        self.sim_domain = self.selector.get_next_state(self)

    def debug_state(self) -> Dict[str, float]:
        proto_ready = float(all(proto is not None for proto in self.proto_emb))
        proto_norms = []
        for proto in self.proto_emb:
            if proto is None:
                continue
            proto_norms.append(float(proto.norm(p=2).item()))
        avg_proto_norm = float(sum(proto_norms) / len(proto_norms)) if proto_norms else 0.0
        return {
            "proto_loss": float(self._latest_proto_loss.item()),
            "proto_ready": proto_ready,
            "selector_epsilon": float(self.selector.epsilon),
            "avg_proto_norm": avg_proto_norm,
            "best_overall_auc": float(self._best_overall_auc) if self._best_overall_auc != float("-inf") else 0.0,
        }

    def on_train_epoch_start(self, ctx: TrainContext) -> None:
        del ctx

    def on_train_epoch_end(self, ctx: TrainContext) -> None:
        del ctx

    def on_eval_epoch_end(self, metrics: Optional[Dict], ctx: TrainContext) -> None:
        self._update_domain_state(metrics, ctx)
        debug = self.debug_state()
        sim_text = " | ".join([f"d{idx}:{state}" for idx, state in enumerate(self.sim_domain)])
        print(
            f"[SDSP Debug] epoch={ctx.epoch} "
            f"overall_auc={self._latest_overall_auc if self._latest_overall_auc is not None else 'None'} "
            f"proto_loss={debug['proto_loss']:.6f} "
            f"selector_eps={debug['selector_epsilon']:.3f} "
            f"proto_ready={int(debug['proto_ready'])} "
            f"avg_proto_norm={debug['avg_proto_norm']:.3f}"
        )
        print(f"[SDSP Debug] sim_domain={sim_text}")
        if self.sim_domain_best:
            best_text = " | ".join([f"d{idx}:{state}" for idx, state in enumerate(self.sim_domain_best)])
            print(f"[SDSP Debug] best_sim_domain={best_text}")


class SDSPMMoEBackbone(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims: Iterable[int] = (128, 64),
            num_experts: Union[int, Sequence[int], None] = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = int(num_domains)
        self.num_experts = _expand_domain_values(num_experts, self.num_domains, default=1)
        self.output_dim = int(expert_dims[-1])

        self.experts = nn.ModuleList([
            nn.ModuleList([
                build_mlp(
                    input_dim,
                    expert_dims,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    batch_norm=batch_norm,
                )
                for _ in range(self.num_experts[domain_idx])
            ])
            for domain_idx in range(self.num_domains)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, sum(self.num_experts)),
                nn.Softmax(dim=-1),
            )
            for _ in range(self.num_domains)
        ])
        self.expert_ranges: List[List[int]] = []
        start = 0
        for num in self.num_experts:
            self.expert_ranges.append(list(range(start, start + num)))
            start += num

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor, sim_domain: Sequence[Sequence[int]]) -> torch.Tensor:
        expert_outs = []
        for domain_idx in range(self.num_domains):
            for expert in self.experts[domain_idx]:
                expert_outs.append(expert(x).unsqueeze(1))
        expert_outs = torch.cat(expert_outs, dim=1)

        gate_outs = []
        for domain_idx in range(self.num_domains):
            gate = self.gates[domain_idx](x).unsqueeze(-1)
            mask_index = [dom for dom in range(self.num_domains) if dom not in sim_domain[domain_idx]]
            if mask_index:
                mask = torch.zeros_like(gate)
                for mask_dom in mask_index:
                    mask[:, self.expert_ranges[mask_dom]] = -float("inf")
                gate = F.softmax(gate + mask, dim=1)
            gate_outs.append(gate)

        domain_outputs = []
        for domain_idx in range(self.num_domains):
            weighted = torch.mul(gate_outs[domain_idx], expert_outs)
            domain_outputs.append(torch.sum(weighted, dim=1))
        return select_domain_output(domain_outputs, domain_ids)


class SDSPMMoEModel(SDSPBaseModel, CustomTrainStepProtocol):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            scaled_batch: Sequence[int],
            expert_dims: Iterable[int] = (128, 64),
            num_experts: Union[int, Sequence[int], None] = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
            proto_num: int = 10,
            proto_gamma: float = 1e-4,
            init_iter: int = 4,
            selector_initial_epsilon: float = 1.0,
            selector_min_epsilon: float = 0.1,
            selector_decay_rate: float = 0.9,
            selector_update_every: int = 2,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            scaled_batch=scaled_batch,
            proto_num=proto_num,
            proto_gamma=proto_gamma,
            init_iter=init_iter,
            selector_initial_epsilon=selector_initial_epsilon,
            selector_min_epsilon=selector_min_epsilon,
            selector_decay_rate=selector_decay_rate,
            selector_update_every=selector_update_every,
        )
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim

        self.backbone = SDSPMMoEBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            expert_dims=expert_dims,
            num_experts=num_experts,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.backbone.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.apply_xavier_initialization()

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long().view(-1)

    def _forward_debug(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        selected_hidden = self.backbone(x, domain_ids, self.sim_domain)
        logits = self.head(selected_hidden, domain_ids)

        domain_hidden: List[Optional[torch.Tensor]] = [None for _ in range(self.num_domains)]
        for d in range(self.num_domains):
            domain_mask = domain_ids == d
            if domain_mask.any():
                domain_hidden[d] = self.backbone(x, torch.full_like(domain_ids, d), self.sim_domain)[domain_mask]
        return {
            "selected_hidden": selected_hidden,
            "logits": logits,
            "domain_hidden": domain_hidden,
        }

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_debug(x, domain_ids)["logits"]

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_debug(x, domain_ids)
        return F.binary_cross_entropy_with_logits(forward_dict["logits"], labels)

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        labels = batch_interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(batch_interaction)
        forward_dict = self._forward_debug(x, domain_ids)
        ori_sample_emb, new_sample_emb, proto_loss = self._proto_encode_decode(forward_dict["domain_hidden"])

        task_loss = F.binary_cross_entropy_with_logits(forward_dict["logits"], labels)
        total_loss = task_loss + self.proto_gamma * proto_loss
        self._latest_proto_loss = proto_loss.detach()
        self._latest_debug = {
            "logits": forward_dict["logits"].detach(),
            "selected_hidden": forward_dict["selected_hidden"].detach(),
        }
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        del ori_sample_emb, new_sample_emb
        return float(total_loss.item())


class SDSPPLELayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            expert_dims,
            num_specific_experts: int = 1,
            num_shared_experts: Union[int, Sequence[int]] = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, (input_dim, input_dim))
        self.num_domains = int(num_domains)
        self.num_specific_experts = int(num_specific_experts)
        self.num_shared_experts_list = _expand_domain_values(num_shared_experts, self.num_domains, default=1)
        self.num_shared_experts = sum(self.num_shared_experts_list)
        self.output_dim = int(expert_dims[-1])

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
                nn.Softmax(dim=-1),
            )
            for _ in range(self.num_domains)
        ])
        self.shared_gate = nn.Sequential(
            nn.Linear(input_dim, self.num_domains * self.num_specific_experts + self.num_shared_experts),
            nn.Softmax(dim=-1),
        )

        self.expert_ranges: List[List[int]] = []
        start = self.num_specific_experts
        for num in self.num_shared_experts_list:
            self.expert_ranges.append(list(range(start, start + num)))
            start += num

    def forward(self, task_inputs: List[torch.Tensor], shared_input: torch.Tensor, sim_domain: Sequence[Sequence[int]]):
        task_outputs = []
        all_specific = []
        shared_outputs = [expert(shared_input) for expert in self.shared_experts]

        for domain_idx in range(self.num_domains):
            task_input = task_inputs[domain_idx]
            specific_outputs = [expert(task_input) for expert in self.specific_experts[domain_idx]]
            all_specific.extend(specific_outputs)
            gate = self.task_gates[domain_idx](task_input).unsqueeze(-1)
            mix_inputs = torch.stack(specific_outputs + shared_outputs, dim=1)
            mask_index = [dom for dom in range(self.num_domains) if dom not in sim_domain[domain_idx]]
            if mask_index:
                mask = torch.zeros_like(gate)
                for mask_dom in mask_index:
                    mask[:, self.expert_ranges[mask_dom]] = -float("inf")
                gate = F.softmax(gate + mask, dim=1)
            task_outputs.append(torch.sum(gate * mix_inputs, dim=1))

        shared_gate = self.shared_gate(shared_input).unsqueeze(-1)
        shared_mix = torch.stack(all_specific + shared_outputs, dim=1)
        shared_output = torch.sum(shared_gate * shared_mix, dim=1)
        return task_outputs, shared_output


class SDSPPLEModel(SDSPBaseModel, CustomTrainStepProtocol):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            scaled_batch: Sequence[int],
            expert_dims: Iterable[int] = (128, 64),
            num_levels: int = 2,
            num_specific_experts: int = 1,
            num_shared_experts: Union[int, Sequence[int]] = 1,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
            proto_num: int = 10,
            proto_gamma: float = 1e-4,
            init_iter: int = 4,
            selector_initial_epsilon: float = 1.0,
            selector_min_epsilon: float = 0.1,
            selector_decay_rate: float = 0.9,
            selector_update_every: int = 2,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            scaled_batch=scaled_batch,
            proto_num=proto_num,
            proto_gamma=proto_gamma,
            init_iter=init_iter,
            selector_initial_epsilon=selector_initial_epsilon,
            selector_min_epsilon=selector_min_epsilon,
            selector_decay_rate=selector_decay_rate,
            selector_update_every=selector_update_every,
        )
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim
        self.num_levels = int(num_levels)
        expert_dims = to_dims(expert_dims, (self.input_dim, self.input_dim))
        self.encoder_output_dim = int(expert_dims[-1])

        self.cgc_layers = nn.ModuleList(
            SDSPPLELayer(
                input_dim=self.input_dim if layer_idx == 0 else self.encoder_output_dim,
                num_domains=num_domains,
                expert_dims=expert_dims,
                num_specific_experts=num_specific_experts,
                num_shared_experts=num_shared_experts,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
            for layer_idx in range(self.num_levels)
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.encoder_output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.apply_xavier_initialization()

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long().view(-1)

    def _forward_debug(self, x: torch.Tensor, domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        task_inputs = [x for _ in range(self.num_domains)]
        shared_input = x
        layer_task_outputs: List[List[torch.Tensor]] = []
        layer_shared_outputs: List[torch.Tensor] = []
        for layer in self.cgc_layers:
            task_inputs, shared_input = layer(task_inputs, shared_input, self.sim_domain)
            layer_task_outputs.append(task_inputs)
            layer_shared_outputs.append(shared_input)

        selected_hidden = select_domain_output(task_inputs, domain_ids)
        logits = self.head(selected_hidden, domain_ids)

        domain_hidden: List[Optional[torch.Tensor]] = [None for _ in range(self.num_domains)]
        for d in range(self.num_domains):
            mask = domain_ids == d
            if mask.any():
                domain_hidden[d] = task_inputs[d][mask]

        return {
            "selected_hidden": selected_hidden,
            "logits": logits,
            "domain_hidden": domain_hidden,
            "layer_shared_outputs": torch.stack(layer_shared_outputs, dim=1) if layer_shared_outputs else None,
        }

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_debug(x, domain_ids)["logits"]

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_debug(x, domain_ids)
        return F.binary_cross_entropy_with_logits(forward_dict["logits"], labels)

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        labels = batch_interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(batch_interaction)
        forward_dict = self._forward_debug(x, domain_ids)
        ori_sample_emb, new_sample_emb, proto_loss = self._proto_encode_decode(forward_dict["domain_hidden"])
        task_loss = F.binary_cross_entropy_with_logits(forward_dict["logits"], labels)
        total_loss = task_loss + self.proto_gamma * proto_loss
        self._latest_proto_loss = proto_loss.detach()
        self._latest_debug = {
            "logits": forward_dict["logits"].detach(),
            "selected_hidden": forward_dict["selected_hidden"].detach(),
        }
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()
        del ori_sample_emb, new_sample_emb
        return float(total_loss.item())
