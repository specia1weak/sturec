from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from betterbole.core.train.context import TrainContext
from betterbole.emb import SchemaManager
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead
from betterbole.models.utils.general import MLP


class ParetoModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Iterable[int] = (256, 128),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            tower_hidden_dims: Iterable[int] = None,
            tower_dropout_rate: float = 0.2,
            weight_temperature: float = 1.0,
            ema_decay: float = 0.9,
            min_weight: float = 0.2,
            max_weight: float = 0.8,
            warmup_steps: int = 100,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = self.input_view.embedding_dim

        hidden_dims = tuple(hidden_dims)
        if len(hidden_dims) < 1:
            raise ValueError("ParetoModel requires at least one hidden dim.")

        self.encoder = MLP(
            self.input_dim,
            *hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.output_dim = int(hidden_dims[-1])
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )

        self.weight_temperature = float(weight_temperature)
        self.ema_decay = float(ema_decay)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.warmup_steps = int(warmup_steps)

        if not (0.0 < self.weight_temperature):
            raise ValueError("weight_temperature must be positive.")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in [0, 1).")
        if not (0.0 < self.min_weight <= self.max_weight <= 1.0):
            raise ValueError("Require 0 < min_weight <= max_weight <= 1.")

        self._validate_bounds()
        self.register_buffer("_ema_domain_loss", torch.ones(num_domains))
        self.register_buffer("_domain_seen", torch.zeros(num_domains, dtype=torch.bool))
        self.latest_domain_weights: Optional[dict[int, float]] = None
        self.latest_domain_losses: Optional[dict[int, float]] = None

    def _validate_bounds(self) -> None:
        if self.num_domains > 1 and self.num_domains * self.min_weight > 1.0 + 1e-6:
            raise ValueError(
                "min_weight is too large to form a valid simplex over all domains."
            )
        if self.num_domains * self.max_weight < 1.0 - 1e-6:
            raise ValueError(
                "max_weight is too small to form a valid simplex over all domains."
            )

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features, domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float().view(-1)
        logits = self.predict(interaction).view(-1)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def _update_domain_ema(self, domain_id: int, loss_value: torch.Tensor) -> None:
        previous = self._ema_domain_loss[domain_id]
        if bool(self._domain_seen[domain_id]):
            updated = self.ema_decay * previous + (1.0 - self.ema_decay) * loss_value
        else:
            updated = loss_value
        self._ema_domain_loss[domain_id] = updated
        self._domain_seen[domain_id] = True

    def _project_weights(self, raw_weights: torch.Tensor) -> torch.Tensor:
        if raw_weights.numel() == 1:
            return torch.ones_like(raw_weights)

        lower = float(self.min_weight)
        upper = float(self.max_weight)
        weights = raw_weights.clamp(min=1e-8)
        weights = weights / weights.sum()

        for _ in range(8):
            weights = weights.clamp(min=lower, max=upper)
            total = weights.sum()
            if torch.isclose(total, torch.tensor(1.0, device=weights.device), atol=1e-6):
                if torch.all(weights >= lower - 1e-6) and torch.all(weights <= upper + 1e-6):
                    break

            fixed_mask = (weights <= lower + 1e-8) | (weights >= upper - 1e-8)
            free_mask = ~fixed_mask
            if not bool(free_mask.any()):
                weights = weights / weights.sum()
                break

            remaining = 1.0 - weights[fixed_mask].sum()
            if remaining <= 0:
                weights[free_mask] = lower
                weights = weights / weights.sum()
                continue

            free_raw = raw_weights[free_mask].clamp(min=1e-8)
            free_weights = free_raw / free_raw.sum() * remaining
            weights[free_mask] = free_weights

        weights = weights.clamp(min=lower, max=upper)
        weights = weights / weights.sum()
        return weights

    def _domain_weight(self, domain_ids: torch.Tensor, ctx: TrainContext) -> torch.Tensor:
        num_observed = int(domain_ids.numel())
        if num_observed == 1 or int(ctx.global_step) < self.warmup_steps:
            return torch.full((num_observed,), 1.0 / num_observed, device=domain_ids.device)

        ema_stats = self._ema_domain_loss[domain_ids]
        raw_weights = torch.softmax(ema_stats / self.weight_temperature, dim=0)
        return self._project_weights(raw_weights)

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        labels = batch_interaction[self.LABEL].float().view(-1)
        x, domain_ids = self.encode_features(batch_interaction)
        logits = self.forward(x, domain_ids).view(-1)

        observed_domains = torch.unique(domain_ids, sorted=True)
        domain_losses = []
        for domain_id in observed_domains.tolist():
            mask = domain_ids == domain_id
            if not bool(mask.any()):
                continue
            loss_value = F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            self._update_domain_ema(domain_id, loss_value.detach())
            domain_losses.append((domain_id, loss_value))

        if not domain_losses:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            ctx.optimizer.step()
            self.latest_domain_weights = None
            self.latest_domain_losses = None
            return float(loss.item())

        ordered_domain_ids = torch.tensor(
            [domain_id for domain_id, _ in domain_losses],
            device=domain_ids.device,
            dtype=torch.long,
        )
        per_domain_loss = torch.stack([loss for _, loss in domain_losses])
        weights = self._domain_weight(ordered_domain_ids, ctx)
        weighted_loss = torch.sum(weights * per_domain_loss)

        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()

        self.latest_domain_weights = {
            int(domain_id): float(weight)
            for domain_id, weight in zip(ordered_domain_ids.tolist(), weights.detach().cpu().tolist())
        }
        self.latest_domain_losses = {
            int(domain_id): float(loss.detach().cpu().item())
            for domain_id, loss in domain_losses
        }
        return float(weighted_loss.item())
