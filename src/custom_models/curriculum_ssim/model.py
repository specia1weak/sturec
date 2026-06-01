from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import math

import torch
import torch.nn.functional as F
from torch import nn

from betterbole.core.train.context import TrainContext
from betterbole.emb import SchemaManager
from betterbole.models.utils.common import build_mlp, to_dims

from betterbole.models.msr.ssim import SSIMModel
from custom_models.curriculum_ssim.curriculum import CurriculumScheduler


def _specificity_weights(
        affinity_logits: torch.Tensor,
        domain_ids: torch.Tensor,
        temperature: float,
) -> torch.Tensor:
    num_domains = affinity_logits.shape[1]
    one_hot = F.one_hot(domain_ids.view(-1).long(), num_classes=num_domains).float()
    own_logits = (affinity_logits * one_hot).sum(dim=1, keepdim=True)
    if num_domains == 1:
        return torch.ones_like(own_logits)
    masked_others = affinity_logits.masked_fill(one_hot.bool(), float("-inf"))
    other_reference = torch.logsumexp(masked_others, dim=1, keepdim=True) - math.log(num_domains - 1)
    return torch.sigmoid((own_logits - other_reference) / max(temperature, 1e-6))


class CurriculumSSIMModel(SSIMModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            branch_hidden_dims: Iterable[int] = (256, 128),
            mask_hidden_dims: Iterable[int] = (256, 128),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            lambda1: float = 1.0,
            threshold: float = 0.49,
            search_epochs: int = 1,
            mask_init_temperature: float = 1.0,
            mask_final_temperature: float = 1000.0,
            search_steps_hint: int = 2600,
            retrain_topk: Optional[float] = None,
            curriculum_enabled: bool = True,
            curriculum_start_ratio: float = 0.8,
            curriculum_end_ratio: float = 0.2,
            curriculum_schedule: str = "cosine",
            curriculum_start_temp: float = 5.0,
            curriculum_end_temp: float = 1.0,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            branch_hidden_dims=branch_hidden_dims,
            mask_hidden_dims=mask_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            lambda1=lambda1,
            threshold=threshold,
            search_epochs=search_epochs,
            mask_init_temperature=mask_init_temperature,
            mask_final_temperature=mask_final_temperature,
            search_steps_hint=search_steps_hint,
            retrain_topk=retrain_topk,
        )
        self.curriculum_enabled = bool(curriculum_enabled)
        self.curriculum_scheduler = CurriculumScheduler(
            total_steps=search_steps_hint,
            start_ratio=curriculum_start_ratio,
            end_ratio=curriculum_end_ratio,
            schedule_type=curriculum_schedule,
            start_temp=curriculum_start_temp,
            end_temp=curriculum_end_temp,
        )

        self._affinity_proj = nn.Linear(self.input_dim, self.num_domains)
        self._curriculum_total_steps_initialized = False
        self.curriculum_share_floor = 0.10

    def _curriculum_total_steps(self, ctx: TrainContext) -> int:
        return max(self.search_steps_hint, int(ctx.cfg.max_epochs) * self.search_steps_hint)

    def _ensure_curriculum_schedule(self, ctx: TrainContext) -> None:
        if self._curriculum_total_steps_initialized:
            return
        self.curriculum_scheduler.total_steps = self._curriculum_total_steps(ctx)
        self._curriculum_total_steps_initialized = True

    def _compute_curriculum_targets(
            self,
            x: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.curriculum_enabled:
            zeros = torch.zeros(x.size(0), 1, device=x.device)
            return zeros, zeros

        affinity_logits = self._affinity_proj(x)
        temperature = self.curriculum_scheduler.get_affinity_temperature()
        specificity = _specificity_weights(affinity_logits, domain_ids, temperature)
        schedule_ratio = self.curriculum_scheduler.get_sharing_ratio()
        share_target = self.curriculum_share_floor + (schedule_ratio - self.curriculum_share_floor) * (1.0 - specificity)
        share_target = share_target.clamp(0.0, 1.0)
        return specificity, share_target

    def _compute_share_mask(
            self,
            mask_logits: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.curriculum_enabled or self.in_retrain_phase:
            return super()._compute_share_mask(mask_logits, domain_ids)
        selected_logits = mask_logits.gather(dim=1, index=domain_ids.view(-1, 1))
        share_temperature = self.curriculum_scheduler.get_affinity_temperature()
        share_prob = torch.sigmoid(selected_logits / max(share_temperature, 1e-6))
        return share_prob, share_prob

    def _phase_loss(
            self,
            branch_logits: torch.Tensor,
            share_mask: torch.Tensor,
            share_prob: torch.Tensor,
            labels: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.curriculum_enabled or self.in_retrain_phase:
            return super()._phase_loss(branch_logits, share_mask, share_prob, labels, domain_ids)

        per_branch_loss = []
        for domain_idx in range(self.num_domains):
            per_branch_loss.append(
                F.binary_cross_entropy_with_logits(
                    branch_logits[:, domain_idx],
                    labels,
                    reduction="none",
                )
            )
        per_branch_loss_tensor = torch.stack(per_branch_loss, dim=1)
        own_branch_loss = per_branch_loss_tensor.gather(dim=1, index=domain_ids.view(-1, 1))
        if self.num_domains > 1:
            cross_domain_loss = (
                per_branch_loss_tensor.sum(dim=1, keepdim=True) - own_branch_loss
            ) / float(self.num_domains - 1)
        else:
            cross_domain_loss = own_branch_loss

        specificity, share_target = self._compute_curriculum_targets(
            self._latest_shared_x,
            domain_ids,
        )
        share_weight = 0.5 * share_prob.clamp(0.0, 1.0) + 0.5 * share_target
        main_loss = ((1.0 - share_weight) * own_branch_loss + share_weight * cross_domain_loss).mean()

        share_reg = F.binary_cross_entropy(
            share_prob.clamp(1e-6, 1 - 1e-6),
            share_target.clamp(1e-6, 1 - 1e-6),
        )
        specificity_reg = F.mse_loss(1.0 - share_prob, specificity)
        total_loss = main_loss + self.lambda1 * share_reg + 0.05 * specificity_reg
        return total_loss, own_branch_loss.mean(), share_reg

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        if not self.curriculum_enabled or self.in_retrain_phase:
            return super().custom_train_step(batch_interaction, ctx)

        self._ensure_curriculum_schedule(ctx)
        labels = batch_interaction[self.LABEL].float().view(-1)
        shared_x, mask_x, domain_ids = self.encode_features(batch_interaction)
        self._latest_shared_x = shared_x

        forward_dict = self._forward_dict(shared_x, mask_x, domain_ids)
        total_loss, own_loss, share_reg = self._phase_loss(
            forward_dict["branch_logits"],
            forward_dict["share_mask"],
            forward_dict["share_prob"],
            labels,
            domain_ids,
        )
        specificity, share_target = self._compute_curriculum_targets(shared_x, domain_ids)

        self._latest_debug = {
            "share_mask": forward_dict["share_mask"].detach(),
            "share_prob": forward_dict["share_prob"].detach(),
            "branch_logits": forward_dict["branch_logits"].detach(),
            "selected_logits": forward_dict["selected_logits"].detach(),
            "domain_ids": domain_ids.detach(),
            "own_loss": own_loss.detach().view(1),
            "share_reg": share_reg.detach().view(1),
            "specificity": specificity.detach(),
            "share_target": share_target.detach(),
        }

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        ctx.optimizer.step()

        self.curriculum_scheduler.step()
        return float(total_loss.item())

    def predict(self, interaction):
        shared_x, mask_x, domain_ids = self.encode_features(interaction)
        return self.forward(shared_x, mask_x, domain_ids)

    def on_eval_epoch_end(self, metrics, ctx):
        if not self.curriculum_enabled:
            super().on_eval_epoch_end(metrics, ctx)
            return

        debug = self._debug_state()
        phase = self.curriculum_scheduler.get_phase()
        sharing = self.curriculum_scheduler.get_sharing_ratio()
        temp = self.curriculum_scheduler.get_affinity_temperature()

        print(
            "[CurriculumSSIM Debug] "
            f"phase={phase} "
            f"epoch={ctx.epoch} "
            f"share_rate={debug['share_rate']:.3f} "
            f"share_target={debug['share_target_mean']:.3f} "
            f"specificity={debug['specificity_mean']:.3f} "
            f"sharing_ratio={sharing:.3f} "
            f"affinity_temp={temp:.3f} "
            f"own_loss={debug['own_loss']:.5f} "
            f"logit_var={debug['selected_logit_var']:.5f} "
        )
        super().on_eval_epoch_end(metrics, ctx)

    def _debug_state(self) -> Dict[str, float]:
        result = super()._debug_state()
        if self.curriculum_enabled:
            result["sharing_ratio"] = self.curriculum_scheduler.get_sharing_ratio()
            result["affinity_temp"] = self.curriculum_scheduler.get_affinity_temperature()
            if "share_target" in self._latest_debug:
                result["share_target_mean"] = float(self._latest_debug["share_target"].float().mean().item())
            else:
                result["share_target_mean"] = 0.0
            if "specificity" in self._latest_debug:
                result["specificity_mean"] = float(self._latest_debug["specificity"].float().mean().item())
            else:
                result["specificity_mean"] = 0.0
        return result
