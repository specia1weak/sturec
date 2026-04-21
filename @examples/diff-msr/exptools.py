import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from betterbole.core.interaction import Interaction
from betterbole.emb import SchemaManager
from betterbole.evaluate.evaluator import Evaluator
from betterbole.models.generative.diffusion.diffusions import CDCDRMlpDiffusion

from expmodel import DiffMSRIdModel, DomainClassifier
from settings import DiffMSRExperimentSettings


def make_model(
        manager: SchemaManager,
        domain_specs: Tuple[Tuple[int, int], ...],
        cfg: DiffMSRExperimentSettings,
) -> DiffMSRIdModel:
    backbone_params = dict(cfg.backbone_params or {})
    if cfg.backbone_name.lower() in {"sharedbottom", "shared_bottom"}:
        backbone_params.setdefault("hidden_dims", cfg.shared_bottom_hidden_dims)
        backbone_params.setdefault("dropout_rate", cfg.head_dropout)
        backbone_params.setdefault("batch_norm", True)
    return DiffMSRIdModel(
        manager,
        domain_head_ids=tuple(mapped_domain for _, mapped_domain in domain_specs),
        embedding_size=cfg.emb_dim,
        shared_bottom_hidden_dims=cfg.shared_bottom_hidden_dims,
        domain_head_hidden_dims=cfg.domain_head_hidden_dims,
        head_dropout=cfg.head_dropout,
        backbone_name=cfg.backbone_name,
        backbone_params=backbone_params,
        stage4_freeze_embeddings=cfg.stage4_freeze_embeddings,
        stage4_freeze_shared_bottom=cfg.stage4_freeze_shared_bottom,
    )


def mapped_domain_id(manager: SchemaManager, raw_domain: int) -> int:
    setting = manager.get_setting(manager.domain_field)
    if setting is None:
        raise ValueError("SchemaManager 缺少 domain setting。")
    return int(setting.vocab.get(str(raw_domain), setting.oov_idx))


def print_split_statistics(
        train_samples_lf: pl.LazyFrame,
        train_lf: pl.LazyFrame,
        val_lf: pl.LazyFrame,
        test_lf: pl.LazyFrame,
) -> None:
    print("=" * 50)
    print("DATASET SPLIT STATISTICS (Row Counts)")
    print("=" * 50)
    print(f"train_samples (Pos only, Seq, unused): {train_samples_lf.select(pl.len()).collect().item()}")
    print(f"train_ple     (Pos + Neg):           {train_lf.select(pl.len()).collect().item()}")
    print(f"val_ple       (Pos + Neg):           {val_lf.select(pl.len()).collect().item()}")
    print(f"test_ple      (Pos + Neg):           {test_lf.select(pl.len()).collect().item()}")
    print("=" * 50)


def collect_domain_counts(frame: pl.LazyFrame, domain_field: str) -> dict:
    counts = (
        frame
        .group_by(domain_field)
        .agg(pl.len().alias("count"))
        .sort(domain_field)
        .collect()
    )
    return {
        str(row[domain_field]): int(row["count"])
        for row in counts.to_dicts()
    }


def print_domain_count_statistics(
        train_samples_lf: pl.LazyFrame,
        train_lf: pl.LazyFrame,
        val_lf: pl.LazyFrame,
        test_lf: pl.LazyFrame,
        domain_field: str,
) -> dict:
    summary = {
        "train_samples": collect_domain_counts(train_samples_lf, domain_field),
        "train_ple": collect_domain_counts(train_lf, domain_field),
        "val_ple": collect_domain_counts(val_lf, domain_field),
        "test_ple": collect_domain_counts(test_lf, domain_field),
    }
    print("=" * 50)
    print("DOMAIN INTERACTION COUNTS")
    print("=" * 50)
    for split_name, counts in summary.items():
        print(f"{split_name}: {counts}")
    print("=" * 50)
    return summary


def lf_to_interaction(frame: pl.DataFrame) -> Interaction:
    return Interaction(frame.to_dict(as_series=False))


class InteractionMapDataset(Dataset):
    def __init__(self, interaction: Interaction):
        self.interaction = interaction
        self.columns = interaction.columns

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, index):
        return {key: self.interaction[key][index] for key in self.columns}


def collate_interactions(batch):
    keys = batch[0].keys()
    return Interaction({key: torch.stack([sample[key] for sample in batch], dim=0) for key in keys})


def make_loader(interaction: Interaction, batch_size: int, shuffle: bool, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        InteractionMapDataset(interaction),
        batch_size=batch_size,
        shuffle=shuffle and len(interaction) > 0,
        drop_last=drop_last,
        collate_fn=collate_interactions,
    )


def clone_interaction(interaction: Interaction) -> Interaction:
    return Interaction({key: value.clone() for key, value in interaction.interaction.items()})


def interaction_with_domain(interaction: Interaction, domain_idx: int, domain_field: str) -> Interaction:
    cloned = clone_interaction(interaction)
    cloned[domain_field] = torch.full_like(cloned[domain_field], domain_idx)
    return cloned


def sample_interaction(interaction: Interaction, sample_size: int, device: torch.device) -> Interaction:
    sample_size = max(1, min(sample_size, len(interaction)))
    indices = torch.randint(0, len(interaction), (sample_size,))
    return interaction[indices.tolist()].to(device)


def add_count_fields(metrics: dict, total_count: int, positive_count: float) -> dict:
    metrics["count"] = int(total_count)
    metrics["pos"] = float(positive_count)
    metrics["neg"] = float(total_count - positive_count)
    return metrics


def finite_metric(value: Optional[float]) -> bool:
    return value is not None and value == value


def weighted_average(values, weights) -> float:
    pairs = [
        (float(value), float(weight))
        for value, weight in zip(values, weights)
        if finite_metric(value) and weight > 0
    ]
    if not pairs:
        return float("nan")
    total_weight = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total_weight


def add_domain_aggregate_metrics(metrics: dict, domain_specs: Tuple[Tuple[int, int], ...]) -> dict:
    domain_metrics = [metrics[f"domain{raw_domain}"] for raw_domain, _ in domain_specs]
    auc_values = [domain_metric.get("auc") for domain_metric in domain_metrics]
    logloss_values = [domain_metric.get("logloss") for domain_metric in domain_metrics]
    counts = [domain_metric.get("count", 0) for domain_metric in domain_metrics]
    valid_auc_values = [value for value in auc_values if finite_metric(value)]
    valid_logloss_values = [value for value in logloss_values if finite_metric(value)]
    metrics["domain_macro"] = {
        "auc": sum(valid_auc_values) / len(valid_auc_values) if valid_auc_values else float("nan"),
        "logloss": sum(valid_logloss_values) / len(valid_logloss_values) if valid_logloss_values else float("nan"),
        "count": sum(counts),
    }
    metrics["domain_weighted"] = {
        "auc": weighted_average(auc_values, counts),
        "logloss": weighted_average(logloss_values, counts),
        "count": sum(counts),
    }
    metrics["micro"] = metrics["overall"]
    return metrics


def extract_auc_summary(metrics: Optional[dict]) -> dict:
    if metrics is None:
        return {}
    overall = metrics.get("overall", {})
    domain_weighted = metrics.get("domain_weighted", {})
    domain_macro = metrics.get("domain_macro", {})
    return {
        "micro_auc": overall.get("auc"),
        "weighted_auc": domain_weighted.get("auc"),
        "macro_auc": domain_macro.get("auc"),
        "micro_logloss": overall.get("logloss"),
        "weighted_logloss": domain_weighted.get("logloss"),
        "macro_logloss": domain_macro.get("logloss"),
        "count": overall.get("count", domain_weighted.get("count")),
    }


def summarize_best_by_domain(best_by_domain: dict) -> dict:
    return {
        str(raw_domain): {
            key: value
            for key, value in info.items()
            if key != "state"
        }
        for raw_domain, info in best_by_domain.items()
    }


def strip_model_states(value):
    if isinstance(value, dict):
        return {
            str(key): strip_model_states(item)
            for key, item in value.items()
            if key != "state"
        }
    if isinstance(value, (list, tuple)):
        return [strip_model_states(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_experiment_record(record: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(record, fp, ensure_ascii=False, indent=2, allow_nan=True)


def format_auc_summary(metrics: dict, domain_specs: Tuple[Tuple[int, int], ...]) -> str:
    return ", ".join(
        [f"domain{raw_domain}={metrics[f'domain{raw_domain}'].get('auc', float('nan')):.6f}" for raw_domain, _ in domain_specs]
        + [
            f"domain_macro={metrics['domain_macro'].get('auc', float('nan')):.6f}",
            f"domain_weighted={metrics['domain_weighted'].get('auc', float('nan')):.6f}",
            f"overall_global={metrics['overall'].get('auc', float('nan')):.6f}",
        ]
    )


def empty_transfer_diagnostic(cfg: DiffMSRExperimentSettings) -> dict:
    return {
        "candidates": 0,
        "selected": 0,
        "scan_count": 0,
        "pred_sum": 0.0,
        "pred_min": float("inf"),
        "pred_max": -float("inf"),
        "below_threshold_counts": {str(threshold): 0 for threshold in cfg.transfer_diagnostic_thresholds},
        "selected_step_counts": {},
    }


def merge_transfer_diagnostic(target: dict, source: dict) -> None:
    target["candidates"] += source.get("candidates", 0)
    target["selected"] += source.get("selected", 0)
    target["scan_count"] += source.get("scan_count", 0)
    target["pred_sum"] += source.get("pred_sum", 0.0)
    if source.get("scan_count", 0) > 0:
        target["pred_min"] = min(target["pred_min"], source.get("pred_min", float("inf")))
        target["pred_max"] = max(target["pred_max"], source.get("pred_max", -float("inf")))
    for threshold_key, count in source.get("below_threshold_counts", {}).items():
        target["below_threshold_counts"][threshold_key] = (
            target["below_threshold_counts"].get(threshold_key, 0) + count
        )
    for step, count in source.get("selected_step_counts", {}).items():
        target["selected_step_counts"][step] = target["selected_step_counts"].get(step, 0) + count


def format_transfer_diagnostic(diag: dict, cfg: DiffMSRExperimentSettings) -> str:
    scan_count = diag.get("scan_count", 0)
    candidates = diag.get("candidates", 0)
    selected = diag.get("selected", 0)
    pred_mean = diag["pred_sum"] / scan_count if scan_count else float("nan")
    pred_min = diag["pred_min"] if scan_count else float("nan")
    pred_max = diag["pred_max"] if scan_count else float("nan")
    below_parts = []
    for threshold in cfg.transfer_diagnostic_thresholds:
        key = str(threshold)
        below_rate = diag["below_threshold_counts"].get(key, 0) / scan_count if scan_count else 0.0
        below_parts.append(f"below{threshold:g}={below_rate:.2%}")
    top_steps = sorted(
        diag["selected_step_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:cfg.transfer_diagnostic_top_steps]
    top_step_text = "none" if not top_steps else ",".join(f"{step}:{count}" for step, count in top_steps)
    selected_rate = selected / candidates if candidates else 0.0
    return (
        f"cand={candidates}, selected={selected} ({selected_rate:.2%}), "
        f"scan_pred_mean={pred_mean:.4f}, scan_pred_range=[{pred_min:.4f},{pred_max:.4f}], "
        f"{' '.join(below_parts)}, top_steps={top_step_text}"
    )


def evaluate_auc_logloss(
        model: DiffMSRIdModel,
        data_loader: DataLoader,
        domain_idx: Optional[int],
        device: torch.device,
        epoch: int,
        name: str,
) -> dict:
    evaluator = Evaluator("AUC", "logloss")
    total_count = 0
    positive_count = 0.0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            if domain_idx is None:
                subset = batch
                scores = model.predict(subset)
            else:
                mask = batch[model.DOMAIN].long() == domain_idx
                if not mask.any().item():
                    continue
                subset = batch[mask]
                scores = model.predict(subset, domain_override=domain_idx)
            labels = subset[model.LABEL].float()
            total_count += len(subset)
            positive_count += labels.sum().item()
            evaluator.collect_pointwise(
                subset[model.USER].detach().cpu(),
                labels.detach().cpu(),
                scores.detach().cpu(),
            )
    metrics = add_count_fields(evaluator.summary(), total_count, positive_count)
    print(f"[Metrics][{name}][Epoch {epoch}] {metrics}")
    return metrics


def evaluate_all_domain_metrics(
        model: DiffMSRIdModel,
        data_loader: DataLoader,
        domain_specs: Tuple[Tuple[int, int], ...],
        device: torch.device,
        epoch: int,
        name: str,
) -> dict:
    metrics = {}
    for raw_domain, mapped_domain in domain_specs:
        metrics[f"domain{raw_domain}"] = evaluate_auc_logloss(
            model,
            data_loader,
            mapped_domain,
            device,
            epoch,
            f"{name}-domain{raw_domain}",
        )
    metrics["overall"] = evaluate_auc_logloss(
        model,
        data_loader,
        None,
        device,
        epoch,
        f"{name}-overall",
    )
    add_domain_aggregate_metrics(metrics, domain_specs)
    print(f"[Metrics][{name}-domain_macro][Epoch {epoch}] {metrics['domain_macro']}")
    print(f"[Metrics][{name}-domain_weighted][Epoch {epoch}] {metrics['domain_weighted']}")
    print(f"[AUCSummary][{name}][Epoch {epoch}] {format_auc_summary(metrics, domain_specs)}")
    return metrics


def evaluate_branch_pair(
        baseline_model: DiffMSRIdModel,
        augment_model: DiffMSRIdModel,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        domain_specs: Tuple[Tuple[int, int], ...],
        device: torch.device,
        epoch: int,
) -> Tuple[dict, dict, dict, dict]:
    baseline_valid = evaluate_all_domain_metrics(
        baseline_model, valid_loader, domain_specs, device, epoch, "baseline-valid"
    )
    baseline_test = evaluate_all_domain_metrics(
        baseline_model, test_loader, domain_specs, device, epoch, "baseline-test"
    )
    augment_valid = evaluate_all_domain_metrics(
        augment_model, valid_loader, domain_specs, device, epoch, "augment-valid"
    )
    augment_test = evaluate_all_domain_metrics(
        augment_model, test_loader, domain_specs, device, epoch, "augment-test"
    )
    return baseline_valid, baseline_test, augment_valid, augment_test


def train_backbone_stage(
        model: DiffMSRIdModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        domain_specs: Tuple[Tuple[int, int], ...],
        device: torch.device,
        cfg: DiffMSRExperimentSettings,
) -> Tuple[DiffMSRIdModel, dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate_stage1, weight_decay=cfg.weight_decay)
    best_state = deepcopy(model.state_dict())
    best_auc = -float("inf")
    best_epoch = None
    best_valid_metrics = None
    best_test_metrics = None
    for epoch in range(cfg.stage1_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        avg_loss = total_loss / max(batch_count, 1)
        metrics = evaluate_all_domain_metrics(model, valid_loader, domain_specs, device, epoch, "stage1-valid")
        auc = metrics["overall"].get("auc", -float("inf"))
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_valid_metrics = metrics
            best_test_metrics = evaluate_all_domain_metrics(model, test_loader, domain_specs, device, epoch, "stage1-best-test")
            print(f"[Stage1][Epoch {epoch}] new best overall valid auc={best_auc:.6f}, best_test={best_test_metrics}")
        print(f"[Stage1][Epoch {epoch}] backbone loss={avg_loss:.6f}, best_overall_auc={best_auc:.6f}")
    if best_test_metrics is not None:
        print(f"[Stage1] best-epoch synced test metrics: {best_test_metrics}")
    model.load_state_dict(best_state)
    best_info = {
        "epoch": best_epoch,
        "auc": best_auc,
        "valid": best_valid_metrics,
        "test": best_test_metrics,
        "valid_auc_summary": extract_auc_summary(best_valid_metrics),
        "test_auc_summary": extract_auc_summary(best_test_metrics),
    }
    print(f"[Stage1] best summary: {best_info}")
    return model, best_info


def train_single_diffusion(
        backbone: DiffMSRIdModel,
        diffusion: CDCDRMlpDiffusion,
        train_loader: DataLoader,
        device: torch.device,
        label_name: str,
        cfg: DiffMSRExperimentSettings,
) -> CDCDRMlpDiffusion:
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=cfg.learning_rate_stage2, weight_decay=cfg.weight_decay)
    backbone.eval()
    best_state = deepcopy(diffusion.state_dict())
    best_loss = float("inf")

    for epoch in range(cfg.stage2_epochs):
        diffusion.train()
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            with torch.no_grad():
                x_0 = backbone.embed_user_item_pair(batch)
            optimizer.zero_grad()
            loss = diffusion.calculate_loss(x_0=x_0, y=None)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        avg_loss = total_loss / max(batch_count, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = deepcopy(diffusion.state_dict())
        print(f"[Stage2][{label_name}][Epoch {epoch}] diffusion loss={avg_loss:.6f}, best={best_loss:.6f}")

    diffusion.load_state_dict(best_state)
    return diffusion


def train_classifier_stage(
        backbone: DiffMSRIdModel,
        scheduler_owner: CDCDRMlpDiffusion,
        classifier: DomainClassifier,
        train_loader: DataLoader,
        target_domain_idx: int,
        device: torch.device,
        cfg: DiffMSRExperimentSettings,
) -> DomainClassifier:
    optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.learning_rate_stage3, weight_decay=cfg.weight_decay)
    backbone.eval()
    scheduler = scheduler_owner.scheduler
    max_t = min(cfg.classifier_noise_max_step, scheduler.num_train_timesteps)
    best_state = deepcopy(classifier.state_dict())
    best_loss = float("inf")

    for epoch in range(cfg.stage3_epochs):
        classifier.train()
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            with torch.no_grad():
                pair_embedding = backbone.embed_user_item_pair(batch)
                t = torch.randint(0, max_t, (len(batch),), device=device).long()
                noisy_pair = scheduler.add_noise(pair_embedding, torch.randn_like(pair_embedding), t)
            domain_labels = (batch[backbone.DOMAIN].long() != target_domain_idx).float()
            preds = torch.cat([classifier(pair_embedding), classifier(noisy_pair)], dim=0)
            labels = torch.cat([domain_labels, domain_labels], dim=0)

            optimizer.zero_grad()
            loss = F.binary_cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        avg_loss = total_loss / max(batch_count, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = deepcopy(classifier.state_dict())
        print(f"[Stage3][target={target_domain_idx}][Epoch {epoch}] classifier loss={avg_loss:.6f}, best={best_loss:.6f}")

    classifier.load_state_dict(best_state)
    return classifier


def select_transfer_steps(
        pair_embedding: torch.Tensor,
        classifier: DomainClassifier,
        diffusion: CDCDRMlpDiffusion,
        cfg: DiffMSRExperimentSettings,
) -> Tuple[torch.Tensor, dict]:
    device = pair_embedding.device
    selected_steps = torch.full((pair_embedding.size(0),), -1, dtype=torch.long, device=device)
    pending = torch.ones(pair_embedding.size(0), dtype=torch.bool, device=device)
    noise = torch.randn_like(pair_embedding)
    diagnostics = empty_transfer_diagnostic(cfg)
    diagnostics["candidates"] = pair_embedding.size(0)

    for k in range(cfg.aug_step1, cfg.aug_step2):
        if not pending.any().item():
            break
        t = torch.full((pending.sum().item(),), k, dtype=torch.long, device=device)
        noisy_pair = diffusion.scheduler.add_noise(pair_embedding[pending], noise[pending], t)
        pred = classifier(noisy_pair)
        pred_detached = pred.detach()
        pred_count = pred_detached.numel()
        diagnostics["scan_count"] += pred_count
        diagnostics["pred_sum"] += pred_detached.sum().item()
        diagnostics["pred_min"] = min(diagnostics["pred_min"], pred_detached.min().item())
        diagnostics["pred_max"] = max(diagnostics["pred_max"], pred_detached.max().item())
        for threshold in cfg.transfer_diagnostic_thresholds:
            diagnostics["below_threshold_counts"][str(threshold)] += int((pred_detached < threshold).sum().item())
        chosen = pred < 0.5
        if chosen.any().item():
            original_idx = pending.nonzero(as_tuple=False).squeeze(-1)
            chosen_idx = original_idx[chosen]
            selected_steps[chosen_idx] = k
            pending[chosen_idx] = False
            diagnostics["selected_step_counts"][int(k)] = int(chosen.sum().item())
    diagnostics["selected"] = int((selected_steps >= 0).sum().item())
    return selected_steps, diagnostics


def denoise_from_intermediate(
        diffusion: CDCDRMlpDiffusion,
        pair_embedding: torch.Tensor,
        steps: torch.Tensor,
) -> torch.Tensor:
    device = pair_embedding.device
    generated = torch.zeros_like(pair_embedding)
    noise = torch.randn_like(pair_embedding)
    total_steps = diffusion.scheduler.num_train_timesteps

    for step in torch.unique(steps[steps >= 0]).tolist():
        mask = steps == step
        t = torch.full((mask.sum().item(),), int(step), dtype=torch.long, device=device)
        x_t = diffusion.scheduler.add_noise(pair_embedding[mask], noise[mask], t)
        start_step = total_steps - 1 - int(step)
        denoised, _ = diffusion.denoise(
            x=x_t,
            y=None,
            null_y=None,
            guidance_scale=1.0,
            num_inference_steps=total_steps,
            start_step=start_step,
        )
        generated[mask] = denoised
    return generated


def generate_pure_fake_pairs(
        diffusion0: CDCDRMlpDiffusion,
        diffusion1: CDCDRMlpDiffusion,
        labels: torch.Tensor,
) -> torch.Tensor:
    device = labels.device
    output = torch.zeros(labels.size(0), *diffusion0.input_shape, device=device)
    for label_value, diffusion in ((0, diffusion0), (1, diffusion1)):
        mask = labels.long() == label_value
        if not mask.any().item():
            continue
        x_t = torch.randn(mask.sum().item(), *diffusion.input_shape, device=device)
        denoised, _ = diffusion.denoise(
            x=x_t,
            y=None,
            null_y=None,
            guidance_scale=1.0,
            num_inference_steps=diffusion.scheduler.num_train_timesteps,
        )
        output[mask] = denoised
    return output


def build_augmented_batch(
        frozen_backbone: DiffMSRIdModel,
        diffusion0: CDCDRMlpDiffusion,
        diffusion1: CDCDRMlpDiffusion,
        classifier: DomainClassifier,
        source_batch: Interaction,
        target_pool: Interaction,
        target_domain_idx: int,
        device: torch.device,
        cfg: DiffMSRExperimentSettings,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    source_as_target = interaction_with_domain(source_batch, target_domain_idx, frozen_backbone.DOMAIN).to(device)
    with torch.no_grad():
        target_domain_e = frozen_backbone.domain_emb(
            torch.full((len(source_as_target),), target_domain_idx, dtype=torch.long, device=device)
        )

    domain_parts = []
    head_domain_id_parts = []
    user_parts = []
    item_parts = []
    label_parts = []
    stats = {
        "source_batch_total": len(source_as_target),
        "source_candidate_total": 0,
        "source_candidate_label0": 0,
        "source_candidate_label1": 0,
        "transferred_total": 0,
        "transferred_label0": 0,
        "transferred_label1": 0,
        "target_true_total": 0,
        "target_fake_total": 0,
        "augment_total": 0,
        "transfer_diag_by_label": {
            0: empty_transfer_diagnostic(cfg),
            1: empty_transfer_diagnostic(cfg),
        },
    }

    for label_value, diffusion in ((0, diffusion0), (1, diffusion1)):
        label_mask = source_as_target[frozen_backbone.LABEL].long() == label_value
        if not label_mask.any().item():
            continue
        subset = source_as_target[label_mask]
        candidate_count = len(subset)
        stats["source_candidate_total"] += candidate_count
        stats[f"source_candidate_label{label_value}"] += candidate_count
        with torch.no_grad():
            pair_embedding = frozen_backbone.embed_user_item_pair(subset)
            selected_steps, transfer_diag = select_transfer_steps(pair_embedding, classifier, diffusion, cfg)
            merge_transfer_diagnostic(stats["transfer_diag_by_label"][label_value], transfer_diag)
            valid_mask = selected_steps >= 0
            if not valid_mask.any().item():
                continue
            transferred_count = int(valid_mask.sum().item())
            stats["transferred_total"] += transferred_count
            stats[f"transferred_label{label_value}"] += transferred_count
            transferred_pair = denoise_from_intermediate(
                diffusion, pair_embedding[valid_mask], selected_steps[valid_mask]
            )
            transferred_user_e, transferred_item_e = frozen_backbone.split_pair_embedding(transferred_pair)
            domain_parts.append(target_domain_e[label_mask][valid_mask])
            head_domain_id_parts.append(
                torch.full((transferred_count,), target_domain_idx, dtype=torch.long, device=device)
            )
            user_parts.append(transferred_user_e)
            item_parts.append(transferred_item_e)
            label_parts.append(subset[frozen_backbone.LABEL][valid_mask].float())

    target_true = target_pool.to(device)
    with torch.no_grad():
        true_domain_e, true_user_e, true_item_e = frozen_backbone.embed_triplet(
            target_true,
            domain_override=target_domain_idx,
        )
    stats["target_true_total"] = len(target_true)
    domain_parts.append(true_domain_e)
    head_domain_id_parts.append(torch.full((len(target_true),), target_domain_idx, dtype=torch.long, device=device))
    user_parts.append(true_user_e)
    item_parts.append(true_item_e)
    label_parts.append(target_true[frozen_backbone.LABEL].float())

    target_fake_anchor = sample_interaction(target_pool, cfg.aug_target_sample_size, device)
    fake_labels = target_fake_anchor[frozen_backbone.LABEL].float()
    with torch.no_grad():
        fake_domain_e = frozen_backbone.domain_emb(
            torch.full((len(target_fake_anchor),), target_domain_idx, dtype=torch.long, device=device)
        )
        fake_pairs = generate_pure_fake_pairs(diffusion0, diffusion1, fake_labels)
        fake_user_e, fake_item_e = frozen_backbone.split_pair_embedding(fake_pairs)
    stats["target_fake_total"] = len(target_fake_anchor)
    domain_parts.append(fake_domain_e)
    head_domain_id_parts.append(
        torch.full((len(target_fake_anchor),), target_domain_idx, dtype=torch.long, device=device)
    )
    user_parts.append(fake_user_e)
    item_parts.append(fake_item_e)
    label_parts.append(fake_labels)
    stats["augment_total"] = sum(part.size(0) for part in label_parts)

    return (
        torch.cat(domain_parts, dim=0),
        torch.cat(head_domain_id_parts, dim=0),
        torch.cat(user_parts, dim=0),
        torch.cat(item_parts, dim=0),
        torch.cat(label_parts, dim=0),
        stats,
    )


def train_true_epoch(
        model: DiffMSRIdModel,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model.calculate_loss(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    return total_loss / max(batch_count, 1)


def train_target_true_epoch(
        model: DiffMSRIdModel,
        target_loader: DataLoader,
        target_domain_idx: int,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch in target_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model.calculate_loss(batch, domain_override=target_domain_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    return total_loss / max(batch_count, 1)


def train_augmented_epoch(
        augment_model: DiffMSRIdModel,
        frozen_backbone: DiffMSRIdModel,
        diffusions_by_domain: dict,
        classifiers_by_domain: dict,
        source_loaders_by_domain: dict,
        target_pools_by_domain: dict,
        domain_specs: Tuple[Tuple[int, int], ...],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg: DiffMSRExperimentSettings,
) -> Tuple[float, dict]:
    augment_model.train()
    frozen_backbone.eval()
    total_loss = 0.0
    batch_count = 0
    epoch_stats = {
        "per_domain": {
            raw_domain: {
                "source_candidate_total": 0,
                "source_candidate_label0": 0,
                "source_candidate_label1": 0,
                "transferred_total": 0,
                "transferred_label0": 0,
                "transferred_label1": 0,
                "target_true_total": 0,
                "target_fake_total": 0,
                "augment_total": 0,
                "num_batches": 0,
                "transfer_diag_by_label": {
                    0: empty_transfer_diagnostic(cfg),
                    1: empty_transfer_diagnostic(cfg),
                },
            }
            for raw_domain, _ in domain_specs
        }
    }
    all_domain_parts = []
    all_head_domain_id_parts = []
    all_user_parts = []
    all_item_parts = []
    all_label_parts = []

    for raw_domain, target_domain_idx in domain_specs:
        diffusion0 = diffusions_by_domain[(target_domain_idx, 0)]
        diffusion1 = diffusions_by_domain[(target_domain_idx, 1)]
        classifier = classifiers_by_domain[target_domain_idx]
        source_loader = source_loaders_by_domain[target_domain_idx]
        target_pool = target_pools_by_domain[target_domain_idx]

        diffusion0.eval()
        diffusion1.eval()
        classifier.eval()

        for source_batch in source_loader:
            source_batch = source_batch.to(device)
            domain_e, head_domain_ids, user_e, item_e, labels, batch_stats = build_augmented_batch(
                frozen_backbone,
                diffusion0,
                diffusion1,
                classifier,
                source_batch,
                target_pool,
                target_domain_idx,
                device,
                cfg,
            )
            all_domain_parts.append(domain_e)
            all_head_domain_id_parts.append(head_domain_ids)
            all_user_parts.append(user_e)
            all_item_parts.append(item_e)
            all_label_parts.append(labels)

            domain_stats = epoch_stats["per_domain"][raw_domain]
            for key, value in batch_stats.items():
                if key == "transfer_diag_by_label":
                    for label_value, label_diag in value.items():
                        merge_transfer_diagnostic(domain_stats["transfer_diag_by_label"][label_value], label_diag)
                elif key in domain_stats:
                    domain_stats[key] += value
            domain_stats["num_batches"] += 1

    if not all_label_parts:
        epoch_stats["num_batches"] = 0
        return 0.0, epoch_stats

    all_domain_e = torch.cat(all_domain_parts, dim=0)
    all_head_domain_ids = torch.cat(all_head_domain_id_parts, dim=0)
    all_user_e = torch.cat(all_user_parts, dim=0)
    all_item_e = torch.cat(all_item_parts, dim=0)
    all_labels = torch.cat(all_label_parts, dim=0)

    permutation = torch.randperm(all_labels.size(0), device=device)
    all_domain_e = all_domain_e[permutation]
    all_head_domain_ids = all_head_domain_ids[permutation]
    all_user_e = all_user_e[permutation]
    all_item_e = all_item_e[permutation]
    all_labels = all_labels[permutation]

    for start in range(0, all_labels.size(0), cfg.resolved_source_aug_batch_size):
        end = min(start + cfg.resolved_source_aug_batch_size, all_labels.size(0))
        batch_domain_e = all_domain_e[start:end]
        batch_head_domain_ids = all_head_domain_ids[start:end]
        batch_user_e = all_user_e[start:end]
        batch_item_e = all_item_e[start:end]
        batch_labels = all_labels[start:end]

        optimizer.zero_grad()
        logits = augment_model.logits_from_embeddings(
            batch_domain_e,
            batch_user_e,
            batch_item_e,
            batch_head_domain_ids,
        )
        loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    epoch_stats["num_batches"] = batch_count
    return total_loss / max(batch_count, 1), epoch_stats


def train_cold_start_single_domain_baselines(
        schema_manager: SchemaManager,
        target_loaders_by_domain: dict,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        domain_specs: Tuple[Tuple[int, int], ...],
        device: torch.device,
        cfg: DiffMSRExperimentSettings,
) -> dict:
    best_by_domain = {}

    for raw_domain, mapped_domain in domain_specs:
        model = make_model(schema_manager, domain_specs, cfg).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.resolved_learning_rate_stage0,
            weight_decay=cfg.weight_decay,
        )
        best_state = deepcopy(model.state_dict())
        best_auc = -float("inf")
        best_epoch = None
        best_valid_metrics = None
        best_test_metrics = None
        target_loader = target_loaders_by_domain[mapped_domain]

        for epoch in range(cfg.resolved_stage0_epochs):
            train_loss = train_target_true_epoch(
                model,
                target_loader,
                mapped_domain,
                optimizer,
                device,
            )
            valid_metrics = evaluate_auc_logloss(
                model,
                valid_loader,
                mapped_domain,
                device,
                epoch,
                f"stage0-domain{raw_domain}-valid",
            )
            test_metrics = evaluate_auc_logloss(
                model,
                test_loader,
                mapped_domain,
                device,
                epoch,
                f"stage0-domain{raw_domain}-test",
            )
            valid_auc = valid_metrics.get("auc", -float("inf"))
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                best_valid_metrics = valid_metrics
                best_test_metrics = test_metrics
                print(
                    f"[Stage0][domain{raw_domain}][Epoch {epoch}] "
                    f"new best valid auc={best_auc:.6f}, best_test={best_test_metrics}"
                )
            print(
                f"[Stage0][domain{raw_domain}][Epoch {epoch}] "
                f"train_loss={train_loss:.6f}, "
                f"valid_auc={valid_metrics.get('auc', float('nan')):.6f}, "
                f"test_auc={test_metrics.get('auc', float('nan')):.6f}, "
                f"best_valid_auc={best_auc:.6f}"
            )

        if best_test_metrics is not None:
            print(f"[Stage0][domain{raw_domain}] best-epoch synced test metrics: {best_test_metrics}")
        best_by_domain[raw_domain] = {
            "auc": best_auc,
            "state": best_state,
            "epoch": best_epoch,
            "valid": best_valid_metrics,
            "test": best_test_metrics,
        }

    print(
        "[Stage0] cold-start per-domain best epochs:",
        {raw_domain: {"epoch": info["epoch"], "auc": info["auc"]} for raw_domain, info in best_by_domain.items()},
    )
    return best_by_domain


def format_multi_domain_aug_stats(epoch_stats: dict, cfg: DiffMSRExperimentSettings) -> str:
    def safe_ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    parts = []
    for raw_domain, stats in epoch_stats["per_domain"].items():
        parts.append(
            f"domain{raw_domain}: "
            f"source_candidates={stats['source_candidate_total']} "
            f"(label0={stats['source_candidate_label0']}, label1={stats['source_candidate_label1']}), "
            f"transferred={stats['transferred_total']} "
            f"(label0={stats['transferred_label0']}, label1={stats['transferred_label1']}), "
            f"transfer_rate={safe_ratio(stats['transferred_total'], stats['source_candidate_total']):.4%}, "
            f"target_true={stats['target_true_total']}, "
            f"target_fake={stats['target_fake_total']}, "
            f"augment_total={stats['augment_total']}, "
            f"transferred_share_in_aug={safe_ratio(stats['transferred_total'], stats['augment_total']):.4%}, "
            f"num_batches={stats['num_batches']}, "
            f"diag_label0=({format_transfer_diagnostic(stats['transfer_diag_by_label'][0], cfg)}), "
            f"diag_label1=({format_transfer_diagnostic(stats['transfer_diag_by_label'][1], cfg)})"
        )
    return " | ".join(parts)


def evaluate_with_domain_best_states(
        model: DiffMSRIdModel,
        data_loader: DataLoader,
        best_states_by_domain: dict,
        domain_specs: Tuple[Tuple[int, int], ...],
        device: torch.device,
        epoch: int,
        name: str,
) -> dict:
    overall_evaluator = Evaluator("AUC", "logloss")
    overall_count = 0
    overall_positive_count = 0.0
    metrics = {}

    for raw_domain, mapped_domain in domain_specs:
        best_info = best_states_by_domain.get(raw_domain)
        if best_info is None or best_info.get("state") is None:
            raise ValueError(f"domain{raw_domain} 缺少 best checkpoint，无法恢复评估。")
        model.load_state_dict(best_info["state"])
        domain_evaluator = Evaluator("AUC", "logloss")
        domain_count = 0
        domain_positive_count = 0.0
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                mask = batch[model.DOMAIN].long() == mapped_domain
                if not mask.any().item():
                    continue
                subset = batch[mask]
                scores = model.predict(subset, domain_override=mapped_domain)
                labels = subset[model.LABEL].float()
                domain_count += len(subset)
                domain_positive_count += labels.sum().item()
                overall_count += len(subset)
                overall_positive_count += labels.sum().item()
                domain_evaluator.collect_pointwise(
                    subset[model.USER].detach().cpu(),
                    labels.detach().cpu(),
                    scores.detach().cpu(),
                )
                overall_evaluator.collect_pointwise(
                    subset[model.USER].detach().cpu(),
                    labels.detach().cpu(),
                    scores.detach().cpu(),
                )
        metrics[f"domain{raw_domain}"] = add_count_fields(
            domain_evaluator.summary(),
            domain_count,
            domain_positive_count,
        )
        print(f"[Metrics][{name}-domain{raw_domain}][Epoch {epoch}] {metrics[f'domain{raw_domain}']}")

    metrics["overall"] = add_count_fields(overall_evaluator.summary(), overall_count, overall_positive_count)
    add_domain_aggregate_metrics(metrics, domain_specs)
    print(f"[Metrics][{name}-overall][Epoch {epoch}] {metrics['overall']}")
    print(f"[Metrics][{name}-domain_macro][Epoch {epoch}] {metrics['domain_macro']}")
    print(f"[Metrics][{name}-domain_weighted][Epoch {epoch}] {metrics['domain_weighted']}")
    print(f"[AUCSummary][{name}][Epoch {epoch}] {format_auc_summary(metrics, domain_specs)}")
    return metrics


def train_stage4_branches(
        backbone: DiffMSRIdModel,
        diffusions_by_domain: dict,
        classifiers_by_domain: dict,
        train_loader: DataLoader,
        source_loaders_by_domain: dict,
        target_pools_by_domain: dict,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        domain_specs: Tuple[Tuple[int, int], ...],
        device: torch.device,
        cfg: DiffMSRExperimentSettings,
) -> Tuple[DiffMSRIdModel, DiffMSRIdModel, dict, dict, dict]:
    baseline_model = deepcopy(backbone).to(device)
    augment_model = deepcopy(backbone).to(device)
    baseline_model.configure_stage4_trainability()
    augment_model.configure_stage4_trainability()

    baseline_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, baseline_model.parameters()),
        lr=cfg.learning_rate_stage4,
        weight_decay=0.0,
    )
    augment_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, augment_model.parameters()),
        lr=cfg.learning_rate_stage4,
        weight_decay=0.0,
    )

    best_baseline_state = deepcopy(baseline_model.state_dict())
    best_augment_state = deepcopy(augment_model.state_dict())
    best_baseline_auc = -float("inf")
    best_augment_auc = -float("inf")
    best_baseline_info = None
    best_augment_info = None
    best_baseline_weighted_auc = -float("inf")
    best_augment_weighted_auc = -float("inf")
    best_baseline_weighted_info = None
    best_augment_weighted_info = None
    baseline_best_by_domain = {
        raw_domain: {"auc": -float("inf"), "state": None, "epoch": None}
        for raw_domain, _ in domain_specs
    }
    augment_best_by_domain = {
        raw_domain: {"auc": -float("inf"), "state": None, "epoch": None}
        for raw_domain, _ in domain_specs
    }

    for epoch in range(cfg.stage4_epochs):
        baseline_loss = train_true_epoch(
            baseline_model,
            train_loader,
            baseline_optimizer,
            device,
        )
        augment_loss, augment_stats = train_augmented_epoch(
            augment_model,
            backbone,
            diffusions_by_domain,
            classifiers_by_domain,
            source_loaders_by_domain,
            target_pools_by_domain,
            domain_specs,
            augment_optimizer,
            device,
            cfg,
        )
        print(f"[Stage4][Epoch {epoch}] baseline_loss={baseline_loss:.6f}, augment_loss={augment_loss:.6f}")
        print(f"[Stage4][Epoch {epoch}][AugStats] {format_multi_domain_aug_stats(augment_stats, cfg)}")

        baseline_valid, baseline_test, augment_valid, augment_test = evaluate_branch_pair(
            baseline_model,
            augment_model,
            valid_loader,
            test_loader,
            domain_specs,
            device,
            epoch,
        )

        baseline_auc = baseline_valid["overall"].get("auc", -float("inf"))
        augment_auc = augment_valid["overall"].get("auc", -float("inf"))
        baseline_weighted_auc = baseline_valid["domain_weighted"].get("auc", -float("inf"))
        augment_weighted_auc = augment_valid["domain_weighted"].get("auc", -float("inf"))
        for raw_domain, _ in domain_specs:
            domain_key = f"domain{raw_domain}"
            baseline_domain_auc = baseline_valid[domain_key].get("auc", -float("inf"))
            if baseline_domain_auc > baseline_best_by_domain[raw_domain]["auc"]:
                baseline_best_by_domain[raw_domain] = {
                    "auc": baseline_domain_auc,
                    "state": deepcopy(baseline_model.state_dict()),
                    "epoch": epoch,
                    "valid": baseline_valid[domain_key],
                    "test": baseline_test[domain_key],
                }
                print(
                    f"[Stage4][Epoch {epoch}] new best baseline domain{raw_domain} valid auc="
                    f"{baseline_domain_auc:.6f}"
                )
            augment_domain_auc = augment_valid[domain_key].get("auc", -float("inf"))
            if augment_domain_auc > augment_best_by_domain[raw_domain]["auc"]:
                augment_best_by_domain[raw_domain] = {
                    "auc": augment_domain_auc,
                    "state": deepcopy(augment_model.state_dict()),
                    "epoch": epoch,
                    "valid": augment_valid[domain_key],
                    "test": augment_test[domain_key],
                }
                print(
                    f"[Stage4][Epoch {epoch}] new best augment domain{raw_domain} valid auc="
                    f"{augment_domain_auc:.6f}"
                )
        if baseline_weighted_auc > best_baseline_weighted_auc:
            best_baseline_weighted_auc = baseline_weighted_auc
            best_baseline_weighted_info = {
                "epoch": epoch,
                "auc": best_baseline_weighted_auc,
                "state": deepcopy(baseline_model.state_dict()),
                "valid": baseline_valid,
                "test": baseline_test,
                "valid_auc_summary": extract_auc_summary(baseline_valid),
                "test_auc_summary": extract_auc_summary(baseline_test),
                "loss": baseline_loss,
            }
            print(
                f"[Stage4][Epoch {epoch}] new best baseline weighted valid auc="
                f"{best_baseline_weighted_auc:.6f}, test={baseline_test}"
            )
        if augment_weighted_auc > best_augment_weighted_auc:
            best_augment_weighted_auc = augment_weighted_auc
            best_augment_weighted_info = {
                "epoch": epoch,
                "auc": best_augment_weighted_auc,
                "state": deepcopy(augment_model.state_dict()),
                "valid": augment_valid,
                "test": augment_test,
                "valid_auc_summary": extract_auc_summary(augment_valid),
                "test_auc_summary": extract_auc_summary(augment_test),
                "loss": augment_loss,
                "augment_stats": augment_stats,
            }
            print(
                f"[Stage4][Epoch {epoch}] new best augment weighted valid auc="
                f"{best_augment_weighted_auc:.6f}, test={augment_test}"
            )
        if baseline_auc > best_baseline_auc:
            best_baseline_auc = baseline_auc
            best_baseline_state = deepcopy(baseline_model.state_dict())
            best_baseline_info = {
                "epoch": epoch,
                "auc": best_baseline_auc,
                "state": deepcopy(baseline_model.state_dict()),
                "valid": baseline_valid,
                "test": baseline_test,
                "valid_auc_summary": extract_auc_summary(baseline_valid),
                "test_auc_summary": extract_auc_summary(baseline_test),
                "loss": baseline_loss,
            }
            print(f"[Stage4][Epoch {epoch}] new best baseline overall valid auc={best_baseline_auc:.6f}, test={baseline_test}")
        if augment_auc > best_augment_auc:
            best_augment_auc = augment_auc
            best_augment_state = deepcopy(augment_model.state_dict())
            best_augment_info = {
                "epoch": epoch,
                "auc": best_augment_auc,
                "state": deepcopy(augment_model.state_dict()),
                "valid": augment_valid,
                "test": augment_test,
                "valid_auc_summary": extract_auc_summary(augment_valid),
                "test_auc_summary": extract_auc_summary(augment_test),
                "loss": augment_loss,
                "augment_stats": augment_stats,
            }
            print(f"[Stage4][Epoch {epoch}] new best augment overall valid auc={best_augment_auc:.6f}, test={augment_test}")

    baseline_model.load_state_dict(best_baseline_state)
    augment_model.load_state_dict(best_augment_state)
    print(
        "[Stage4] baseline per-domain best epochs:",
        {raw_domain: {"epoch": info["epoch"], "auc": info["auc"]} for raw_domain, info in baseline_best_by_domain.items()},
    )
    print(
        "[Stage4] augment per-domain best epochs:",
        {raw_domain: {"epoch": info["epoch"], "auc": info["auc"]} for raw_domain, info in augment_best_by_domain.items()},
    )
    stage4_info = {
        "baseline_overall_best": best_baseline_info,
        "augment_overall_best": best_augment_info,
        "baseline_weighted_best": best_baseline_weighted_info,
        "augment_weighted_best": best_augment_weighted_info,
    }
    print(
        "[Stage4] weighted-best epochs:",
        {
            "baseline": {
                "epoch": None if best_baseline_weighted_info is None else best_baseline_weighted_info["epoch"],
                "auc": best_baseline_weighted_auc,
            },
            "augment": {
                "epoch": None if best_augment_weighted_info is None else best_augment_weighted_info["epoch"],
                "auc": best_augment_weighted_auc,
            },
        },
    )
    return baseline_model, augment_model, baseline_best_by_domain, augment_best_by_domain, stage4_info

def parse_and_format_experiment_results(json_path):
    import pandas as pd
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取各个阶段在 Test 集上的评估结果
    stage0 = data['stage0']['final_test']
    stage1 = data['stage1']['test']
    stage4_base = data['stage4_final']['baseline_test']
    stage4_aug = data['stage4_final']['augment_test']

    # 定义我们想要提取的指标维度 (展示名称, JSON中的key, 提取的指标)
    metric_targets = [
        ("Weighted AUC", "domain_weighted", "auc"),
        ("Macro AUC", "domain_macro", "auc"),
        ("Overall AUC", "overall", "auc"),
        ("Domain 0 AUC", "domain0", "auc"),
        ("Domain 1 AUC", "domain1", "auc"),
        ("Domain 2 AUC", "domain2", "auc"),
        ("Weighted Logloss", "domain_weighted", "logloss"),
    ]

    records = []

    for display_name, domain_key, metric_key in metric_targets:
        # 获取各个阶段的数值
        val_s0 = stage0[domain_key][metric_key]
        val_s1 = stage1[domain_key][metric_key]
        val_s4b = stage4_base[domain_key][metric_key]
        val_s4a = stage4_aug[domain_key][metric_key]

        # 核心对比逻辑：计算 Augment 相较于 最强基线 (Stage 1) 的提升
        if metric_key == "auc":
            imp_abs = val_s4a - val_s1
            imp_rel = (imp_abs / val_s1) * 100
            # 格式化提升，加号显示
            imp_abs_str = f"+{imp_abs:.4f}" if imp_abs > 0 else f"{imp_abs:.4f}"
            imp_rel_str = f"+{imp_rel:.2f}%" if imp_rel > 0 else f"{imp_rel:.2f}%"
        else:
            # 对于 Logloss，下降才是提升
            imp_abs = val_s4a - val_s1
            imp_rel = (imp_abs / val_s1) * 100
            imp_abs_str = f"{imp_abs:.4f}" if imp_abs < 0 else f"+{imp_abs:.4f}"
            imp_rel_str = f"{imp_rel:.2f}%" if imp_rel < 0 else f"+{imp_rel:.2f}%"

        records.append({
            "Metric": display_name,
            "Stage 0 (Single)": f"{val_s0:.4f}",
            "Stage 1 (Shared)": f"{val_s1:.4f}",
            "Stage 4 (Base)": f"{val_s4b:.4f}",
            "Stage 4 (Aug)": f"{val_s4a:.4f}",
            "Abs Improv. (vs S1)": imp_abs_str,
            "Rel Improv. (%)": imp_rel_str
        })

    # 转换为 DataFrame 并输出
    df = pd.DataFrame(records)

    print("=" * 85)
    print(" 🚀 Experiment Results Comparison (Test Set)")
    print("=" * 85)
    print(df.to_markdown(index=False, colalign=("left", "center", "center", "center", "center", "right", "right")))
    print("=" * 85)
    print("* Note: Absolute and Relative improvements are calculated comparing Stage 4 (Aug) against Stage 1 (Shared).")
    print("* Logloss improvement is indicated by a negative value (lower is better).\n")
