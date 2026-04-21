import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from betterbole.emb import SchemaManager
from betterbole.emb.emblayer import InterSideEmb, ItemSideEmb, UserSideEmb
from betterbole.emb.schema import SparseEmbSetting
from betterbole.core.enum_type import FeatureSource
from betterbole.evaluate.evaluator import Evaluator
from betterbole.core.interaction import Interaction
from betterbole.models.generative.diffusion.diffusions import CDCDRMlpDiffusion
from betterbole.models.generative.diffusion.schedulers import DDIMScheduler
from betterbole.utils import change_root_workdir
from split_tool import generate_hybrid_splits_polars

class DiffExperimentDataset:
    BASE_DIR = Path("raw/diffmsr")
    AMAZON_LARGE = BASE_DIR / "amazon_time.csv"
    AMAZON_SMALL = BASE_DIR / "amazonsmall.csv"
    DOUBAN = BASE_DIR / "douban3.csv"
change_root_workdir()

ALL_RAW_DOMAINS = (0, 1, 2)
EMB_DIM = 16
TRAIN_BATCH_SIZE = 2048
SOURCE_AUG_BATCH_SIZE = TRAIN_BATCH_SIZE * 10
EVAL_BATCH_SIZE = 4096
TIMESTEP = 500
DIFFUSION_BETA = 0.0002
DIFFUSION_SCHEDULE = "other"
DIFFUSION_OBJECTIVE = "pred_v"
WORK_DIR = "workspace/diffmsr-amazonS-multi-msr-workdir"
DATASET = DiffExperimentDataset.AMAZON_SMALL

SHARED_BOTTOM_HIDDEN_DIMS = (256, 256)
DOMAIN_HEAD_HIDDEN_DIMS = (256,)
HEAD_DROPOUT = 0.0
STAGE4_FREEZE_EMBEDDINGS = True
STAGE4_FREEZE_SHARED_BOTTOM = True

STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 40
STAGE3_EPOCHS = 15
STAGE4_EPOCHS = 10
STAGE0_EPOCHS = STAGE1_EPOCHS

LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-4
LEARNING_RATE_STAGE3 = 1e-3
LEARNING_RATE_STAGE4 = 2e-3
LEARNING_RATE_STAGE0 = LEARNING_RATE_STAGE1
WEIGHT_DECAY = 1e-7

CLASSIFIER_NOISE_MAX_STEP = 70
AUG_STEP1 = 30
AUG_STEP2 = 50
AUG_TARGET_SAMPLE_SIZE = 512
TRANSFER_DIAGNOSTIC_THRESHOLDS = (0.5, 0.6, 0.7)
TRANSFER_DIAGNOSTIC_TOP_STEPS = 5





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


def build_mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: Optional[int] = None) -> nn.Module:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(last_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(HEAD_DROPOUT),
        ])
        last_dim = hidden_dim
    if output_dim is None:
        return nn.Identity() if not layers else nn.Sequential(*layers)
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


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


def empty_transfer_diagnostic() -> dict:
    return {
        "candidates": 0,
        "selected": 0,
        "scan_count": 0,
        "pred_sum": 0.0,
        "pred_min": float("inf"),
        "pred_max": -float("inf"),
        "below_threshold_counts": {str(threshold): 0 for threshold in TRANSFER_DIAGNOSTIC_THRESHOLDS},
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


def format_transfer_diagnostic(diag: dict) -> str:
    scan_count = diag.get("scan_count", 0)
    candidates = diag.get("candidates", 0)
    selected = diag.get("selected", 0)
    pred_mean = diag["pred_sum"] / scan_count if scan_count else float("nan")
    pred_min = diag["pred_min"] if scan_count else float("nan")
    pred_max = diag["pred_max"] if scan_count else float("nan")
    below_parts = []
    for threshold in TRANSFER_DIAGNOSTIC_THRESHOLDS:
        key = str(threshold)
        below_rate = diag["below_threshold_counts"].get(key, 0) / scan_count if scan_count else 0.0
        below_parts.append(f"below{threshold:g}={below_rate:.2%}")
    top_steps = sorted(
        diag["selected_step_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:TRANSFER_DIAGNOSTIC_TOP_STEPS]
    top_step_text = "none" if not top_steps else ",".join(f"{step}:{count}" for step, count in top_steps)
    selected_rate = selected / candidates if candidates else 0.0
    return (
        f"cand={candidates}, selected={selected} ({selected_rate:.2%}), "
        f"scan_pred_mean={pred_mean:.4f}, scan_pred_range=[{pred_min:.4f},{pred_max:.4f}], "
        f"{' '.join(below_parts)}, top_steps={top_step_text}"
    )


class DiffMSRIdModel(nn.Module):
    def __init__(
            self,
            schema_manager: SchemaManager,
            domain_head_ids: Tuple[int, ...],
            embedding_size: int = EMB_DIM,
    ):
        super().__init__()
        self.manager = schema_manager
        self.USER = schema_manager.uid_field
        self.ITEM = schema_manager.iid_field
        self.DOMAIN = schema_manager.domain_field
        self.LABEL = schema_manager.label_field
        self.domain_head_ids = tuple(int(domain_id) for domain_id in domain_head_ids)
        self.domain_to_head_idx = {
            domain_id: head_idx for head_idx, domain_id in enumerate(self.domain_head_ids)
        }

        self.user_side_emb = UserSideEmb(schema_manager.settings)
        self.item_side_emb = ItemSideEmb(schema_manager.settings)
        self.inter_side_emb = InterSideEmb(schema_manager.settings)

        self.user_emb = self.user_side_emb.embedding.emb_modules[self.USER]
        self.item_emb = self.item_side_emb.embedding.emb_modules[self.ITEM]
        self.domain_emb = self.inter_side_emb.embedding.emb_modules[self.DOMAIN]

        input_dim = embedding_size * 3
        self.shared_bottom = build_mlp(input_dim, SHARED_BOTTOM_HIDDEN_DIMS)
        shared_output_dim = SHARED_BOTTOM_HIDDEN_DIMS[-1] if SHARED_BOTTOM_HIDDEN_DIMS else input_dim
        self.domain_heads = nn.ModuleList([
            build_mlp(shared_output_dim, DOMAIN_HEAD_HIDDEN_DIMS, output_dim=1)
            for _ in self.domain_head_ids
        ])

    def embed_user_item_pair(self, interaction: Interaction) -> torch.Tensor:
        user_e = self.user_emb(interaction[self.USER].long())
        item_e = self.item_emb(interaction[self.ITEM].long())
        return torch.stack([user_e, item_e], dim=1)

    def split_pair_embedding(self, pair_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return pair_embedding[:, 0, :], pair_embedding[:, 1, :]

    def embed_triplet(
            self,
            interaction: Interaction,
            domain_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        domain_ids = interaction[self.DOMAIN].long()
        if domain_override is not None:
            domain_ids = torch.full_like(domain_ids, domain_override)
        domain_e = self.domain_emb(domain_ids)
        user_e = self.user_emb(interaction[self.USER].long())
        item_e = self.item_emb(interaction[self.ITEM].long())
        return domain_e, user_e, item_e

    def map_domain_ids_to_head_indices(self, domain_ids: torch.Tensor) -> torch.Tensor:
        head_indices = torch.full_like(domain_ids, -1)
        for domain_id, head_idx in self.domain_to_head_idx.items():
            head_indices = torch.where(
                domain_ids == domain_id,
                torch.full_like(head_indices, head_idx),
                head_indices,
            )
        if (head_indices < 0).any().item():
            unknown = torch.unique(domain_ids[head_indices < 0]).tolist()
            raise ValueError(f"存在未配置 head 的 domain ids: {unknown}, configured={self.domain_head_ids}")
        return head_indices

    def shared_representation_from_embeddings(
            self,
            domain_e: torch.Tensor,
            user_e: torch.Tensor,
            item_e: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([domain_e, user_e, item_e], dim=-1)
        return self.shared_bottom(features)

    def logits_from_embeddings(
            self,
            domain_e: torch.Tensor,
            user_e: torch.Tensor,
            item_e: torch.Tensor,
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        shared_rep = self.shared_representation_from_embeddings(domain_e, user_e, item_e)
        all_logits = torch.cat([head(shared_rep) for head in self.domain_heads], dim=1)
        head_indices = self.map_domain_ids_to_head_indices(domain_ids.long())
        return all_logits.gather(1, head_indices.unsqueeze(-1)).squeeze(-1)

    def logits(self, interaction: Interaction, domain_override: Optional[int] = None) -> torch.Tensor:
        domain_ids = interaction[self.DOMAIN].long()
        if domain_override is not None:
            domain_ids = torch.full_like(domain_ids, domain_override)
        return self.logits_from_embeddings(
            *self.embed_triplet(interaction, domain_override=domain_override),
            domain_ids=domain_ids,
        )

    def calculate_loss(self, interaction: Interaction, domain_override: Optional[int] = None) -> torch.Tensor:
        logits = self.logits(interaction, domain_override=domain_override)
        return F.binary_cross_entropy_with_logits(logits, interaction[self.LABEL].float())

    @torch.no_grad()
    def predict(self, interaction: Interaction, domain_override: Optional[int] = None) -> torch.Tensor:
        return torch.sigmoid(self.logits(interaction, domain_override=domain_override))

    def configure_stage4_trainability(
            self,
            freeze_embeddings: bool = STAGE4_FREEZE_EMBEDDINGS,
            freeze_shared_bottom: bool = STAGE4_FREEZE_SHARED_BOTTOM,
    ) -> None:
        for param in self.parameters():
            param.requires_grad = True
        if freeze_embeddings:
            for module in (self.user_side_emb, self.item_side_emb, self.inter_side_emb):
                for param in module.parameters():
                    param.requires_grad = False
        if freeze_shared_bottom:
            for param in self.shared_bottom.parameters():
                param.requires_grad = False
        for head in self.domain_heads:
            for param in head.parameters():
                param.requires_grad = True


class DomainClassifier(nn.Module):
    def __init__(self, field_dim: int, num_fields: int = 2):
        super().__init__()
        self.input_dim = field_dim * num_fields
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, pair_embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(pair_embedding.view(pair_embedding.size(0), -1)).squeeze(-1))


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
        baseline_model,
        valid_loader,
        domain_specs,
        device,
        epoch,
        "baseline-valid",
    )
    baseline_test = evaluate_all_domain_metrics(
        baseline_model,
        test_loader,
        domain_specs,
        device,
        epoch,
        "baseline-test",
    )
    augment_valid = evaluate_all_domain_metrics(
        augment_model,
        valid_loader,
        domain_specs,
        device,
        epoch,
        "augment-valid",
    )
    augment_test = evaluate_all_domain_metrics(
        augment_model,
        test_loader,
        domain_specs,
        device,
        epoch,
        "augment-test",
    )
    return baseline_valid, baseline_test, augment_valid, augment_test


def train_backbone_stage(
        model: DiffMSRIdModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        domain_specs: Tuple[Tuple[int, int], ...],
        device: torch.device,
) -> Tuple[DiffMSRIdModel, dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_STAGE1, weight_decay=WEIGHT_DECAY)
    best_state = deepcopy(model.state_dict())
    best_auc = -float("inf")
    best_epoch = None
    best_valid_metrics = None
    best_test_metrics = None
    for epoch in range(STAGE1_EPOCHS):
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
) -> CDCDRMlpDiffusion:
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=LEARNING_RATE_STAGE2, weight_decay=WEIGHT_DECAY)
    backbone.eval()
    best_state = deepcopy(diffusion.state_dict())
    best_loss = float("inf")

    for epoch in range(STAGE2_EPOCHS):
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
) -> DomainClassifier:
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE_STAGE3, weight_decay=WEIGHT_DECAY)
    backbone.eval()
    scheduler = scheduler_owner.scheduler
    max_t = min(CLASSIFIER_NOISE_MAX_STEP, scheduler.num_train_timesteps)
    best_state = deepcopy(classifier.state_dict())
    best_loss = float("inf")

    for epoch in range(STAGE3_EPOCHS):
        classifier.train()
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            subset = batch
            with torch.no_grad():
                pair_embedding = backbone.embed_user_item_pair(subset)
                t = torch.randint(0, max_t, (len(subset),), device=device).long()
                noisy_pair = scheduler.add_noise(pair_embedding, torch.randn_like(pair_embedding), t)
            domain_labels = (subset[backbone.DOMAIN].long() != target_domain_idx).float()
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
) -> Tuple[torch.Tensor, dict]:
    device = pair_embedding.device
    selected_steps = torch.full((pair_embedding.size(0),), -1, dtype=torch.long, device=device)
    pending = torch.ones(pair_embedding.size(0), dtype=torch.bool, device=device)
    noise = torch.randn_like(pair_embedding)
    diagnostics = empty_transfer_diagnostic()
    diagnostics["candidates"] = pair_embedding.size(0)

    for k in range(AUG_STEP1, AUG_STEP2):
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
        for threshold in TRANSFER_DIAGNOSTIC_THRESHOLDS:
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    source_as_target = interaction_with_domain(source_batch, target_domain_idx, frozen_backbone.DOMAIN).to(device)
    with torch.no_grad():
        target_domain_e = frozen_backbone.domain_emb(
            torch.full(
                (len(source_as_target),),
                target_domain_idx,
                dtype=torch.long,
                device=device,
            )
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
            0: empty_transfer_diagnostic(),
            1: empty_transfer_diagnostic(),
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
            selected_steps, transfer_diag = select_transfer_steps(pair_embedding, classifier, diffusion)
            merge_transfer_diagnostic(stats["transfer_diag_by_label"][label_value], transfer_diag)
            valid_mask = selected_steps >= 0
            if not valid_mask.any().item():
                continue
            transferred_count = int(valid_mask.sum().item())
            stats["transferred_total"] += transferred_count
            stats[f"transferred_label{label_value}"] += transferred_count
            transferred_pair = denoise_from_intermediate(
                diffusion,
                pair_embedding[valid_mask],
                selected_steps[valid_mask],
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
    head_domain_id_parts.append(
        torch.full((len(target_true),), target_domain_idx, dtype=torch.long, device=device)
    )
    user_parts.append(true_user_e)
    item_parts.append(true_item_e)
    label_parts.append(target_true[frozen_backbone.LABEL].float())

    target_fake_anchor = sample_interaction(target_pool, AUG_TARGET_SAMPLE_SIZE, device)
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
                    0: empty_transfer_diagnostic(),
                    1: empty_transfer_diagnostic(),
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
                        merge_transfer_diagnostic(
                            domain_stats["transfer_diag_by_label"][label_value],
                            label_diag,
                        )
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

    for start in range(0, all_labels.size(0), SOURCE_AUG_BATCH_SIZE):
        end = min(start + SOURCE_AUG_BATCH_SIZE, all_labels.size(0))
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
) -> dict:
    domain_head_ids = tuple(mapped_domain for _, mapped_domain in domain_specs)
    best_by_domain = {}

    for raw_domain, mapped_domain in domain_specs:
        model = DiffMSRIdModel(
            schema_manager,
            domain_head_ids=domain_head_ids,
            embedding_size=EMB_DIM,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE_STAGE0,
            weight_decay=WEIGHT_DECAY,
        )
        best_state = deepcopy(model.state_dict())
        best_auc = -float("inf")
        best_epoch = None
        best_valid_metrics = None
        best_test_metrics = None
        target_loader = target_loaders_by_domain[mapped_domain]

        for epoch in range(STAGE0_EPOCHS):
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


def format_multi_domain_aug_stats(epoch_stats: dict) -> str:
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
            f"diag_label0=({format_transfer_diagnostic(stats['transfer_diag_by_label'][0])}), "
            f"diag_label1=({format_transfer_diagnostic(stats['transfer_diag_by_label'][1])})"
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
) -> Tuple[DiffMSRIdModel, DiffMSRIdModel, dict, dict, dict]:
    baseline_model = deepcopy(backbone).to(device)
    augment_model = deepcopy(backbone).to(device)
    baseline_model.configure_stage4_trainability()
    augment_model.configure_stage4_trainability()

    baseline_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, baseline_model.parameters()),
        lr=LEARNING_RATE_STAGE4,
        weight_decay=0.0,
    )
    augment_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, augment_model.parameters()),
        lr=LEARNING_RATE_STAGE4,
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

    for epoch in range(STAGE4_EPOCHS):
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
        )
        print(f"[Stage4][Epoch {epoch}] baseline_loss={baseline_loss:.6f}, augment_loss={augment_loss:.6f}")
        print(f"[Stage4][Epoch {epoch}][AugStats] {format_multi_domain_aug_stats(augment_stats)}")

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


if __name__ == "__main__":
    from betterbole.utils.task_chain import auto_queue

    auto_queue()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = DATASET

    settings_list = [
        SparseEmbSetting("user", FeatureSource.USER_ID, EMB_DIM, min_freq=1, use_oov=True),
        SparseEmbSetting("item", FeatureSource.ITEM_ID, EMB_DIM, min_freq=1, use_oov=True),
        SparseEmbSetting("domain_indicator", FeatureSource.INTERACTION, EMB_DIM, padding_zero=True, use_oov=False),
    ]
    manager = SchemaManager(
        settings_list,
        WORK_DIR,
        label_fields="label",
        domain_fields="domain_indicator",
        time_field="time",
    )

    whole_lf = pl.scan_csv(dataset_path)
    train_samples_lf, train_ple_lf, val_ple_lf, test_ple_lf = generate_hybrid_splits_polars(whole_lf)
    print_split_statistics(train_samples_lf, train_ple_lf, val_ple_lf, test_ple_lf)
    domain_count_summary = print_domain_count_statistics(
        train_samples_lf,
        train_ple_lf,
        val_ple_lf,
        test_ple_lf,
        manager.domain_field,
    )

    manager.fit(train_ple_lf)
    domain_specs = tuple((raw_domain, mapped_domain_id(manager, raw_domain)) for raw_domain in ALL_RAW_DOMAINS)
    print(f"Mapped domains: {domain_specs}")

    train_lf = manager.transform(train_ple_lf).sort(by="time").collect()
    valid_lf = manager.transform(val_ple_lf).sort(by="time").collect()
    test_lf = manager.transform(test_ple_lf).sort(by="time").collect()

    train_interaction = lf_to_interaction(train_lf)
    valid_interaction = lf_to_interaction(valid_lf)
    test_interaction = lf_to_interaction(test_lf)

    train_loader = make_loader(train_interaction, TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = make_loader(valid_interaction, EVAL_BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_interaction, EVAL_BATCH_SIZE, shuffle=False)

    domain_interactions = {}
    domain_label_loaders = {}
    source_loaders_by_domain = {}
    target_pools_by_domain = {}
    target_loaders_by_domain = {}
    diffusions_by_domain = {}
    classifiers_by_domain = {}

    for raw_domain, mapped_domain in domain_specs:
        domain_lf = train_lf.filter(pl.col(manager.domain_field) == mapped_domain)
        domain_interaction = lf_to_interaction(domain_lf)
        domain_interactions[mapped_domain] = domain_interaction
        target_pools_by_domain[mapped_domain] = domain_interaction
        target_loaders_by_domain[mapped_domain] = make_loader(
            domain_interaction,
            TRAIN_BATCH_SIZE,
            shuffle=True,
        )
        domain_label_loaders[(mapped_domain, 0)] = make_loader(
            lf_to_interaction(domain_lf.filter(pl.col(manager.label_field) == 0)),
            TRAIN_BATCH_SIZE,
            shuffle=True,
        )
        domain_label_loaders[(mapped_domain, 1)] = make_loader(
            lf_to_interaction(domain_lf.filter(pl.col(manager.label_field) == 1)),
            TRAIN_BATCH_SIZE,
            shuffle=True,
        )
        source_lf = train_lf.filter(pl.col(manager.domain_field) != mapped_domain)
        source_loaders_by_domain[mapped_domain] = make_loader(
            lf_to_interaction(source_lf),
            SOURCE_AUG_BATCH_SIZE,
            shuffle=True,
        )

    checkpoint_dir = manager.work_dir / "diffmsr_id_stage_ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Stage 0: Cold-start single-domain baseline =====")
    stage0_best_by_domain = train_cold_start_single_domain_baselines(
        manager,
        target_loaders_by_domain,
        valid_loader,
        test_loader,
        domain_specs,
        device,
    )
    for raw_domain, _ in domain_specs:
        torch.save(stage0_best_by_domain[raw_domain]["state"], checkpoint_dir / f"stage0_domain{raw_domain}_single_domain.pt")

    backbone = DiffMSRIdModel(
        manager,
        domain_head_ids=tuple(mapped_domain for _, mapped_domain in domain_specs),
        embedding_size=EMB_DIM,
    ).to(device)

    print("\n===== Stage 1: Backbone pretrain =====")
    backbone, stage1_best_info = train_backbone_stage(backbone, train_loader, valid_loader, test_loader, domain_specs, device)
    torch.save(backbone.state_dict(), checkpoint_dir / "stage1_backbone.pt")

    print("\n===== Stage 2: Per-domain diffusion label=0/1 =====")
    for raw_domain, mapped_domain in domain_specs:
        for label_value in (0, 1):
            diffusion = CDCDRMlpDiffusion(
                DDIMScheduler(
                    num_train_timesteps=TIMESTEP,
                    schedule_type=DIFFUSION_SCHEDULE,
                    beta_start=DIFFUSION_BETA,
                ),
                EMB_DIM,
                uncon_p=0.0,
                num_fields=2,
                objective=DIFFUSION_OBJECTIVE,
            ).to(device)
            diffusion = train_single_diffusion(
                backbone,
                diffusion,
                domain_label_loaders[(mapped_domain, label_value)],
                device,
                f"domain{raw_domain}-label{label_value}",
            )
            diffusions_by_domain[(mapped_domain, label_value)] = diffusion
            torch.save(diffusion.state_dict(), checkpoint_dir / f"stage2_domain{raw_domain}_label{label_value}.pt")

    print("\n===== Stage 3: Per-domain domain classifier =====")
    for raw_domain, mapped_domain in domain_specs:
        classifier = DomainClassifier(EMB_DIM, num_fields=2).to(device)
        classifier = train_classifier_stage(
            backbone,
            diffusions_by_domain[(mapped_domain, 0)],
            classifier,
            train_loader,
            mapped_domain,
            device,
        )
        classifiers_by_domain[mapped_domain] = classifier
        torch.save(classifier.state_dict(), checkpoint_dir / f"stage3_domain{raw_domain}_classifier.pt")

    print("\n===== Stage 4: Branch baseline vs all-domain diffusion-augment =====")
    baseline_model, augment_model, baseline_best_by_domain, augment_best_by_domain, stage4_info = train_stage4_branches(
        backbone,
        diffusions_by_domain,
        classifiers_by_domain,
        train_loader,
        source_loaders_by_domain,
        target_pools_by_domain,
        valid_loader,
        test_loader,
        domain_specs,
        device,
    )
    torch.save(baseline_model.state_dict(), checkpoint_dir / "stage4_baseline_model.pt")
    torch.save(augment_model.state_dict(), checkpoint_dir / "stage4_augment_model.pt")
    if stage4_info.get("baseline_weighted_best") is not None:
        torch.save(
            stage4_info["baseline_weighted_best"]["state"],
            checkpoint_dir / "stage4_baseline_weighted_best.pt",
        )
    if stage4_info.get("augment_weighted_best") is not None:
        torch.save(
            stage4_info["augment_weighted_best"]["state"],
            checkpoint_dir / "stage4_augment_weighted_best.pt",
        )
    if stage4_info.get("baseline_overall_best") is not None:
        torch.save(
            stage4_info["baseline_overall_best"]["state"],
            checkpoint_dir / "stage4_baseline_overall_best.pt",
        )
    if stage4_info.get("augment_overall_best") is not None:
        torch.save(
            stage4_info["augment_overall_best"]["state"],
            checkpoint_dir / "stage4_augment_overall_best.pt",
        )
    for raw_domain, _ in domain_specs:
        torch.save(
            baseline_best_by_domain[raw_domain]["state"],
            checkpoint_dir / f"stage4_baseline_domain{raw_domain}_best.pt",
        )
        torch.save(
            augment_best_by_domain[raw_domain]["state"],
            checkpoint_dir / f"stage4_augment_domain{raw_domain}_best.pt",
        )

    print("\n===== Final metrics on all domains (per-domain best checkpoints) =====")
    stage0_eval_model = DiffMSRIdModel(
        manager,
        domain_head_ids=tuple(mapped_domain for _, mapped_domain in domain_specs),
        embedding_size=EMB_DIM,
    ).to(device)
    final_stage0_valid = evaluate_with_domain_best_states(
        stage0_eval_model,
        valid_loader,
        stage0_best_by_domain,
        domain_specs,
        device,
        STAGE0_EPOCHS,
        "final-stage0-valid",
    )
    final_stage0_test = evaluate_with_domain_best_states(
        stage0_eval_model,
        test_loader,
        stage0_best_by_domain,
        domain_specs,
        device,
        STAGE0_EPOCHS,
        "final-stage0-test",
    )
    print(f"[Stage0][Final][valid] {extract_auc_summary(final_stage0_valid)}")
    print(f"[Stage0][Final][test] {extract_auc_summary(final_stage0_test)}")
    final_baseline_valid = evaluate_with_domain_best_states(
        baseline_model,
        valid_loader,
        baseline_best_by_domain,
        domain_specs,
        device,
        STAGE4_EPOCHS,
        "final-baseline-valid",
    )
    final_baseline_test = evaluate_with_domain_best_states(
        baseline_model,
        test_loader,
        baseline_best_by_domain,
        domain_specs,
        device,
        STAGE4_EPOCHS,
        "final-baseline-test",
    )
    final_augment_valid = evaluate_with_domain_best_states(
        augment_model,
        valid_loader,
        augment_best_by_domain,
        domain_specs,
        device,
        STAGE4_EPOCHS,
        "final-augment-valid",
    )
    final_augment_test = evaluate_with_domain_best_states(
        augment_model,
        test_loader,
        augment_best_by_domain,
        domain_specs,
        device,
        STAGE4_EPOCHS,
        "final-augment-test",
    )
    print(f"[Stage4][Final][baseline-valid] {extract_auc_summary(final_baseline_valid)}")
    print(f"[Stage4][Final][baseline-test] {extract_auc_summary(final_baseline_test)}")
    print(f"[Stage4][Final][augment-valid] {extract_auc_summary(final_augment_valid)}")
    print(f"[Stage4][Final][augment-test] {extract_auc_summary(final_augment_test)}")
    final_comparison = {
        "stage0_single_domain_valid": final_stage0_valid,
        "stage0_single_domain_test": final_stage0_test,
        "baseline_valid": final_baseline_valid,
        "baseline_test": final_baseline_test,
        "augment_valid": final_augment_valid,
        "augment_test": final_augment_test,
    }
    experiment_record = {
        "dataset": str(dataset_path),
        "work_dir": str(manager.work_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "domain_specs": [
            {"raw_domain": raw_domain, "mapped_domain": mapped_domain}
            for raw_domain, mapped_domain in domain_specs
        ],
        "domain_count_summary": domain_count_summary,
        "hyperparameters": {
            "emb_dim": EMB_DIM,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "source_aug_batch_size": SOURCE_AUG_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "stage0_epochs": STAGE0_EPOCHS,
            "stage1_epochs": STAGE1_EPOCHS,
            "stage2_epochs": STAGE2_EPOCHS,
            "stage3_epochs": STAGE3_EPOCHS,
            "stage4_epochs": STAGE4_EPOCHS,
            "learning_rate_stage0": LEARNING_RATE_STAGE0,
            "learning_rate_stage1": LEARNING_RATE_STAGE1,
            "learning_rate_stage2": LEARNING_RATE_STAGE2,
            "learning_rate_stage3": LEARNING_RATE_STAGE3,
            "learning_rate_stage4": LEARNING_RATE_STAGE4,
            "weight_decay": WEIGHT_DECAY,
            "stage4_freeze_embeddings": STAGE4_FREEZE_EMBEDDINGS,
            "stage4_freeze_shared_bottom": STAGE4_FREEZE_SHARED_BOTTOM,
        },
        "checkpoints": {
            "stage0_single_domain": {
                f"domain{raw_domain}": str(checkpoint_dir / f"stage0_domain{raw_domain}_single_domain.pt")
                for raw_domain, _ in domain_specs
            },
            "stage1_backbone": str(checkpoint_dir / "stage1_backbone.pt"),
            "stage4_baseline_model": str(checkpoint_dir / "stage4_baseline_model.pt"),
            "stage4_augment_model": str(checkpoint_dir / "stage4_augment_model.pt"),
            "stage4_baseline_weighted_best": str(checkpoint_dir / "stage4_baseline_weighted_best.pt"),
            "stage4_augment_weighted_best": str(checkpoint_dir / "stage4_augment_weighted_best.pt"),
            "stage4_baseline_overall_best": str(checkpoint_dir / "stage4_baseline_overall_best.pt"),
            "stage4_augment_overall_best": str(checkpoint_dir / "stage4_augment_overall_best.pt"),
            "stage4_baseline_domain_best": {
                f"domain{raw_domain}": str(checkpoint_dir / f"stage4_baseline_domain{raw_domain}_best.pt")
                for raw_domain, _ in domain_specs
            },
            "stage4_augment_domain_best": {
                f"domain{raw_domain}": str(checkpoint_dir / f"stage4_augment_domain{raw_domain}_best.pt")
                for raw_domain, _ in domain_specs
            },
        },
        "stage0": {
            "best_by_domain": summarize_best_by_domain(stage0_best_by_domain),
            "final_valid": final_stage0_valid,
            "final_test": final_stage0_test,
            "final_valid_auc_summary": extract_auc_summary(final_stage0_valid),
            "final_test_auc_summary": extract_auc_summary(final_stage0_test),
        },
        "stage1": strip_model_states(stage1_best_info),
        "stage4_training": {
            "overall_best": {
                "baseline": strip_model_states(stage4_info.get("baseline_overall_best")),
                "augment": strip_model_states(stage4_info.get("augment_overall_best")),
            },
            "weighted_best": {
                "baseline": strip_model_states(stage4_info.get("baseline_weighted_best")),
                "augment": strip_model_states(stage4_info.get("augment_weighted_best")),
            },
        },
        "stage4_final": {
            "baseline_best_by_domain": summarize_best_by_domain(baseline_best_by_domain),
            "augment_best_by_domain": summarize_best_by_domain(augment_best_by_domain),
            "baseline_valid": final_baseline_valid,
            "baseline_test": final_baseline_test,
            "augment_valid": final_augment_valid,
            "augment_test": final_augment_test,
            "baseline_valid_auc_summary": extract_auc_summary(final_baseline_valid),
            "baseline_test_auc_summary": extract_auc_summary(final_baseline_test),
            "augment_valid_auc_summary": extract_auc_summary(final_augment_valid),
            "augment_test_auc_summary": extract_auc_summary(final_augment_test),
        },
        "final_comparison": final_comparison,
    }
    record_path = manager.work_dir / "diffmsr_amazonS_multi_msr_experiment_record.json"
    save_experiment_record(strip_model_states(experiment_record), record_path)
    print(f"[ExperimentRecord] saved to {record_path}")
    print(
        "Final comparison:",
        final_comparison,
    )
