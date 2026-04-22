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
MERGED_SOURCE_DOMAIN_RAW = 0
MERGED_TARGET_DOMAIN_RAW = 1
TARGET_DOMAIN_RAW = 0
EMB_DIM = 16
TRAIN_BATCH_SIZE = 2048
SOURCE_AUG_BATCH_SIZE = TRAIN_BATCH_SIZE * 10
EVAL_BATCH_SIZE = 4096
TIMESTEP = 500
DIFFUSION_BETA = 0.0002
DIFFUSION_SCHEDULE = "other"
DIFFUSION_OBJECTIVE = "pred_v"
WORK_DIR_PREFIX = "diffmsr-amazonS-from02to1-workdir"
DATASET = DiffExperimentDataset.AMAZON_SMALL

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 40
STAGE3_EPOCHS = 15
STAGE4_EPOCHS = 20

LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-4
LEARNING_RATE_STAGE3 = 1e-3
LEARNING_RATE_STAGE4 = 2e-3
WEIGHT_DECAY = 1e-7

CLASSIFIER_NOISE_MAX_STEP = 70
AUG_STEP1 = 30
AUG_STEP2 = 50
AUG_TARGET_SAMPLE_SIZE = 512





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


def collapse_domains_after_split(
        frame: pl.LazyFrame,
        target_domain_raw: int,
        domain_field: str = "domain_indicator",
) -> pl.LazyFrame:
    return frame.with_columns(
        pl.when(pl.col(domain_field) == target_domain_raw)
        .then(pl.lit(MERGED_TARGET_DOMAIN_RAW))
        .otherwise(pl.lit(MERGED_SOURCE_DOMAIN_RAW))
        .alias(domain_field)
    )


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


class DiffMSRIdModel(nn.Module):
    def __init__(self, schema_manager: SchemaManager, embedding_size: int = EMB_DIM):
        super().__init__()
        self.manager = schema_manager
        self.USER = schema_manager.uid_field
        self.ITEM = schema_manager.iid_field
        self.DOMAIN = schema_manager.domain_field
        self.LABEL = schema_manager.label_field

        self.user_side_emb = UserSideEmb(schema_manager.settings)
        self.item_side_emb = ItemSideEmb(schema_manager.settings)
        self.inter_side_emb = InterSideEmb(schema_manager.settings)

        self.user_emb = self.user_side_emb.embedding.emb_modules[self.USER]
        self.item_emb = self.item_side_emb.embedding.emb_modules[self.ITEM]
        self.domain_emb = self.inter_side_emb.embedding.emb_modules[self.DOMAIN]

        input_dim = embedding_size * 3
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(256, 1),
        )

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

    def logits_from_embeddings(
            self,
            domain_e: torch.Tensor,
            user_e: torch.Tensor,
            item_e: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(torch.cat([domain_e, user_e, item_e], dim=-1)).squeeze(-1)

    def logits(self, interaction: Interaction, domain_override: Optional[int] = None) -> torch.Tensor:
        return self.logits_from_embeddings(*self.embed_triplet(interaction, domain_override=domain_override))

    def calculate_loss(self, interaction: Interaction, domain_override: Optional[int] = None) -> torch.Tensor:
        logits = self.logits(interaction, domain_override=domain_override)
        return F.binary_cross_entropy_with_logits(logits, interaction[self.LABEL].float())

    @torch.no_grad()
    def predict(self, interaction: Interaction, domain_override: Optional[int] = None) -> torch.Tensor:
        return torch.sigmoid(self.logits(interaction, domain_override=domain_override))

    def freeze_embeddings_train_head(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
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
        domain_idx: int,
        device: torch.device,
        epoch: int,
        name: str,
) -> dict:
    evaluator = Evaluator("AUC", "logloss")
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            mask = batch[model.DOMAIN].long() == domain_idx
            if not mask.any().item():
                continue
            subset = batch[mask]
            scores = model.predict(subset, domain_override=domain_idx)
            evaluator.collect_pointwise(
                subset[model.USER].detach().cpu(),
                subset[model.LABEL].float().detach().cpu(),
                scores.detach().cpu(),
            )
    metrics = evaluator.summary()
    print(f"[Metrics][{name}][Epoch {epoch}] {metrics}")
    return metrics


def evaluate_branch_pair(
        baseline_model: DiffMSRIdModel,
        augment_model: DiffMSRIdModel,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        target_domain_idx: int,
        device: torch.device,
        epoch: int,
) -> Tuple[dict, dict, dict, dict]:
    baseline_valid = evaluate_auc_logloss(
        baseline_model,
        valid_loader,
        target_domain_idx,
        device,
        epoch,
        "baseline-valid",
    )
    baseline_test = evaluate_auc_logloss(
        baseline_model,
        test_loader,
        target_domain_idx,
        device,
        epoch,
        "baseline-test",
    )
    augment_valid = evaluate_auc_logloss(
        augment_model,
        valid_loader,
        target_domain_idx,
        device,
        epoch,
        "augment-valid",
    )
    augment_test = evaluate_auc_logloss(
        augment_model,
        test_loader,
        target_domain_idx,
        device,
        epoch,
        "augment-test",
    )
    return baseline_valid, baseline_test, augment_valid, augment_test


def train_backbone_stage(
        model: DiffMSRIdModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        target_domain_idx: int,
        device: torch.device,
) -> DiffMSRIdModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_STAGE1, weight_decay=WEIGHT_DECAY)
    best_state = deepcopy(model.state_dict())
    best_auc = -float("inf")
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
        metrics = evaluate_auc_logloss(model, valid_loader, target_domain_idx, device, epoch, "stage1-target-valid")
        auc = metrics.get("auc", -float("inf"))
        if auc > best_auc:
            best_auc = auc
            best_state = deepcopy(model.state_dict())
        print(f"[Stage1][Epoch {epoch}] backbone loss={avg_loss:.6f}, best_target_auc={best_auc:.6f}")
    model.load_state_dict(best_state)
    return model


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
        diffusion0: CDCDRMlpDiffusion,
        classifier: DomainClassifier,
        train_loader: DataLoader,
        source_domain_idx: int,
        target_domain_idx: int,
        device: torch.device,
) -> DomainClassifier:
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE_STAGE3, weight_decay=WEIGHT_DECAY)
    backbone.eval()
    scheduler = diffusion0.scheduler
    max_t = min(CLASSIFIER_NOISE_MAX_STEP, scheduler.num_train_timesteps)
    best_state = deepcopy(classifier.state_dict())
    best_loss = float("inf")

    for epoch in range(STAGE3_EPOCHS):
        classifier.train()
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            domains = batch[backbone.DOMAIN].long()
            mask = (domains == source_domain_idx) | (domains == target_domain_idx)
            if not mask.any().item():
                continue
            subset = batch[mask]
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
        print(f"[Stage3][Epoch {epoch}] classifier loss={avg_loss:.6f}, best={best_loss:.6f}")

    classifier.load_state_dict(best_state)
    return classifier


def select_transfer_steps(
        pair_embedding: torch.Tensor,
        classifier: DomainClassifier,
        diffusion: CDCDRMlpDiffusion,
) -> torch.Tensor:
    device = pair_embedding.device
    selected_steps = torch.full((pair_embedding.size(0),), -1, dtype=torch.long, device=device)
    pending = torch.ones(pair_embedding.size(0), dtype=torch.bool, device=device)
    noise = torch.randn_like(pair_embedding)

    for k in range(AUG_STEP1, AUG_STEP2):
        if not pending.any().item():
            break
        t = torch.full((pending.sum().item(),), k, dtype=torch.long, device=device)
        noisy_pair = diffusion.scheduler.add_noise(pair_embedding[pending], noise[pending], t)
        pred = classifier(noisy_pair)
        chosen = pred < 0.5
        if chosen.any().item():
            original_idx = pending.nonzero(as_tuple=False).squeeze(-1)
            chosen_idx = original_idx[chosen]
            selected_steps[chosen_idx] = k
            pending[chosen_idx] = False
    return selected_steps


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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
            selected_steps = select_transfer_steps(pair_embedding, classifier, diffusion)
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
    user_parts.append(fake_user_e)
    item_parts.append(fake_item_e)
    label_parts.append(fake_labels)
    stats["augment_total"] = sum(part.size(0) for part in label_parts)

    return (
        torch.cat(domain_parts, dim=0),
        torch.cat(user_parts, dim=0),
        torch.cat(item_parts, dim=0),
        torch.cat(label_parts, dim=0),
        stats,
    )


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
        diffusion0: CDCDRMlpDiffusion,
        diffusion1: CDCDRMlpDiffusion,
        classifier: DomainClassifier,
        source_loader: DataLoader,
        target_pool: Interaction,
        target_domain_idx: int,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
) -> Tuple[float, dict]:
    augment_model.train()
    frozen_backbone.eval()
    diffusion0.eval()
    diffusion1.eval()
    classifier.eval()
    total_loss = 0.0
    batch_count = 0
    epoch_stats = {
        "source_batch_total": 0,
        "source_candidate_total": 0,
        "source_candidate_label0": 0,
        "source_candidate_label1": 0,
        "transferred_total": 0,
        "transferred_label0": 0,
        "transferred_label1": 0,
        "target_true_total": 0,
        "target_fake_total": 0,
        "augment_total": 0,
    }
    for source_batch in source_loader:
        source_batch = source_batch.to(device)
        domain_e, user_e, item_e, labels, batch_stats = build_augmented_batch(
            frozen_backbone,
            diffusion0,
            diffusion1,
            classifier,
            source_batch,
            target_pool,
            target_domain_idx,
            device,
        )
        for key, value in batch_stats.items():
            epoch_stats[key] += value
        optimizer.zero_grad()
        logits = augment_model.logits_from_embeddings(domain_e, user_e, item_e)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    epoch_stats["num_batches"] = batch_count
    return total_loss / max(batch_count, 1), epoch_stats


def train_stage4_branches(
        backbone: DiffMSRIdModel,
        diffusion0: CDCDRMlpDiffusion,
        diffusion1: CDCDRMlpDiffusion,
        classifier: DomainClassifier,
        source_loader: DataLoader,
        target_loader: DataLoader,
        target_pool: Interaction,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        target_domain_idx: int,
        device: torch.device,
) -> Tuple[DiffMSRIdModel, DiffMSRIdModel]:
    def safe_ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    baseline_model = deepcopy(backbone).to(device)
    augment_model = deepcopy(backbone).to(device)
    baseline_model.freeze_embeddings_train_head()
    augment_model.freeze_embeddings_train_head()

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

    for epoch in range(STAGE4_EPOCHS):
        baseline_loss = train_target_true_epoch(
            baseline_model,
            target_loader,
            target_domain_idx,
            baseline_optimizer,
            device,
        )
        augment_loss, augment_stats = train_augmented_epoch(
            augment_model,
            backbone,
            diffusion0,
            diffusion1,
            classifier,
            source_loader,
            target_pool,
            target_domain_idx,
            augment_optimizer,
            device,
        )
        print(f"[Stage4][Epoch {epoch}] baseline_loss={baseline_loss:.6f}, augment_loss={augment_loss:.6f}")
        print(
            f"[Stage4][Epoch {epoch}][AugStats] "
            f"source_candidates={augment_stats['source_candidate_total']} "
            f"(label0={augment_stats['source_candidate_label0']}, label1={augment_stats['source_candidate_label1']}), "
            f"transferred={augment_stats['transferred_total']} "
            f"(label0={augment_stats['transferred_label0']}, label1={augment_stats['transferred_label1']}), "
            f"transfer_rate={safe_ratio(augment_stats['transferred_total'], augment_stats['source_candidate_total']):.4%}, "
            f"transfer_rate_label0={safe_ratio(augment_stats['transferred_label0'], augment_stats['source_candidate_label0']):.4%}, "
            f"transfer_rate_label1={safe_ratio(augment_stats['transferred_label1'], augment_stats['source_candidate_label1']):.4%}, "
            f"target_true={augment_stats['target_true_total']}, "
            f"target_fake={augment_stats['target_fake_total']}, "
            f"augment_total={augment_stats['augment_total']}, "
            f"transferred_share_in_aug={safe_ratio(augment_stats['transferred_total'], augment_stats['augment_total']):.4%}, "
            f"num_batches={augment_stats['num_batches']}"
        )

        baseline_valid, baseline_test, augment_valid, augment_test = evaluate_branch_pair(
            baseline_model,
            augment_model,
            valid_loader,
            test_loader,
            target_domain_idx,
            device,
            epoch,
        )

        baseline_auc = baseline_valid.get("auc", -float("inf"))
        augment_auc = augment_valid.get("auc", -float("inf"))
        if baseline_auc > best_baseline_auc:
            best_baseline_auc = baseline_auc
            best_baseline_state = deepcopy(baseline_model.state_dict())
            print(f"[Stage4][Epoch {epoch}] new best baseline valid auc={best_baseline_auc:.6f}, test={baseline_test}")
        if augment_auc > best_augment_auc:
            best_augment_auc = augment_auc
            best_augment_state = deepcopy(augment_model.state_dict())
            print(f"[Stage4][Epoch {epoch}] new best augment valid auc={best_augment_auc:.6f}, test={augment_test}")

    baseline_model.load_state_dict(best_baseline_state)
    augment_model.load_state_dict(best_augment_state)
    return baseline_model, augment_model


if __name__ == "__main__":
    from betterbole.utils import auto_queue

    auto_queue()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_domain_raw = TARGET_DOMAIN_RAW
    source_domain_raws = tuple(domain for domain in ALL_RAW_DOMAINS if domain != target_domain_raw)
    work_dir = f"{WORK_DIR_PREFIX}-target{target_domain_raw}"

    settings_list = [
        SparseEmbSetting("user", FeatureSource.USER_ID, EMB_DIM, min_freq=1, use_oov=True),
        SparseEmbSetting("item", FeatureSource.ITEM_ID, EMB_DIM, min_freq=1, use_oov=True),
        SparseEmbSetting("domain_indicator", FeatureSource.INTERACTION, EMB_DIM, padding_zero=True, use_oov=False),
    ]
    manager = SchemaManager(
        settings_list,
        work_dir,
        label_fields="label",
        domain_fields="domain_indicator",
        time_field="time",
    )

    whole_lf = pl.scan_csv(DATASET)
    train_samples_lf, train_ple_lf, val_ple_lf, test_ple_lf = generate_hybrid_splits_polars(whole_lf)
    train_ple_lf = collapse_domains_after_split(train_ple_lf, target_domain_raw=target_domain_raw)
    val_ple_lf = collapse_domains_after_split(val_ple_lf, target_domain_raw=target_domain_raw)
    test_ple_lf = collapse_domains_after_split(test_ple_lf, target_domain_raw=target_domain_raw)
    print(
        f"Target raw domain={target_domain_raw}; merged source raw domains={source_domain_raws}; "
        f"post-split collapsed domains: source->{MERGED_SOURCE_DOMAIN_RAW}, target->{MERGED_TARGET_DOMAIN_RAW}"
    )
    print_split_statistics(train_samples_lf, train_ple_lf, val_ple_lf, test_ple_lf)

    manager.fit(train_ple_lf)
    source_domain_idx = mapped_domain_id(manager, MERGED_SOURCE_DOMAIN_RAW)
    target_domain_idx = mapped_domain_id(manager, MERGED_TARGET_DOMAIN_RAW)
    print(
        f"Mapped domains after collapse: "
        f"source {MERGED_SOURCE_DOMAIN_RAW}->{source_domain_idx}, "
        f"target {MERGED_TARGET_DOMAIN_RAW}->{target_domain_idx}"
    )

    train_lf = manager.transform(train_ple_lf).sort(by="time").collect()
    valid_lf = manager.transform(val_ple_lf).sort(by="time").collect()
    test_lf = manager.transform(test_ple_lf).sort(by="time").collect()

    source_train_lf = train_lf.filter(pl.col(manager.domain_field) == source_domain_idx)
    target_train_lf = train_lf.filter(pl.col(manager.domain_field) == target_domain_idx)
    target_train_0_lf = target_train_lf.filter(pl.col(manager.label_field) == 0)
    target_train_1_lf = target_train_lf.filter(pl.col(manager.label_field) == 1)

    train_interaction = lf_to_interaction(train_lf)
    source_train_interaction = lf_to_interaction(source_train_lf)
    target_train_interaction = lf_to_interaction(target_train_lf)
    target_train_0_interaction = lf_to_interaction(target_train_0_lf)
    target_train_1_interaction = lf_to_interaction(target_train_1_lf)
    valid_interaction = lf_to_interaction(valid_lf)
    test_interaction = lf_to_interaction(test_lf)

    train_loader = make_loader(train_interaction, TRAIN_BATCH_SIZE, shuffle=True)
    source_aug_loader = make_loader(source_train_interaction, SOURCE_AUG_BATCH_SIZE, shuffle=True)
    target_loader = make_loader(target_train_interaction, TRAIN_BATCH_SIZE, shuffle=True)
    target_train_0_loader = make_loader(target_train_0_interaction, TRAIN_BATCH_SIZE, shuffle=True)
    target_train_1_loader = make_loader(target_train_1_interaction, TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = make_loader(valid_interaction, EVAL_BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_interaction, EVAL_BATCH_SIZE, shuffle=False)

    backbone = DiffMSRIdModel(manager, embedding_size=EMB_DIM).to(device)
    diffusion0 = CDCDRMlpDiffusion(
        DDIMScheduler(num_train_timesteps=TIMESTEP, schedule_type=DIFFUSION_SCHEDULE, beta_start=DIFFUSION_BETA),
        EMB_DIM,
        uncon_p=0.0,
        num_fields=2,
        objective=DIFFUSION_OBJECTIVE,
    ).to(device)
    diffusion1 = CDCDRMlpDiffusion(
        DDIMScheduler(num_train_timesteps=TIMESTEP, schedule_type=DIFFUSION_SCHEDULE, beta_start=DIFFUSION_BETA),
        EMB_DIM,
        uncon_p=0.0,
        num_fields=2,
        objective=DIFFUSION_OBJECTIVE,
    ).to(device)
    classifier = DomainClassifier(EMB_DIM, num_fields=2).to(device)

    checkpoint_dir = manager.work_dir / "diffmsr_id_stage_ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Stage 1: Backbone pretrain =====")
    backbone = train_backbone_stage(backbone, train_loader, valid_loader, target_domain_idx, device)
    torch.save(backbone.state_dict(), checkpoint_dir / "stage1_backbone.pt")

    print("\n===== Stage 2: Target-domain diffusion label=0/1 =====")
    diffusion0 = train_single_diffusion(backbone, diffusion0, target_train_0_loader, device, "label0")
    diffusion1 = train_single_diffusion(backbone, diffusion1, target_train_1_loader, device, "label1")
    torch.save(diffusion0.state_dict(), checkpoint_dir / "stage2_diffusion_label0.pt")
    torch.save(diffusion1.state_dict(), checkpoint_dir / "stage2_diffusion_label1.pt")

    print("\n===== Stage 3: Domain classifier =====")
    classifier = train_classifier_stage(
        backbone,
        diffusion0,
        classifier,
        train_loader,
        source_domain_idx,
        target_domain_idx,
        device,
    )
    torch.save(classifier.state_dict(), checkpoint_dir / "stage3_domain_classifier.pt")

    print("\n===== Stage 4: Branch baseline vs diffusion-augment =====")
    baseline_model, augment_model = train_stage4_branches(
        backbone,
        diffusion0,
        diffusion1,
        classifier,
        source_aug_loader,
        target_loader,
        target_train_interaction,
        valid_loader,
        test_loader,
        target_domain_idx,
        device,
    )
    torch.save(baseline_model.state_dict(), checkpoint_dir / "stage4_baseline_model.pt")
    torch.save(augment_model.state_dict(), checkpoint_dir / "stage4_augment_model.pt")

    print("\n===== Final branch metrics on target domain =====")
    final_baseline_valid = evaluate_auc_logloss(
        baseline_model,
        valid_loader,
        target_domain_idx,
        device,
        STAGE4_EPOCHS,
        "final-baseline-valid",
    )
    final_baseline_test = evaluate_auc_logloss(
        baseline_model,
        test_loader,
        target_domain_idx,
        device,
        STAGE4_EPOCHS,
        "final-baseline-test",
    )
    final_augment_valid = evaluate_auc_logloss(
        augment_model,
        valid_loader,
        target_domain_idx,
        device,
        STAGE4_EPOCHS,
        "final-augment-valid",
    )
    final_augment_test = evaluate_auc_logloss(
        augment_model,
        test_loader,
        target_domain_idx,
        device,
        STAGE4_EPOCHS,
        "final-augment-test",
    )
    print(
        "Final comparison:",
        {
            "baseline_valid": final_baseline_valid,
            "baseline_test": final_baseline_test,
            "augment_valid": final_augment_valid,
            "augment_test": final_augment_test,
        },
    )
