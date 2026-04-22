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
WORK_DIR = "workspace/diffmsr-amazonL-multi-workdir"
DATASET = DiffExperimentDataset.AMAZON_LARGE

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 40
STAGE3_EPOCHS = 15
STAGE4_EPOCHS = 10

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
        domain_idx: Optional[int],
        device: torch.device,
        epoch: int,
        name: str,
) -> dict:
    evaluator = Evaluator("AUC", "logloss")
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
            evaluator.collect_pointwise(
                subset[model.USER].detach().cpu(),
                subset[model.LABEL].float().detach().cpu(),
                scores.detach().cpu(),
            )
    metrics = evaluator.summary()
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
    auc_summary = ", ".join(
        [f"domain{raw_domain}={metrics[f'domain{raw_domain}'].get('auc', float('nan')):.6f}" for raw_domain, _ in domain_specs]
        + [f"overall={metrics['overall'].get('auc', float('nan')):.6f}"]
    )
    print(f"[AUCSummary][{name}][Epoch {epoch}] {auc_summary}")
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
) -> DiffMSRIdModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_STAGE1, weight_decay=WEIGHT_DECAY)
    best_state = deepcopy(model.state_dict())
    best_auc = -float("inf")
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
            best_state = deepcopy(model.state_dict())
            best_test_metrics = evaluate_all_domain_metrics(model, test_loader, domain_specs, device, epoch, "stage1-best-test")
            print(f"[Stage1][Epoch {epoch}] new best overall valid auc={best_auc:.6f}, best_test={best_test_metrics}")
        print(f"[Stage1][Epoch {epoch}] backbone loss={avg_loss:.6f}, best_overall_auc={best_auc:.6f}")
    if best_test_metrics is not None:
        print(f"[Stage1] best-epoch synced test metrics: {best_test_metrics}")
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
    epoch_stats = {"per_domain": {}}

    for raw_domain, target_domain_idx in domain_specs:
        diffusion0 = diffusions_by_domain[(target_domain_idx, 0)]
        diffusion1 = diffusions_by_domain[(target_domain_idx, 1)]
        classifier = classifiers_by_domain[target_domain_idx]
        source_loader = source_loaders_by_domain[target_domain_idx]
        target_pool = target_pools_by_domain[target_domain_idx]

        diffusion0.eval()
        diffusion1.eval()
        classifier.eval()

        domain_stats = {
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
                if key in domain_stats:
                    domain_stats[key] += value
            optimizer.zero_grad()
            logits = augment_model.logits_from_embeddings(domain_e, user_e, item_e)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            domain_stats["num_batches"] += 1

        epoch_stats["per_domain"][raw_domain] = domain_stats

    epoch_stats["num_batches"] = batch_count
    return total_loss / max(batch_count, 1), epoch_stats


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
            f"num_batches={stats['num_batches']}"
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
    metrics = {}

    for raw_domain, mapped_domain in domain_specs:
        best_info = best_states_by_domain.get(raw_domain)
        if best_info is None or best_info.get("state") is None:
            raise ValueError(f"domain{raw_domain} 缺少 best checkpoint，无法恢复评估。")
        model.load_state_dict(best_info["state"])
        domain_evaluator = Evaluator("AUC", "logloss")
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                mask = batch[model.DOMAIN].long() == mapped_domain
                if not mask.any().item():
                    continue
                subset = batch[mask]
                scores = model.predict(subset, domain_override=mapped_domain)
                domain_evaluator.collect_pointwise(
                    subset[model.USER].detach().cpu(),
                    subset[model.LABEL].float().detach().cpu(),
                    scores.detach().cpu(),
                )
                overall_evaluator.collect_pointwise(
                    subset[model.USER].detach().cpu(),
                    subset[model.LABEL].float().detach().cpu(),
                    scores.detach().cpu(),
                )
        metrics[f"domain{raw_domain}"] = domain_evaluator.summary()
        print(f"[Metrics][{name}-domain{raw_domain}][Epoch {epoch}] {metrics[f'domain{raw_domain}']}")

    metrics["overall"] = overall_evaluator.summary()
    print(f"[Metrics][{name}-overall][Epoch {epoch}] {metrics['overall']}")
    auc_summary = ", ".join(
        [f"domain{raw_domain}={metrics[f'domain{raw_domain}'].get('auc', float('nan')):.6f}" for raw_domain, _ in domain_specs]
        + [f"overall={metrics['overall'].get('auc', float('nan')):.6f}"]
    )
    print(f"[AUCSummary][{name}][Epoch {epoch}] {auc_summary}")
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
) -> Tuple[DiffMSRIdModel, DiffMSRIdModel, dict, dict]:
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
        for raw_domain, _ in domain_specs:
            domain_key = f"domain{raw_domain}"
            baseline_domain_auc = baseline_valid[domain_key].get("auc", -float("inf"))
            if baseline_domain_auc > baseline_best_by_domain[raw_domain]["auc"]:
                baseline_best_by_domain[raw_domain] = {
                    "auc": baseline_domain_auc,
                    "state": deepcopy(baseline_model.state_dict()),
                    "epoch": epoch,
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
                }
                print(
                    f"[Stage4][Epoch {epoch}] new best augment domain{raw_domain} valid auc="
                    f"{augment_domain_auc:.6f}"
                )
        if baseline_auc > best_baseline_auc:
            best_baseline_auc = baseline_auc
            best_baseline_state = deepcopy(baseline_model.state_dict())
            print(f"[Stage4][Epoch {epoch}] new best baseline overall valid auc={best_baseline_auc:.6f}, test={baseline_test}")
        if augment_auc > best_augment_auc:
            best_augment_auc = augment_auc
            best_augment_state = deepcopy(augment_model.state_dict())
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
    return baseline_model, augment_model, baseline_best_by_domain, augment_best_by_domain


if __name__ == "__main__":
    from betterbole.utils import auto_queue

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

    backbone = DiffMSRIdModel(manager, embedding_size=EMB_DIM).to(device)

    checkpoint_dir = manager.work_dir / "diffmsr_id_stage_ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Stage 1: Backbone pretrain =====")
    backbone = train_backbone_stage(backbone, train_loader, valid_loader, test_loader, domain_specs, device)
    torch.save(backbone.state_dict(), checkpoint_dir / "stage1_backbone.pt")

    domain_interactions = {}
    domain_label_loaders = {}
    source_loaders_by_domain = {}
    target_pools_by_domain = {}
    diffusions_by_domain = {}
    classifiers_by_domain = {}

    for raw_domain, mapped_domain in domain_specs:
        domain_lf = train_lf.filter(pl.col(manager.domain_field) == mapped_domain)
        domain_interaction = lf_to_interaction(domain_lf)
        domain_interactions[mapped_domain] = domain_interaction
        target_pools_by_domain[mapped_domain] = domain_interaction
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
    baseline_model, augment_model, baseline_best_by_domain, augment_best_by_domain = train_stage4_branches(
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

    print("\n===== Final branch metrics on all domains (per-domain best checkpoints) =====")
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
    print(
        "Final comparison:",
        {
            "baseline_valid": final_baseline_valid,
            "baseline_test": final_baseline_test,
            "augment_valid": final_augment_valid,
            "augment_test": final_augment_test,
        },
    )
