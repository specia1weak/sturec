from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from betterbole.core.enum_type import FeatureSource
from betterbole.emb import SchemaManager
from betterbole.emb.schema import SparseEmbSetting


class DiffExperimentDataset:
    BASE_DIR = Path("raw/diffmsr")
    AMAZON_LARGE = BASE_DIR / "amazon_time.csv"
    AMAZON_SMALL = BASE_DIR / "amazonsmall.csv"
    DOUBAN = BASE_DIR / "douban3.csv"


PathLike = Union[Path, str]


def dataset_name(dataset: PathLike) -> str:
    return Path(dataset).stem.replace("_", "-").replace(" ", "-")


def preset_workdir(
        custom: str,
        dataset: PathLike = DiffExperimentDataset.AMAZON_SMALL,
        base_dir: PathLike = "workspace",
) -> str:
    custom = custom.strip("-")
    return str(Path(base_dir) / f"{dataset_name(dataset)}-multi-{custom}")


@dataclass
class DiffMSRExperimentSettings:
    dataset: Path = DiffExperimentDataset.AMAZON_SMALL
    custom: str = "msr-workdir"
    work_dir: Optional[str] = None

    all_raw_domains: Tuple[int, ...] = (0, 1, 2)
    emb_dim: int = 16
    train_batch_size: int = 2048
    source_aug_batch_multiplier: int = 10
    source_aug_batch_size: Optional[int] = None
    eval_batch_size: int = 4096

    timestep: int = 500
    diffusion_beta: float = 0.0002
    diffusion_schedule: str = "other"
    diffusion_objective: str = "pred_v"

    shared_bottom_hidden_dims: Tuple[int, ...] = (256, 256)
    domain_head_hidden_dims: Tuple[int, ...] = (256,)
    head_dropout: float = 0.0
    backbone_name: str = "sharedbottom"
    backbone_params: Optional[Dict[str, Any]] = None
    stage4_freeze_embeddings: bool = True
    stage4_freeze_shared_bottom: bool = True

    stage1_epochs: int = 8
    stage2_epochs: int = 40
    stage3_epochs: int = 15
    stage4_epochs: int = 10
    stage0_epochs: Optional[int] = None

    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 1e-4
    learning_rate_stage3: float = 1e-3
    learning_rate_stage4: float = 2e-3
    learning_rate_stage0: Optional[float] = None
    weight_decay: float = 1e-7

    classifier_noise_max_step: int = 70
    aug_step1: int = 30
    aug_step2: int = 50
    aug_target_sample_size: int = 512
    transfer_diagnostic_thresholds: Tuple[float, ...] = (0.5, 0.6, 0.7)
    transfer_diagnostic_top_steps: int = 5

    @property
    def resolved_work_dir(self) -> str:
        return self.work_dir or preset_workdir(self.custom, self.dataset)

    @property
    def resolved_source_aug_batch_size(self) -> int:
        return self.source_aug_batch_size or self.train_batch_size * self.source_aug_batch_multiplier

    @property
    def resolved_stage0_epochs(self) -> int:
        return self.stage0_epochs if self.stage0_epochs is not None else self.stage1_epochs

    @property
    def resolved_learning_rate_stage0(self) -> float:
        return self.learning_rate_stage0 if self.learning_rate_stage0 is not None else self.learning_rate_stage1

    def with_overrides(self, **overrides):
        return replace(self, **overrides)

    def to_record(self) -> dict:
        return {
            "dataset": str(self.dataset),
            "custom": self.custom,
            "work_dir": self.resolved_work_dir,
            "all_raw_domains": self.all_raw_domains,
            "emb_dim": self.emb_dim,
            "train_batch_size": self.train_batch_size,
            "source_aug_batch_size": self.resolved_source_aug_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "timestep": self.timestep,
            "diffusion_beta": self.diffusion_beta,
            "diffusion_schedule": self.diffusion_schedule,
            "diffusion_objective": self.diffusion_objective,
            "shared_bottom_hidden_dims": self.shared_bottom_hidden_dims,
            "domain_head_hidden_dims": self.domain_head_hidden_dims,
            "head_dropout": self.head_dropout,
            "backbone_name": self.backbone_name,
            "backbone_params": self.backbone_params,
            "stage4_freeze_embeddings": self.stage4_freeze_embeddings,
            "stage4_freeze_shared_bottom": self.stage4_freeze_shared_bottom,
            "stage0_epochs": self.resolved_stage0_epochs,
            "stage1_epochs": self.stage1_epochs,
            "stage2_epochs": self.stage2_epochs,
            "stage3_epochs": self.stage3_epochs,
            "stage4_epochs": self.stage4_epochs,
            "learning_rate_stage0": self.resolved_learning_rate_stage0,
            "learning_rate_stage1": self.learning_rate_stage1,
            "learning_rate_stage2": self.learning_rate_stage2,
            "learning_rate_stage3": self.learning_rate_stage3,
            "learning_rate_stage4": self.learning_rate_stage4,
            "weight_decay": self.weight_decay,
            "classifier_noise_max_step": self.classifier_noise_max_step,
            "aug_step1": self.aug_step1,
            "aug_step2": self.aug_step2,
            "aug_target_sample_size": self.aug_target_sample_size,
            "transfer_diagnostic_thresholds": self.transfer_diagnostic_thresholds,
            "transfer_diagnostic_top_steps": self.transfer_diagnostic_top_steps,
        }


def build_settings(
        dataset: PathLike = DiffExperimentDataset.AMAZON_SMALL,
        custom: str = "msr-workdir",
        work_dir: Optional[str] = None,
        **overrides,
) -> DiffMSRExperimentSettings:
    return DiffMSRExperimentSettings(
        dataset=Path(dataset),
        custom=custom,
        work_dir=work_dir,
    ).with_overrides(**overrides)


def build_schema_manager(settings: DiffMSRExperimentSettings) -> SchemaManager:
    settings_list = [
        SparseEmbSetting("user", FeatureSource.USER_ID, settings.emb_dim, min_freq=1, use_oov=True),
        SparseEmbSetting("item", FeatureSource.ITEM_ID, settings.emb_dim, min_freq=1, use_oov=True),
        SparseEmbSetting("domain_indicator", FeatureSource.INTERACTION, settings.emb_dim, padding_zero=True, use_oov=False),
    ]
    return SchemaManager(
        settings_list,
        settings.resolved_work_dir,
        label_fields="label",
        domain_fields="domain_indicator",
        time_field="time",
    )
