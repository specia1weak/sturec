import argparse
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch

from betterbole.models.generative.diffusion.diffusions import CDCDRMlpDiffusion
from betterbole.models.generative.diffusion.schedulers import DDIMScheduler
from betterbole.utils import change_root_workdir

from expmodel import DomainClassifier
from exptools import (
    evaluate_with_domain_best_states,
    extract_auc_summary,
    lf_to_interaction,
    make_loader,
    make_model,
    mapped_domain_id,
    print_domain_count_statistics,
    print_split_statistics,
    save_experiment_record,
    strip_model_states,
    summarize_best_by_domain,
    train_backbone_stage,
    train_classifier_stage,
    train_cold_start_single_domain_baselines,
    train_single_diffusion,
    train_stage4_branches,
)
from settings import DiffExperimentDataset, build_schema_manager, build_settings
from split_tool import generate_hybrid_splits_polars


DATASET_OPTIONS = [
    ("amazon-small", DiffExperimentDataset.AMAZON_SMALL),
    ("amazon-large", DiffExperimentDataset.AMAZON_LARGE),
    ("douban", DiffExperimentDataset.DOUBAN),
]

BACKBONE_OPTIONS = [
    ("sharedbottom", {}),
    ("mmoe", {}),
    ("ple", {}),
    ("star", {}),
    ("m2m", {}),
    ("epnet", {}),
    ("ppnet", {}),
    ("m3oe", {}),
]


def seed_everything(seed: int = 2026) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_option(options, index: int, option_type: str):
    if index < 0 or index >= len(options):
        available = ", ".join(f"{idx}:{name}" for idx, (name, _) in enumerate(options))
        raise ValueError(f"Invalid {option_type}_index={index}. Available: {available}")
    return options[index]


def preset_custom(dataset_name: str, backbone_name: str, suffix: str = "msr") -> str:
    suffix = suffix.strip("-")
    return f"{suffix}-{dataset_name}-{backbone_name}" if suffix else f"{dataset_name}-{backbone_name}"


def run_experiment(
        dataset_index: int = 0,
        backbone_index: int = 0,
        custom_suffix: str = "msr",
        custom: Optional[str] = None,
) -> dict:
    change_root_workdir()
    seed_everything(2026)
    dataset_name, dataset_path = select_option(DATASET_OPTIONS, dataset_index, "dataset")
    backbone_name, backbone_params = select_option(BACKBONE_OPTIONS, backbone_index, "backbone")
    custom = custom or preset_custom(dataset_name, backbone_name, custom_suffix)
    cfg = build_settings(
        dataset=dataset_path,
        custom=custom,
        backbone_name=backbone_name,
        backbone_params=dict(backbone_params),
    )
    print(
        "[ExperimentOptions]",
        {
            "seed": 2026,
            "dataset_index": dataset_index,
            "dataset": dataset_name,
            "backbone_index": backbone_index,
            "backbone": backbone_name,
            "custom": custom,
            "work_dir": cfg.resolved_work_dir,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = build_schema_manager(cfg)
    dataset_path = Path(cfg.dataset)

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
    domain_specs = tuple((raw_domain, mapped_domain_id(manager, raw_domain)) for raw_domain in cfg.all_raw_domains)
    print(f"Mapped domains: {domain_specs}")

    train_lf = manager.transform(train_ple_lf).sort(by="time").collect()
    valid_lf = manager.transform(val_ple_lf).sort(by="time").collect()
    test_lf = manager.transform(test_ple_lf).sort(by="time").collect()

    train_interaction = lf_to_interaction(train_lf)
    valid_interaction = lf_to_interaction(valid_lf)
    test_interaction = lf_to_interaction(test_lf)

    train_loader = make_loader(train_interaction, cfg.train_batch_size, shuffle=True)
    valid_loader = make_loader(valid_interaction, cfg.eval_batch_size, shuffle=False)
    test_loader = make_loader(test_interaction, cfg.eval_batch_size, shuffle=False)

    domain_label_loaders = {}
    source_loaders_by_domain = {}
    target_pools_by_domain = {}
    target_loaders_by_domain = {}
    diffusions_by_domain = {}
    classifiers_by_domain = {}

    for raw_domain, mapped_domain in domain_specs:
        domain_lf = train_lf.filter(pl.col(manager.domain_field) == mapped_domain)
        domain_interaction = lf_to_interaction(domain_lf)
        target_pools_by_domain[mapped_domain] = domain_interaction
        target_loaders_by_domain[mapped_domain] = make_loader(
            domain_interaction,
            cfg.train_batch_size,
            shuffle=True,
        )
        domain_label_loaders[(mapped_domain, 0)] = make_loader(
            lf_to_interaction(domain_lf.filter(pl.col(manager.label_field) == 0)),
            cfg.train_batch_size,
            shuffle=True,
        )
        domain_label_loaders[(mapped_domain, 1)] = make_loader(
            lf_to_interaction(domain_lf.filter(pl.col(manager.label_field) == 1)),
            cfg.train_batch_size,
            shuffle=True,
        )
        source_lf = train_lf.filter(pl.col(manager.domain_field) != mapped_domain)
        source_loaders_by_domain[mapped_domain] = make_loader(
            lf_to_interaction(source_lf),
            cfg.resolved_source_aug_batch_size,
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
        cfg,
    )
    for raw_domain, _ in domain_specs:
        torch.save(stage0_best_by_domain[raw_domain]["state"], checkpoint_dir / f"stage0_domain{raw_domain}_single_domain.pt")

    backbone = make_model(manager, domain_specs, cfg).to(device)

    print("\n===== Stage 1: Backbone pretrain =====")
    backbone, stage1_best_info = train_backbone_stage(
        backbone,
        train_loader,
        valid_loader,
        test_loader,
        domain_specs,
        device,
        cfg,
    )
    torch.save(backbone.state_dict(), checkpoint_dir / "stage1_backbone.pt")

    print("\n===== Stage 2: Per-domain diffusion label=0/1 =====")
    for raw_domain, mapped_domain in domain_specs:
        for label_value in (0, 1):
            diffusion = CDCDRMlpDiffusion(
                DDIMScheduler(
                    num_train_timesteps=cfg.timestep,
                    schedule_type=cfg.diffusion_schedule,
                    beta_start=cfg.diffusion_beta,
                ),
                cfg.emb_dim,
                uncon_p=0.0,
                num_fields=2,
                objective=cfg.diffusion_objective,
            ).to(device)
            diffusion = train_single_diffusion(
                backbone,
                diffusion,
                domain_label_loaders[(mapped_domain, label_value)],
                device,
                f"domain{raw_domain}-label{label_value}",
                cfg,
            )
            diffusions_by_domain[(mapped_domain, label_value)] = diffusion
            torch.save(diffusion.state_dict(), checkpoint_dir / f"stage2_domain{raw_domain}_label{label_value}.pt")

    print("\n===== Stage 3: Per-domain domain classifier =====")
    for raw_domain, mapped_domain in domain_specs:
        classifier = train_classifier_stage(
            backbone,
            diffusions_by_domain[(mapped_domain, 0)],
            DomainClassifier(cfg.emb_dim, num_fields=2).to(device),
            train_loader,
            mapped_domain,
            device,
            cfg,
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
        cfg,
        work_dir=manager.work_dir,
    )
    torch.save(baseline_model.state_dict(), checkpoint_dir / "stage4_baseline_model.pt")
    torch.save(augment_model.state_dict(), checkpoint_dir / "stage4_augment_model.pt")
    if stage4_info.get("embedding_dump") is not None:
        print(f"[Stage4][EmbeddingDump] {stage4_info['embedding_dump']}")
    if stage4_info.get("baseline_weighted_best") is not None:
        torch.save(stage4_info["baseline_weighted_best"]["state"], checkpoint_dir / "stage4_baseline_weighted_best.pt")
    if stage4_info.get("augment_weighted_best") is not None:
        torch.save(stage4_info["augment_weighted_best"]["state"], checkpoint_dir / "stage4_augment_weighted_best.pt")
    if stage4_info.get("baseline_overall_best") is not None:
        torch.save(stage4_info["baseline_overall_best"]["state"], checkpoint_dir / "stage4_baseline_overall_best.pt")
    if stage4_info.get("augment_overall_best") is not None:
        torch.save(stage4_info["augment_overall_best"]["state"], checkpoint_dir / "stage4_augment_overall_best.pt")
    for raw_domain, _ in domain_specs:
        torch.save(baseline_best_by_domain[raw_domain]["state"], checkpoint_dir / f"stage4_baseline_domain{raw_domain}_best.pt")
        torch.save(augment_best_by_domain[raw_domain]["state"], checkpoint_dir / f"stage4_augment_domain{raw_domain}_best.pt")

    print("\n===== Final metrics on all domains (per-domain best checkpoints) =====")
    stage0_eval_model = make_model(manager, domain_specs, cfg).to(device)
    final_stage0_valid = evaluate_with_domain_best_states(
        stage0_eval_model,
        valid_loader,
        stage0_best_by_domain,
        domain_specs,
        device,
        cfg.resolved_stage0_epochs,
        "final-stage0-valid",
    )
    final_stage0_test = evaluate_with_domain_best_states(
        stage0_eval_model,
        test_loader,
        stage0_best_by_domain,
        domain_specs,
        device,
        cfg.resolved_stage0_epochs,
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
        cfg.stage4_epochs,
        "final-baseline-valid",
    )
    final_baseline_test = evaluate_with_domain_best_states(
        baseline_model,
        test_loader,
        baseline_best_by_domain,
        domain_specs,
        device,
        cfg.stage4_epochs,
        "final-baseline-test",
    )
    final_augment_valid = evaluate_with_domain_best_states(
        augment_model,
        valid_loader,
        augment_best_by_domain,
        domain_specs,
        device,
        cfg.stage4_epochs,
        "final-augment-valid",
    )
    final_augment_test = evaluate_with_domain_best_states(
        augment_model,
        test_loader,
        augment_best_by_domain,
        domain_specs,
        device,
        cfg.stage4_epochs,
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
        "seed": 2026,
        "dataset": str(dataset_path),
        "work_dir": str(manager.work_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "domain_specs": [
            {"raw_domain": raw_domain, "mapped_domain": mapped_domain}
            for raw_domain, mapped_domain in domain_specs
        ],
        "domain_count_summary": domain_count_summary,
        "hyperparameters": cfg.to_record(),
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
            "stage4_embedding_dump": stage4_info.get("embedding_dump"),
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
    record_path = manager.work_dir / "diffmsr_amazonS_multi_msr_experiment_record.refactor.json"
    save_experiment_record(strip_model_states(experiment_record), record_path)
    print(f"[ExperimentRecord] saved to {record_path}")
    print("Final comparison:", final_comparison)
    return strip_model_states(experiment_record)


def validate_single_index(index: int, max_index: int, arg_name: str) -> int:
    if index < 0 or index > max_index:
        raise ValueError(f"{arg_name} must be within 0..{max_index}, got {index}")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasetindex",
        type=int,
        default=0,
        help="Single dataset index, e.g. 0",
    )
    parser.add_argument(
        "--backboneindex",
        type=int,
        default=0,
        help="Single backbone index, e.g. 0",
    )
    parser.add_argument(
        "--customsuffix",
        default="msr",
        help="Suffix used to auto-build custom/workdir names",
    )
    args = parser.parse_args()

    dataset_index = validate_single_index(args.datasetindex, len(DATASET_OPTIONS) - 1, "datasetindex")
    backbone_index = validate_single_index(args.backboneindex, len(BACKBONE_OPTIONS) - 1, "backboneindex")
    run_experiment(dataset_index=dataset_index, backbone_index=backbone_index, custom_suffix=args.customsuffix)
