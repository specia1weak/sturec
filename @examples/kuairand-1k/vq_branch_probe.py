import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import polars as pl
import torch

from betterbole.core.enum_type import FeatureSource
from betterbole.core.train import EarlyStopper
from betterbole.core.train.context import TrainerComponents, TrainerDataLoaders
from betterbole.core.train.trainer import BaseTrainer
from betterbole.data.dataset import ParquetStreamDataset
from betterbole.datasets.kuairand import KuaiRandDataset
from betterbole.emb import SchemaManager
from betterbole.emb.schema import MultiSparseSetting, SparseEmbSetting
from betterbole.evaluate.evaluator import Evaluator
from betterbole.evaluate.manager import DomainFilter, EvaluatorManager
from betterbole.experiment import WORKSPACE, change_root_workdir
from betterbole.experiment.param import ConfigBase, ParamManager
from betterbole.models.base import BaseModel
from betterbole.models.msr import build_model, update_register
from betterbole.utils.optimize import split_params_by_decay
from custom_models import CUSTOM_MODEL_REGISTRY


change_root_workdir()
update_register(**CUSTOM_MODEL_REGISTRY)


@dataclass
class ProbeConfig(ConfigBase):
    dataset_name: str = "kuairand-rand"
    seed: int = 2024
    device: str = "cuda"
    max_epochs: int = 2
    ckpt_dir: str = ""
    batch_size: int = 4096
    id_emb: int = 32
    side_emb: int = 32
    shuffle_buffer_size: int = 2000000
    model: str = "vq_share_gate_v1"
    log_name: str = "vq_branch_probe.log"
    weight_decay: float = 1e-6
    report_name: str = "vq_branch_probe_seed2024.json"


MODEL_KWARGS: Dict[str, Dict[str, object]] = {
    "ple": {
        "num_levels": 2,
        "num_shared_experts": 1,
    },
    "shavq_h2_ste_v1": {
        "warmup_samples": 8192,
        "shared_warmup_samples": 8192,
        "specific_warmup_samples": 8192,
    },
    "shavq_h2_ste_v1_r001": {
        "warmup_samples": 8192,
        "shared_warmup_samples": 8192,
        "specific_warmup_samples": 8192,
        "residual_diversity_weight": 0.01,
    },
    "shavq_h2_ste_v1_r001_b050": {
        "warmup_samples": 8192,
        "shared_warmup_samples": 8192,
        "specific_warmup_samples": 8192,
        "residual_diversity_weight": 0.01,
        "specific_residual_scale": 0.5,
    },
    "shavq_h2_ste_v1_r001_fuse": {
        "warmup_samples": 8192,
        "shared_warmup_samples": 8192,
        "specific_warmup_samples": 8192,
        "residual_diversity_weight": 0.01,
        "specific_quantized_fusion": True,
    },
}


class KuairandTrainer(BaseTrainer):
    def __init__(
            self,
            model: BaseModel,
            optimizer: torch.optim.Optimizer,
            manager: SchemaManager,
            loaders: TrainerDataLoaders,
            components: TrainerComponents,
            cfg: ConfigBase,
    ):
        super().__init__(model, optimizer, manager, loaders, components, cfg)

    def train_epoch(self):
        if self.epoch != 0:
            self.model.omni_embedding.reinitialize_large_vocab_embeddings(1001, init_std=1e-2)
            print("重初始化")
        super().train_epoch()


def build_settings(cfg: ProbeConfig):
    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, cfg.id_emb, min_freq=10, use_oov=True)
    item_setting = SparseEmbSetting("video_id", FeatureSource.ITEM_ID, cfg.id_emb, min_freq=10, use_oov=True)
    return [
        user_setting,
        item_setting,
        SparseEmbSetting("tab", FeatureSource.INTERACTION, cfg.side_emb, padding_zero=False, use_oov=False, use_null=False),
        SparseEmbSetting("user_active_degree", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("is_live_streamer", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("is_video_author", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("friend_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("register_days_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("follow_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("fans_user_num_range", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("is_lowactive_period", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat0", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat1", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat2", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat3", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat4", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat5", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat6", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat7", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat8", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat9", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat10", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat11", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat12", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat13", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat14", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat15", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat16", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("onehot_feat17", FeatureSource.USER, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("author_id", FeatureSource.ITEM, cfg.id_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("music_id", FeatureSource.ITEM, cfg.id_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("video_type", FeatureSource.ITEM, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("music_type", FeatureSource.ITEM, cfg.side_emb, min_freq=10, use_oov=True),
        SparseEmbSetting("visible_status", FeatureSource.ITEM, cfg.side_emb, min_freq=10, use_oov=True),
        MultiSparseSetting("tag", FeatureSource.ITEM, cfg.side_emb, max_tag_len=5, is_string_format=True, separator=",", min_freq=10, use_oov=True),
    ]


def build_data(cfg: ProbeConfig):
    settings_list = build_settings(cfg)
    workdir = WORKSPACE / cfg.dataset_name
    manager = SchemaManager(settings_list, workdir, time_field="time_ms", label_fields="is_click", domain_fields="tab")

    user_lf = pl.scan_csv(KuaiRandDataset.USER_FEATURES)
    item_lf = pl.scan_csv(KuaiRandDataset.VIDEO_FEATURES)
    inter_lf = pl.scan_csv([KuaiRandDataset.STD_LOG_FORMER_DATA_P1, KuaiRandDataset.STD_LOG_FORMER_DATA_P2, KuaiRandDataset.RAND_LOG_FORMER_DATA])
    whole_lf = inter_lf.join(item_lf, on="video_id", how="left").join(user_lf, on="user_id", how="left")

    top_5_df = whole_lf.group_by("tab").len().sort("len", descending=True).head(5).select("tab").collect()
    top_5_list = top_5_df.get_column("tab").to_list()
    whole_lf = whole_lf.filter(pl.col("tab").is_in(top_5_list))

    whole_lf = whole_lf.with_row_index("__row_idx").with_columns(
        (pl.col("__row_idx") * 1103515245 + 12345).cast(pl.UInt64).alias("__rand_order")
    ).drop("__row_idx")
    whole_lf = manager.make_checkpoint(whole_lf, redo=False, sort_by="__rand_order").drop("__rand_order")

    train_raw = whole_lf.filter(pl.col("date") <= 20220506)
    valid_raw = whole_lf.filter(pl.col("date") == 20220507)
    test_raw = whole_lf.filter(pl.col("date") == 20220508)

    manager.fit(train_raw, low_memory=True)
    train_lf = manager.transform(train_raw)
    valid_lf = manager.transform(valid_raw)
    test_lf = manager.transform(test_raw)
    train_path, valid_path, test_path = manager.save_as_dataset(train_lf, valid_lf, test_lf)

    tab_setting = manager.get_setting("tab")
    tab_vocab = getattr(tab_setting, "vocab", {}) if tab_setting is not None else {}
    tab_domain_map = {
        int(raw_tab): int(domain_id)
        for raw_tab, domain_id in sorted(tab_vocab.items(), key=lambda item: int(item[1]))
    }

    return manager, train_path, valid_path, test_path, top_5_list, tab_domain_map


def build_evaluator_manager(manager: SchemaManager, log_path: Path, title: str, num_domains: int) -> EvaluatorManager:
    evaluator_manager = EvaluatorManager(log_path=log_path, title=title)
    evaluator_manager.register("overall", Evaluator("auc"))
    for domain in range(num_domains):
        evaluator_manager.register(
            f"domain{domain}",
            Evaluator("auc"),
            filter_fn=DomainFilter(manager.domain_field, domain),
        )
    return evaluator_manager


def init_domain_debug_stats(num_domains: int):
    names = [
        "shared_logit_abs_sum",
        "specific_logit_abs_sum",
        "shared_norm_sum",
        "residual_norm_sum",
        "quantized_cos_sum",
        "gate_shared_sum",
        "gate_specific_sum",
        "gate_entropy_sum",
        "residual_gain_sum",
        "shared_scale_sum",
        "specific_scale_sum",
    ]
    stats = {"count": torch.zeros(num_domains, dtype=torch.double)}
    for name in names:
        stats[name] = torch.zeros(num_domains, dtype=torch.double)
    return stats


def accumulate_domain_debug_stats(stats, debug, domain_ids: torch.Tensor, num_domains: int):
    domain_ids = domain_ids.long().view(-1).detach().cpu()
    counts = torch.bincount(domain_ids, minlength=num_domains).double()
    stats["count"] += counts

    def add_sum(name: str, values: torch.Tensor):
        values = values.detach().cpu().double().view(-1)
        stats[name].index_add_(0, domain_ids, values)

    add_sum("shared_logit_abs_sum", debug["shared_logits"].abs())
    add_sum("specific_logit_abs_sum", debug["specific_logits"].abs())
    add_sum("shared_norm_sum", debug["shared_norm"])
    add_sum("residual_norm_sum", debug["residual_norm"])
    add_sum("quantized_cos_sum", debug["quantized_cos"])

    branch_gate_weights = debug.get("branch_gate_weights")
    if branch_gate_weights is not None:
        branch_gate_weights = branch_gate_weights.detach().cpu().double()
        gate_entropy = -(branch_gate_weights.clamp_min(1e-12) * branch_gate_weights.clamp_min(1e-12).log()).sum(dim=-1)
        stats["gate_shared_sum"].index_add_(0, domain_ids, branch_gate_weights[:, 0])
        stats["gate_specific_sum"].index_add_(0, domain_ids, branch_gate_weights[:, 1])
        stats["gate_entropy_sum"].index_add_(0, domain_ids, gate_entropy)

    residual_gain = debug.get("residual_gain")
    if residual_gain is not None:
        add_sum("residual_gain_sum", residual_gain)

    shared_scale = debug.get("shared_logit_scale")
    if shared_scale is not None:
        add_sum("shared_scale_sum", shared_scale)

    specific_scale = debug.get("specific_logit_scale")
    if specific_scale is not None:
        add_sum("specific_scale_sum", specific_scale)


def finalize_domain_debug_stats(stats, domain_to_tab_map: Dict[int, int]):
    count = stats["count"].clone()
    result = {}
    for domain_id in range(int(count.numel())):
        denom = max(float(count[domain_id].item()), 1.0)
        shared_abs = float(stats["shared_logit_abs_sum"][domain_id].item() / denom)
        specific_abs = float(stats["specific_logit_abs_sum"][domain_id].item() / denom)
        result[f"domain{domain_id}"] = {
            "tab": int(domain_to_tab_map.get(domain_id, domain_id)),
            "count": int(count[domain_id].item()),
            "shared_logit_abs_mean": shared_abs,
            "specific_logit_abs_mean": specific_abs,
            "specific_to_shared_abs_ratio": float(specific_abs / max(shared_abs, 1e-12)),
            "shared_norm_mean": float(stats["shared_norm_sum"][domain_id].item() / denom),
            "residual_norm_mean": float(stats["residual_norm_sum"][domain_id].item() / denom),
            "quantized_cos_mean": float(stats["quantized_cos_sum"][domain_id].item() / denom),
        }
        if stats["gate_shared_sum"].sum().item() > 0:
            result[f"domain{domain_id}"]["gate_shared_mean"] = float(stats["gate_shared_sum"][domain_id].item() / denom)
            result[f"domain{domain_id}"]["gate_specific_mean"] = float(stats["gate_specific_sum"][domain_id].item() / denom)
            result[f"domain{domain_id}"]["gate_entropy_mean"] = float(stats["gate_entropy_sum"][domain_id].item() / denom)
        if stats["residual_gain_sum"].sum().item() > 0:
            result[f"domain{domain_id}"]["residual_gain_mean"] = float(stats["residual_gain_sum"][domain_id].item() / denom)
        if stats["shared_scale_sum"].sum().item() > 0:
            result[f"domain{domain_id}"]["shared_logit_scale_mean"] = float(stats["shared_scale_sum"][domain_id].item() / denom)
        if stats["specific_scale_sum"].sum().item() > 0:
            result[f"domain{domain_id}"]["specific_logit_scale_mean"] = float(stats["specific_scale_sum"][domain_id].item() / denom)
    return result


@torch.no_grad()
def evaluate_modes(model, manager, data_loader, num_domains: int, device: str, domain_to_tab_map: Dict[int, int]):
    evaluators = {
        mode: build_evaluator_manager(
            manager=manager,
            log_path=None,
            title=f"branch_probe_{mode}",
            num_domains=num_domains,
        )
        for mode in ["full", "shared_only", "specific_only"]
    }
    shared_override_domains = {
        f"full_except_domain{domain}_shared": {domain}
        for domain in range(num_domains)
    }
    shared_override_domains["full_except_domain2_4_shared"] = {2, 4}
    shared_override_domains["full_except_domain3_4_shared"] = {3, 4}
    shared_override_domains["full_except_domain2_3_4_shared"] = {2, 3, 4}

    for policy_name in shared_override_domains:
        evaluators[policy_name] = build_evaluator_manager(
            manager=manager,
            log_path=None,
            title=f"branch_probe_{policy_name}",
            num_domains=num_domains,
        )

    domain_code_rows = []
    domain_debug_stats = init_domain_debug_stats(num_domains)
    for batch_interaction in data_loader:
        batch_interaction = batch_interaction.to(device)
        uids = batch_interaction[manager.uid_field]
        labels = batch_interaction[manager.label_field]
        domain_ids = batch_interaction[manager.domain_field].long().view(-1)

        full_scores = model.predict_with_mode(batch_interaction, mode="full")
        components = model.component_logits()
        shared_scores = components["shared_only"]
        specific_scores = components["specific_only"]

        evaluators["full"].collect(uids, labels, batch_interaction, batch_preds_1d=full_scores)
        evaluators["shared_only"].collect(uids, labels, batch_interaction, batch_preds_1d=shared_scores)
        evaluators["specific_only"].collect(uids, labels, batch_interaction, batch_preds_1d=specific_scores)

        for policy_name, override_set in shared_override_domains.items():
            policy_scores = full_scores.clone()
            mask = torch.zeros_like(domain_ids, dtype=torch.bool)
            for domain_id in override_set:
                mask |= domain_ids == int(domain_id)
            policy_scores[mask] = shared_scores[mask]
            evaluators[policy_name].collect(uids, labels, batch_interaction, batch_preds_1d=policy_scores)

        debug = model._latest_debug
        codes = debug["code_indices"].detach().cpu()
        domains = debug["domain_ids"].detach().cpu()
        domain_code_rows.append(torch.stack([domains, codes], dim=1))
        accumulate_domain_debug_stats(domain_debug_stats, debug, domains, num_domains)

    metrics_by_mode = {name: evaluator.summary() for name, evaluator in evaluators.items()}
    domain_debug_summary = finalize_domain_debug_stats(domain_debug_stats, domain_to_tab_map=domain_to_tab_map)
    return metrics_by_mode, domain_code_rows, domain_debug_summary


def summarize_code_knowledge(domain_code_rows, num_domains: int, codebook_size: int):
    if not domain_code_rows:
        return {}

    pairs = torch.cat(domain_code_rows, dim=0)
    domain_ids = pairs[:, 0].long()
    code_indices = pairs[:, 1].long()

    usage = torch.zeros(num_domains, codebook_size, dtype=torch.long)
    for domain, code in zip(domain_ids.tolist(), code_indices.tolist()):
        usage[domain, code] += 1

    code_totals = usage.sum(dim=0)
    active_codes = code_totals > 0
    if not active_codes.any():
        return {}

    probs = usage[:, active_codes].float()
    probs = probs / probs.sum(dim=0, keepdim=True).clamp_min(1.0)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=0)
    normalized_entropy = entropy / torch.log(torch.tensor(float(num_domains)))
    top_domain_ratio = probs.max(dim=0).values
    cross_domain_count = (usage[:, active_codes] > 0).sum(dim=0).float()

    weighted = code_totals[active_codes].float()
    weighted = weighted / weighted.sum().clamp_min(1.0)

    return {
        "active_code_count": int(active_codes.sum().item()),
        "mean_normalized_domain_entropy": float(normalized_entropy.mean().item()),
        "weighted_normalized_domain_entropy": float((normalized_entropy * weighted).sum().item()),
        "mean_top_domain_ratio": float(top_domain_ratio.mean().item()),
        "weighted_top_domain_ratio": float((top_domain_ratio * weighted).sum().item()),
        "mean_active_domain_count": float(cross_domain_count.mean().item()),
        "codes_used_by_1_domain_ratio": float(((cross_domain_count == 1).float().mean()).item()),
        "codes_used_by_3plus_domains_ratio": float(((cross_domain_count >= 3).float().mean()).item()),
    }


def metric_deltas(full_metrics, branch_metrics):
    result = {}
    for scope, metrics in branch_metrics.items():
        if scope == "full":
            continue
        scope_delta = {}
        for name, sub in metrics.items():
            scope_delta[name] = {}
            for metric_name, value in sub.items():
                scope_delta[name][metric_name] = float(value - full_metrics[name][metric_name])
        result[scope] = scope_delta
    return result


def summarize_shared_override_effects(metrics_by_mode):
    full_metrics = metrics_by_mode["full"]
    results = {}
    for name, metrics in metrics_by_mode.items():
        if not name.startswith("full_except_"):
            continue
        results[name] = {
            scope: {
                metric_name: float(metric_value - full_metrics[scope][metric_name])
                for metric_name, metric_value in scope_metrics.items()
            }
            for scope, scope_metrics in metrics.items()
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vq_share_gate_v1")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_name", default="vq_branch_probe.log")
    parser.add_argument("--report_name", default="vq_branch_probe_seed2024.json")
    args = parser.parse_args()

    pm = ParamManager(ProbeConfig)
    cfg: ProbeConfig = pm.build(
        experiment_name=ProbeConfig.model,
        model=args.model,
        seed=args.seed,
        max_epochs=args.max_epochs,
        device=args.device,
        log_name=args.log_name,
        report_name=args.report_name,
    )

    manager, train_path, valid_path, test_path, top_5_list, tab_domain_map = build_data(cfg)
    domain_to_tab_map = {domain_id: raw_tab for raw_tab, domain_id in tab_domain_map.items()}
    num_domains = 5
    model_kwargs = MODEL_KWARGS.get(cfg.model, {})
    model = build_model(manager, num_domains, cfg.model, **model_kwargs)

    train_loader = ParquetStreamDataset(train_path, manager, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    eval_loader = ParquetStreamDataset(test_path, manager, batch_size=4096 * 2, shuffle=False, drop_last=False)
    valid_loader_for_reference = ParquetStreamDataset(valid_path, manager, batch_size=4096 * 2, shuffle=False, drop_last=False)

    evaluator_manager = build_evaluator_manager(
        manager=manager,
        log_path=WORKSPACE / cfg.dataset_name / cfg.log_name,
        title=f"{cfg.model}_branch_probe",
        num_domains=num_domains,
    )
    params = split_params_by_decay(model.named_parameters(), weight_decay=cfg.weight_decay, no_decay_keywords=["embedding"])
    optimizer = torch.optim.Adam(params, lr=1e-3)
    trainer = KuairandTrainer(
        model=model,
        optimizer=optimizer,
        manager=manager,
        loaders=TrainerDataLoaders(train=train_loader, valid=eval_loader, test=valid_loader_for_reference),
        components=TrainerComponents(evaluator_manager=evaluator_manager, early_stepper=EarlyStopper()),
        cfg=cfg,
    )

    report = {
        "model": cfg.model,
        "seed": cfg.seed,
        "max_epochs": cfg.max_epochs,
        "top5_tabs_in_order": top_5_list,
        "tab_to_domain_vocab_order": tab_domain_map,
        "domain_to_tab_vocab_order": domain_to_tab_map,
        "note": "Current training entry evaluates on 20220508 test split via valid_loader; 20220507 is built but not used in trainer.run().",
        "epochs": [],
    }

    for _ in range(cfg.max_epochs):
        trainer.train_epoch()
        full_metrics = trainer.evaluate_epoch()
        branch_metrics, domain_code_rows, domain_debug_summary = evaluate_modes(
            model=trainer.model,
            manager=trainer.manager,
            data_loader=eval_loader,
            num_domains=num_domains,
            device=cfg.device,
            domain_to_tab_map=domain_to_tab_map,
        )
        knowledge = summarize_code_knowledge(
            domain_code_rows=domain_code_rows,
            num_domains=num_domains,
            codebook_size=getattr(model, "codebook_size", 0),
        )
        contrib = model.contribution_state() if hasattr(model, "contribution_state") else {}

        epoch_report = {
            "epoch": trainer.epoch,
            "full_metrics": full_metrics,
            "branch_metrics": branch_metrics,
            "branch_deltas_vs_full": metric_deltas(full_metrics, branch_metrics),
            "shared_override_deltas_vs_full": summarize_shared_override_effects(branch_metrics),
            "contribution_state": contrib,
            "domain_debug_summary": domain_debug_summary,
            "code_knowledge": knowledge,
        }
        report["epochs"].append(epoch_report)
        print(json.dumps(epoch_report, ensure_ascii=False, indent=2))

    report_path = WORKSPACE / cfg.dataset_name / cfg.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
