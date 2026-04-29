from typing import Iterable, Optional

import torch
from torch import nn

from betterbole.core.train.context import TrainContext
from betterbole.emb import SchemaManager
from betterbole.emb.schema import EmbType
from betterbole.models.msr.automtl.supernet import AutoMTLSuperNet
from betterbole.models.msr.base import MSRModel
from betterbole.models.msr.components import DomainTowerHead


class AutoMTLModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            num_experts: int = 4,
            num_expert_layers: int = 2,
            expert_num_layers: int = 3,
            expert_in_features: int = 64,
            expert_out_features: int = 64,
            expert_candidate_ops: Iterable[str] = (
                "Identity",
                "MLP-16",
                "MLP-32",
                "MLP-64",
                "MLP-128",
                "MLP-256",
                "MLP-512",
                "MLP-1024",
            ),
            dropout_rate: float = 0.2,
            tower_hidden_dims: Iterable[int] = (64,),
            tower_dropout_rate: float = 0.2,
            include_domain_in_input: bool = False,
            warmup_epochs: int = 2,
            search_epochs: int = 3,
            discretize_ops: int = 10,
            arch_lr: float = 1e-4,
            arch_weight_decay: float = 0.0,
            arch_init_type: str = "normal",
            arch_init_ratio: float = 1e-3,
            freeze_arch_after_export: bool = True,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field

        self.include_domain_in_input = bool(include_domain_in_input)
        self.input_view = (
            self.omni_embedding.whole if self.include_domain_in_input
            else self.omni_embedding.whole_without_domain
        )

        excluded_fields = set()
        if not self.include_domain_in_input:
            excluded_fields.update(manager.domain_fields)

        self.feature_settings = []
        self.sparse_field_names = []
        self.dense_field_names = []
        self.embedding_dim = None
        for setting in manager.settings:
            if setting.field_name in excluded_fields:
                continue
            if setting.emb_type == EmbType.SPARSE_SEQ:
                raise ValueError(
                    f"AutoMTLModel does not support sequence field '{setting.field_name}' yet."
                )
            self.feature_settings.append(setting)
            if setting.emb_type == EmbType.DENSE:
                self.dense_field_names.append(setting.field_name)
            else:
                self.sparse_field_names.append(setting.field_name)
                if self.embedding_dim is None:
                    self.embedding_dim = int(setting.embedding_dim)
                elif int(setting.embedding_dim) != self.embedding_dim:
                    raise ValueError(
                        "AutoMTLModel requires all sparse-like fields to share the same embedding_dim. "
                        f"Found {self.embedding_dim} and {setting.embedding_dim}."
                    )

        if not self.sparse_field_names:
            raise ValueError("AutoMTLModel requires at least one sparse-like field.")
        if self.embedding_dim is None:
            raise ValueError("AutoMTLModel failed to infer sparse embedding_dim.")

        self.backbone = AutoMTLSuperNet(
            embedding_dim=self.embedding_dim,
            num_sparse_fields=len(self.sparse_field_names),
            num_dense_fields=len(self.dense_field_names),
            num_domains=num_domains,
            num_experts=num_experts,
            num_expert_layers=num_expert_layers,
            expert_num_layers=expert_num_layers,
            expert_in_features=expert_in_features,
            expert_out_features=expert_out_features,
            dropout_rate=dropout_rate,
            expert_candidate_ops=expert_candidate_ops,
        )
        self.backbone.init_arch_params(init_type=arch_init_type, init_ratio=arch_init_ratio)
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.backbone.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
        )

        self.warmup_epochs = int(warmup_epochs)
        self.search_epochs = int(search_epochs)
        self.discretize_ops = int(discretize_ops)
        self.arch_lr = float(arch_lr)
        self.arch_weight_decay = float(arch_weight_decay)
        self.freeze_arch_after_export = bool(freeze_arch_after_export)

        self.arch_optimizer: Optional[torch.optim.Optimizer] = None
        self.stage = "warmup"
        self.current_epoch = -1
        self.exported = False
        self.best_metric = None

    @property
    def input_dim(self) -> int:
        return self.backbone.input_dim

    @property
    def exported_arch(self) -> dict:
        return self.backbone.exported_arch

    def architecture_parameters(self):
        yield from self.backbone.architecture_parameters()

    def alpha_parameters(self):
        yield from self.backbone.alpha_parameters()

    def beta_parameters(self):
        yield from self.backbone.beta_parameters()

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if "alpha" not in name and "beta" not in name:
                yield param

    def _ensure_arch_optimizer(self) -> Optional[torch.optim.Optimizer]:
        if self.arch_optimizer is None:
            params = list(self.architecture_parameters())
            if params:
                self.arch_optimizer = torch.optim.Adam(
                    params,
                    lr=self.arch_lr,
                    weight_decay=self.arch_weight_decay,
                )
        return self.arch_optimizer

    def _freeze_arch_parameters(self) -> None:
        for param in self.architecture_parameters():
            param.requires_grad = False

    def _clear_arch_grads(self) -> None:
        for param in self.architecture_parameters():
            param.grad = None

    def _resolve_stage(self, epoch: int) -> str:
        if epoch < self.warmup_epochs:
            return "warmup"
        if epoch < self.warmup_epochs + self.search_epochs:
            return "search"
        return "finetune"

    def _is_last_search_epoch(self, epoch: int) -> bool:
        return epoch == (self.warmup_epochs + self.search_epochs - 1)

    def encode_features(
            self,
            interaction,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        emb_by_name = self.input_view(interaction, split_by="name")
        sparse_embs = []
        for field_name in self.sparse_field_names:
            emb = emb_by_name[field_name]
            if emb.dim() != 2:
                raise ValueError(
                    f"AutoMTLModel expects field '{field_name}' to produce [B, E], got {tuple(emb.shape)}."
                )
            sparse_embs.append(emb)
        dense_features = None
        if self.dense_field_names:
            dense_tensors = []
            for field_name in self.dense_field_names:
                value = emb_by_name[field_name]
                if value.dim() == 1:
                    value = value.unsqueeze(-1)
                dense_tensors.append(value.float())
            dense_features = torch.cat(dense_tensors, dim=1)
        sparse_tensor = torch.stack(sparse_embs, dim=1)
        return sparse_tensor, dense_features, interaction[self.DOMAIN].long()

    def forward(
            self,
            sparse_embs: torch.Tensor,
            dense_features: Optional[torch.Tensor],
            domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.backbone(sparse_embs, dense_features, domain_ids)
        return self.head(representation, domain_ids)

    def predict(self, interaction):
        sparse_embs, dense_features, domain_ids = self.encode_features(interaction)
        return self.forward(sparse_embs, dense_features, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        arch_optimizer = self._ensure_arch_optimizer()
        if arch_optimizer is not None:
            arch_optimizer.zero_grad(set_to_none=True)
        loss = self.calculate_loss(batch_interaction)
        loss.backward()
        if self.stage == "search" and not self.exported and arch_optimizer is not None:
            arch_optimizer.step()
        self._clear_arch_grads()
        ctx.optimizer.step()
        return float(loss.item())

    def on_train_epoch_start(self, ctx: TrainContext) -> None:
        self.current_epoch = int(ctx.epoch)
        self.stage = self._resolve_stage(self.current_epoch)
        if self.stage != "warmup":
            self.backbone.set_chosen_op_active()
        if self.stage == "finetune" and self.freeze_arch_after_export:
            self._freeze_arch_parameters()

    def on_train_epoch_end(self, ctx: TrainContext) -> None:
        if self.stage != "search" or self.exported:
            return
        for _ in range(self.discretize_ops):
            if self.backbone.discretize_one_op() is None:
                break
        if self._is_last_search_epoch(self.current_epoch):
            self.backbone.export_architecture()
            self.exported = True
            if self.freeze_arch_after_export:
                self._freeze_arch_parameters()

    def on_eval_epoch_end(self, metrics: Optional[dict], ctx: TrainContext) -> None:
        if metrics:
            self.best_metric = metrics
