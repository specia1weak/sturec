from pathlib import Path

import torch

from betterbole.data.dataset import ParquetStreamDataset
from betterbole.emb.schema import SparseEmbSetting
from betterbole.emb import SchemaManager
import polars as pl
from betterbole.core.enum_type import FeatureSource
from betterbole.evaluate.evaluator import Evaluator, LogDecorator

from torch import nn
import torch.nn.functional as F

from betterbole.models.backbone.star import STAR
from betterbole.models.backbone.shabtm import SharedBottomLess
from betterbole.models.utils.general import MLP
from betterbole.utils import change_root_workdir
from betterbole.utils import ExplicitFeatureMonitor
from betterbole.utils import create_optimizer_groups
from split_tool import generate_hybrid_splits_polars
change_root_workdir()
Backbone = STAR
EMB_DIM = 16

class DiffExperimentDataset:
    BASE_DIR =  Path("raw/diffmsr")
    AMAZON_LARGE = BASE_DIR / "amazon_time.csv"
    AMAZON_SMALL = BASE_DIR / "amazonsmall.csv"
    DOUBAN = BASE_DIR / "douban3.csv"

class SimplePLE(nn.Module):
    def __init__(self, schema_manager: SchemaManager):
        super(SimplePLE, self).__init__()
        manager = schema_manager
        self.user_emb_layer = UserSideEmb(manager.settings)
        self.item_emb_layer = ItemSideEmb(manager.settings)

        self.manager = manager
        self.input_dim = manager.source2emb_dim(FeatureSource.USER_ID, FeatureSource.ITEM_ID, )
        print(f"输入总维度{self.input_dim}")
        self.DOMAIN = manager.domain_field
        num_domains = manager.get_setting(self.DOMAIN).vocab_size
        # self.ple = PLE(self.input_dim, num_domains, expert_dims=(64, 32), num_sha_experts=1, num_spe_experts=1)
        # self.ple = STAR(self.input_dim, num_domains)
        self.ple = SharedBottomLess(self.input_dim, num_domains)
        self.LABEL = manager.label_field

    def concat_embed_input_fields(self, interaction):
        user_emb = self.user_emb_layer.forward(interaction)
        item_emb = self.item_emb_layer.forward(interaction)
        return torch.cat([user_emb, item_emb], dim=-1)

    def forward(self, x, domain_ids):
        return self.ple.forward(x, domain_ids)

    def predict(self, interaction):
        x = self.concat_embed_input_fields(interaction)
        domain_ids = interaction[self.DOMAIN] - 1
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        return torch.sigmoid(final_logits)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        domain_ids = interaction[self.DOMAIN] - 1
        x = self.concat_embed_input_fields(interaction)
        x = torch.flatten(x, start_dim=1)
        final_logits = self.forward(x, domain_ids)
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels)
        return loss

class SpecialModel(nn.Module):
    def __init__(self, schema_manager: SchemaManager,):
        super(SpecialModel, self).__init__()
        manager = schema_manager
        self.user_profile_encoder = UserSideEmb(manager.settings)
        self.item_profile_encoder = ItemSideEmb(manager.settings)

        self.manager = manager
        self.LABEL = manager.label_field
        self.DOMAIN = manager.domain_field

        self.whole_input_dim = manager.source2emb_dim(FeatureSource.USER_ID, FeatureSource.ITEM_ID)
        self.item_id_input_dim = manager.source2emb_dim(FeatureSource.ITEM_ID)
        self.user_id_input_dim = manager.source2emb_dim(FeatureSource.USER_ID)

        self.specific_expert = MLP(self.whole_input_dim, 64, 1)
        self.user_id_expert = MLP(self.user_id_input_dim, 64, 1)
        self.item_id_expert = MLP(self.item_id_input_dim, 64, 1)

        self.cross_expert = MLP(self.item_id_input_dim, 64, 1)
        self.add_expert = MLP(self.item_id_input_dim, 64, 1)

        # self.batch_norm = BatchNorm1d(6)
        self.gate_expert = MLP(self.whole_input_dim, 64, 3)

        self.gate_monitor = ExplicitFeatureMonitor()
        self._train_gate = True
        self._last_gates = None

    def train_gate(self):
        self._train_gate = True
    def freeze_gate(self):
        self._train_gate = False

    def _drop_none(self, list):
        return [emb for emb in list if emb is not None]

    def concat_embed_input_fields(self, interaction):
        user_emb_dict = self.user_profile_encoder.forward(interaction, split_by="source")
        user_id_emb = user_emb_dict[FeatureSource.USER_ID]

        item_emb_dict = self.item_profile_encoder.forward(interaction, split_by="source")
        item_id_emb = item_emb_dict[FeatureSource.ITEM_ID]

        specific_emb = torch.cat(self._drop_none([user_id_emb, item_id_emb]), dim=-1)

        return specific_emb, user_id_emb, item_id_emb

    def forward(self, spe, uid, iid):
        uid_logits = self.user_id_expert(uid)
        iid_logits = self.item_id_expert(iid)
        spe_logits = self.specific_expert(spe)
        gates = self.gate_expert(spe)
        if not self._train_gate:
            gates = gates.detach()

        all_logits = torch.cat([uid_logits, iid_logits, spe_logits], dim=-1)

        # 选择mask掉谁
        mask = torch.zeros_like(gates, dtype=torch.bool)
        # mask[:, -1] = True
        # mask[:, -2] = False
        gates = gates.masked_fill(mask, float('-inf'))
        # 结束

        gates = F.softmax(gates, dim=-1)
        self.gate_monitor.record("three_expert", gates)
        self.gate_monitor.record("ctr_contribute", gates * all_logits)
        final_logits = torch.sum(gates * all_logits, dim=-1)  # B
        self._last_gates = gates
        return final_logits

    def predict(self, interaction):
        spe, uid, iid = self.concat_embed_input_fields(interaction)
        return self.forward(spe, uid, iid)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        final_logits = self.predict(interaction)
        gate_loss = -0.02 * torch.var(self._last_gates, dim=0).mean()
        loss = nn.functional.binary_cross_entropy_with_logits(final_logits, labels) + gate_loss
        return loss

if __name__ == '__main__':
    from betterbole.utils import auto_queue
    auto_queue()
    device = "cuda"
    user_setting = SparseEmbSetting("user", FeatureSource.USER_ID, EMB_DIM, min_freq=1, use_oov=True)
    item_setting = SparseEmbSetting("item", FeatureSource.ITEM_ID, EMB_DIM, min_freq=1, use_oov=True)
    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("domain_indicator", FeatureSource.INTERACTION, 16, padding_zero=True, use_oov=False)
    ]
    manager = SchemaManager(settings_list, "diffmsr-amazonL-workdir", label_fields="label", domain_fields="domain_indicator", time_field="time")

    whole_lf = pl.scan_csv(DiffExperimentDataset.AMAZON_LARGE)
    train_samples_lf, train_ple_lf, val_ple_lf, test_ple_lf = generate_hybrid_splits_polars(whole_lf)

    # 2. 额外添加的统计输出代码
    print("=" * 50)
    print("DATASET SPLIT STATISTICS (Row Counts)")
    print("=" * 50)

    # 使用高效的 pl.len() 触发图计算获取行数
    num_train_seq = train_samples_lf.select(pl.len()).collect().item()
    num_train_ple = train_ple_lf.select(pl.len()).collect().item()
    num_val_ple = val_ple_lf.select(pl.len()).collect().item()
    num_test_ple = test_ple_lf.select(pl.len()).collect().item()

    print(f"train_samples (Pos only, Seq): {num_train_seq}")
    print(f"train_ple     (Pos + Neg):     {num_train_ple}")
    print(f"val_ple       (Pos + Neg):     {num_val_ple}")
    print(f"test_ple      (Pos + Neg):     {num_test_ple}")
    print("=" * 50)

    manager.fit(train_ple_lf)
    train_lf = manager.transform(train_ple_lf)
    valid_lf = manager.transform(val_ple_lf)
    test_lf = manager.transform(test_ple_lf)

    train_lf = train_lf.sort(by="time")

    train_path, valid_path, test_path = manager.save_as_dataset(train_lf, valid_lf, test_lf)
    print("架构编译成功，可供调用。")
    evaluator = LogDecorator(Evaluator("AUC"),
                             save_path=manager.work_dir / "logs4backbone.log", title=f"DIM={EMB_DIM},{Backbone.__name__}")
    from betterbole.emb.emblayer import UserSideEmb, ItemSideEmb

    ps_dataset = ParquetStreamDataset(train_path, manager.fields()) # 更少的读取
    ps_valid = ParquetStreamDataset(test_path, manager.fields(), batch_size=4096 * 2, shuffle=False) # 更少的读取
    model = SimplePLE(manager).to(device)

    from betterbole.utils import CudaNamedTimer
    ntr = CudaNamedTimer()
    named_parameters = create_optimizer_groups(model, weight_decay=1e-5, no_decay_keywords=["embedding"]) # 对Embedding施加wd反而更好
    optimizer = torch.optim.Adam(named_parameters, lr=1e-3)
    for epoch in range(50):
        total_loss = 0.
        batch_count = 0
        model.train()
        with ntr("epoch"):
            for batch_interaction in ps_dataset:
                with ntr("prepare"):
                    batch_interaction = batch_interaction.to(device)

                # 4. 前向传播与优化
                with ntr("train"):
                    optimizer.zero_grad()
                    loss = model.calculate_loss(batch_interaction)
                    loss.backward()
                    optimizer.step()

                # 5. 累加损失（必须用 .item() 否则会 OOM）
                with ntr("add_loss"):
                    total_loss += loss.item()

                batch_count += 1
                if batch_count % 100 == 0:
                    ntr.report()
                    if hasattr(model, "gate_monitor"):
                        print(model.gate_monitor.get_window_stats())
                    print(f"Epoch {epoch}, Batch {batch_count}, Current Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            for batch_interaction in ps_valid:
                # 1. 先整体推到 GPU
                batch_interaction = batch_interaction.to(device)

                # 2. 干净地取出 GPU 上的特征
                uids = batch_interaction[manager.uid_field]
                labels = batch_interaction[manager.label_field]

                # 3. 预测并打分
                scores = model.predict(batch_interaction)
                evaluator.collect_pointwise(uids, labels, batch_preds_1d=scores)

        metrics_result = evaluator.summary(epoch)
        print(f"Validation Metrics: {metrics_result}")
        evaluator.clear()
        ntr.report()
        # print(f"=== Epoch {epoch} Done, Average Loss: {total_loss / batch_count:.4f} ===")