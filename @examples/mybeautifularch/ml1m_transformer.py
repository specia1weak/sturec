import numpy as np
import torch
from torch.nn import BatchNorm1d
from torch.utils.data import DataLoader

from src.betterbole.data.dataset import ParquetStreamDataset
from src.betterbole.emb.schema import SparseEmbSetting, QuantileEmbSetting, \
    SparseSetEmbSetting, IdSeqEmbSetting, MinMaxDenseSetting
from src.betterbole.emb import SchemaManager
import polars as pl
from src.betterbole.enum_type import FeatureSource
from src.betterbole.evaluate.evaluator import Evaluator, LogDecorator
from src.betterbole.interaction import Interaction
from src.betterbole.plutils import extract_history_items, extract_history_dict
from src.betterbole.sample import PolarsUISampler

from torch import nn

from src.model.utils.general import MLP
from src.model.utils.transfomer import TransformerEncoder
from src.utils import change_root_workdir

change_root_workdir()
Backbone = MLP
EMB_DIM = 16

class TransformerModel(nn.Module):
    """
    基于 Transformer 的推荐模型（二分支架构）

    架构说明：
    1. 从 interaction 中提取 user_id 和 item_id 的 embedding
    2. 组织成 [B, 2, d_model] 的特征序列
    3. 通过双层 Transformer (num_heads=2) 进行特征交互编码
    4. 分别提取 Transformer 输出的两个位置 [0] 和 [1]
    5. 分别通过 User 分类头和 Item 分类头
    6. 两个输出相加得到最终 CTR logits

    特征序列：
    - 位置 0: user_id [B, d_model]
    - 位置 1: item_id [B, d_model]
    """

    def __init__(self, schema_manager: SchemaManager):
        super(TransformerModel, self).__init__()
        manager = schema_manager
        self.user_profile_encoder = UserProfileEncoder(
            manager.settings, manager.work_dir / manager.USER_PROFILE_NAME
        )
        self.item_profile_encoder = ItemProfileEncoder(
            manager.settings, manager.work_dir / manager.ITEM_PROFILE_NAME
        )
        self.inter_emb_layer = InterSideEmb(manager.settings)

        self.manager = manager
        self.LABEL = manager.label_field
        self.DOMAIN = manager.domain_field

        # ========== 参数配置 ==========
        self.d_model = EMB_DIM  # 特征维度（每个特征的embedding dim）
        self.num_heads = 2       # Transformer 注意力头数
        self.num_layers = 1      # Transformer 编码器层数
        self.d_ff = EMB_DIM * 4  # 前馈网络隐层维度

        # TODO: 需要查看 emblayer.py 源码，理解：
        # 1. 如何从 user_profile_encoder/item_profile_encoder 获取各个特征
        # 2. 每个特征的具体维度和个数
        # 3. 如何将这些特征重新组织成序列形式

        # ========== Transformer 编码器 ==========
        self.transformer_encoder = TransformerEncoder(
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=0.1
        )

        # ========== 分类头：User 分支 ==========
        # 从 user_id token 的输出进行预测
        self.user_classification_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # ========== 分类头：Item 分支 ==========
        # 从 item_id token 的输出进行预测
        self.item_classification_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def concat_embed_input_fields(self, interaction):
        """
        从 interaction 中提取各部分特征的 embedding，返回按特征拆分的字典

        使用 split_by="name" 参数，使得 ProfileEncoder 返回以特征名为 key 的字典
        """
        uid = interaction[self.manager.uid_field]
        iid = interaction[self.manager.iid_field]

        # split_by="name" 会返回 {"user_id": [...], "age": [...], ...} 的字典
        user_emb_dict = self.user_profile_encoder.forward(uid, split_by="name")
        item_emb_dict = self.item_profile_encoder.forward(iid, split_by="name")
        inter_emb_dict = self.inter_emb_layer.forward(interaction, split_by="name")

        return user_emb_dict, item_emb_dict, inter_emb_dict

    def _prepare_feature_sequence(self, user_emb_dict, item_emb_dict, _inter_emb_dict):
        """
        将 user_id 和 item_id 组织成特征序列

        返回形状：[B, 2, d_model]
        - 位置 0: user_id embedding [B, d_model]
        - 位置 1: item_id embedding [B, d_model]

        Args:
            user_emb_dict: {"user_id": [B, 8], ...}
            item_emb_dict: {"item_id": [B, 8], ...}
            _inter_emb_dict: 暂未使用

        Returns:
            feature_sequence: [B, 2, d_model]
        """
        feature_emb_list = []

        # 先添加 user_id
        if isinstance(user_emb_dict, dict) and "user_id" in user_emb_dict:
            emb = user_emb_dict["user_id"]  # [B, d_model]
            feature_emb_list.append(emb.unsqueeze(1))  # [B, 1, d_model]

        # 再添加 item_id
        if isinstance(item_emb_dict, dict) and "movie_id" in item_emb_dict:
            emb = item_emb_dict["movie_id"]  # [B, d_model]
            feature_emb_list.append(emb.unsqueeze(1))  # [B, 1, d_model]

        if len(feature_emb_list) != 2:
            raise ValueError("Must have both user_id and item_id embeddings")

        # 沿序列维度 (dim=1) 拼接
        # 结果: [B, 2, d_model]
        feature_sequence = torch.cat(feature_emb_list, dim=1)

        return feature_sequence

    def forward(self, feature_sequence):
        """
        前向传播

        Args:
            feature_sequence: [B, 2, d_model] - 特征序列 (user_id, item_id)

        Returns:
            logits: [B] - CTR 预测的 logits（两个分类头的输出相加）
        """
        # 1. Transformer 编码
        # 注意：不需要 mask，所有位置都可以互相注意
        encoded_sequence = self.transformer_encoder(feature_sequence)  # [B, 2, d_model]

        # 2. 分别提取 user_id 和 item_id 的编码输出
        # 位置 0：user_id，位置 1：item_id
        user_output = encoded_sequence[:, 0, :]  # [B, d_model]
        item_output = encoded_sequence[:, 1, :]  # [B, d_model]

        # 3. 分别通过两个分类头
        user_logits = self.user_classification_head(user_output).squeeze(-1)  # [B]
        item_logits = self.item_classification_head(item_output).squeeze(-1)  # [B]

        # 4. 合并两个输出
        logits = user_logits + item_logits  # [B]

        return logits

    def predict(self, interaction):
        """
        预测接口
        """
        # 1. 获取各部分 embedding 字典
        user_emb_dict, item_emb_dict, inter_emb_dict = \
            self.concat_embed_input_fields(interaction)

        # 2. 组织成特征序列 [B, num_features, d_model]
        feature_sequence = self._prepare_feature_sequence(
            user_emb_dict, item_emb_dict, inter_emb_dict
        )

        # 3. 前向传播获得 logits
        logits = self.forward(feature_sequence)

        # 4. 转换为概率
        predictions = torch.sigmoid(logits)

        return predictions

    def calculate_loss(self, interaction):
        """
        计算损失
        """
        label_tensor = interaction[self.LABEL].float()

        # 获取预测（logits，还没有 sigmoid）
        user_emb_dict, item_emb_dict, inter_emb_dict = \
            self.concat_embed_input_fields(interaction)

        feature_sequence = self._prepare_feature_sequence(
            user_emb_dict, item_emb_dict, inter_emb_dict
        )

        logits = self.forward(feature_sequence)

        # 二分类交叉熵损失
        loss = nn.functional.binary_cross_entropy_with_logits(logits, label_tensor)

        return loss


if __name__ == '__main__':
    from src.utils.task_chain import auto_queue
    auto_queue()
    device = "cuda"

    user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, EMB_DIM) # 显存太低没法拉高dim
    item_setting = SparseEmbSetting("movie_id", FeatureSource.ITEM_ID, EMB_DIM) # 显存太低没法拉高dim
    settings_list = [
        user_setting,
        item_setting,
        SparseEmbSetting("age", FeatureSource.USER, EMB_DIM),
        SparseEmbSetting("gender", FeatureSource.USER, EMB_DIM),
        SparseEmbSetting("occupation", FeatureSource.USER, EMB_DIM),

        SparseSetEmbSetting("genres", FeatureSource.ITEM, EMB_DIM),
        # IdSeqEmbSetting("history", "history_len", target_setting=item_setting, max_len=50)
    ]

    manager = SchemaManager(settings_list, "movielens-workdir", time_field="timestamp", label_fields="label", domain_fields="gender")
    from src.dataset.movielens import MovieLensDataset
    user_lf = pl.from_pandas(MovieLensDataset.USER_FEATURES_DF).lazy()
    item_lf = pl.from_pandas(MovieLensDataset.ITEM_FEATURES_DF).lazy()
    inter_lf = pl.from_pandas(MovieLensDataset.INTERACTION_DF).lazy()
    whole_lf: pl.LazyFrame = inter_lf.join(item_lf, on="movie_id", how="left").join(user_lf, on="user_id", how="left")
    whole_lf = whole_lf.with_columns(
        pl.col("genres").str.split("|"),
        (pl.col("rating") >= 4).cast(pl.Int8).alias("label")
    )
    whole_lf = whole_lf.with_columns(
        pl.when(pl.col("age") < 25).then(0)
        .when(pl.col("age") < 35).then(1)
        .otherwise(2)
        .alias("domain_id")
    )

    max_seq_len = 50
    whole_lf = extract_history_items(whole_lf, max_seq_len=50,
                                     user_col="user_id",
                                     time_col="timestamp",
                                     item_col="movie_id",
                                     seq_col="history",
                                     seq_len_col="history_len")

    transformed_lf = manager.prepare_data(whole_lf)
    manager.generate_profiles(transformed_lf)
    train_path, valid_path, _ = manager.split_dataset(transformed_lf, strategy="random_ratio")
    # train_path, valid_path, _ = manager.save_as_dataset(train_lf, valid_lf)
    print("架构编译成功，可供调用。")
    evaluator = LogDecorator(Evaluator("AUC"),
                             save_path=manager.work_dir / "logs4backbone.log", title=f"DIM={EMB_DIM}")
    from src.betterbole.emb.emblayer import InterSideEmb, \
        ItemProfileEncoder, UserProfileEncoder

    ps_dataset = ParquetStreamDataset(train_path, manager.fields())
    ps_valid = ParquetStreamDataset(valid_path, manager.fields(), batch_size=4096 * 2, shuffle_and_drop_last=False)
    ps_dataloader = DataLoader(
        ps_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=True
    )

    model = TransformerModel(manager).to(device)


    from src.utils.time import CudaNamedTimer
    ntr = CudaNamedTimer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(50):
        total_loss = 0.
        batch_count = 0
        model.train()
        # if epoch == 25:
        # #     model.train_gate()
        #     print("启动门控")
        with ntr("epoch"):
            for batch_interaction in ps_dataset:
                # 1. 采样：获取负样本
                # 确保传入的是 numpy，且返回后立刻转为 long 并送到指定的 device
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
                    # print(model.gate_monitor.get_window_stats())
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