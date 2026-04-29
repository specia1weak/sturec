# ==========================================
# 测试用例: test_pipeline_seq_group.py
# 完美挑战 3D序列特征 (Tags_Seq) 与 2D序列 (Item_Seq) 融合！
# ==========================================
import os
import shutil
from typing import List

import polars as pl
import torch
import numpy as np

# 请确保 import 路径与你本地一致
from betterbole.core.enum_type import FeatureSource
from betterbole.emb.schema import SparseEmbSetting, SparseSetEmbSetting, MinMaxDenseSetting, QuantileEmbSetting, \
    SeqGroupEmbSetting
from betterbole.emb.manager import SchemaManager
from betterbole.emb.emblayer import OmniEmbLayer
from betterbole.utils.sequential import extract_history_sequences


# ==========================================
# 🛠️ 核心组件补丁区 (未来你可以把这段移回 schema.py)
# ==========================================
class PatchedSeqGroupEmbSetting(SeqGroupEmbSetting):
    """解决原版无法处理 3D 嵌套列表 (tags_seq) 的终极补丁"""

    def get_transform_expr(self) -> List[pl.Expr]:
        exprs = []
        for seq_col, target in self.target_dict.items():
            keys = pl.Series(list(target.vocab.keys()), dtype=pl.Utf8)
            vals = pl.Series(list(target.vocab.values()), dtype=pl.UInt32)

            if isinstance(target, SparseSetEmbSetting):
                # 【3D 降维打击】处理 List[List[str]]
                expr = pl.col(seq_col).fill_null([])
                if self.is_string_format:
                    expr = expr.list.eval(pl.element().str.split(self.separator))

                mapped_expr = (
                    expr.list.eval(  # 外层循环: 时间步
                        pl.element().list.eval(  # 内层循环: 标签集合
                            pl.element()
                            .cast(pl.Utf8)
                            .replace_strict(old=keys, new=vals, default=pl.lit(target.oov_idx, dtype=pl.UInt32))
                            .cast(pl.UInt32)
                        ).list.tail(target.max_len)  # 截断集合维度
                    )
                    .list.tail(self.max_len)  # 截断时间步维度
                    .alias(seq_col)
                )
            else:
                # 【2D 常规处理】处理 List[str]
                expr = pl.col(seq_col).fill_null([])
                mapped_expr = (
                    expr.list.eval(
                        pl.element()
                        .cast(pl.Utf8)
                        .replace_strict(old=keys, new=vals, default=pl.lit(target.oov_idx, dtype=pl.UInt32))
                        .cast(pl.UInt32)
                    )
                    .list.tail(self.max_len)
                    .alias(seq_col)
                )
            exprs.append(mapped_expr)
        return exprs

    def compute_tensor(self, interaction, emb_modules):
        group_embs = []
        for seq_col, target in self.target_dict.items():
            # ✨ 极其优雅的转接：伪造一个单特征字典，让 SparseSet 自己去做 Pooling！
            mock_interaction = {target.field_name: interaction[seq_col]}
            emb = target.compute_tensor(mock_interaction, emb_modules)
            group_embs.append(emb)

        if self.combiner == "concat":
            return torch.cat(group_embs, dim=-1)
        elif self.combiner == "sum":
            return torch.sum(torch.stack(group_embs, dim=0), dim=0)


# ==========================================
# 🛠️ DataLoader Padding 工具 (处理变长 Tensor)
# ==========================================
def pad_list_column(sequences, max_len, pad_val=0):
    """用于 2D 序列 (如 item_id_seq)"""
    padded = np.full((len(sequences), max_len), pad_val, dtype=np.int64)
    for i, seq in enumerate(sequences):
        if seq is None or (isinstance(seq, float) and np.isnan(seq)):
            continue
        valid_len = min(len(seq), max_len)
        padded[i, :valid_len] = list(seq)[:valid_len]
    return padded


def pad_nested_list_column(sequences, max_seq_len, max_tag_len, pad_val=0):
    """用于 3D 嵌套序列 (如 tags_seq) -> [Batch, SeqLen, TagLen]"""
    padded = np.full((len(sequences), max_seq_len, max_tag_len), pad_val, dtype=np.int64)
    for i, seq in enumerate(sequences):
        if seq is None or (isinstance(seq, float) and np.isnan(seq)):
            continue
        valid_seq_len = min(len(seq), max_seq_len)
        for j in range(valid_seq_len):
            tags = seq[j]
            if tags is None: continue
            valid_tag_len = min(len(tags), max_tag_len)
            padded[i, j, :valid_tag_len] = list(tags)[:valid_tag_len]
    return padded


# ==========================================
# 🚀 主流程测试
# ==========================================
def test_pipeline():
    work_dir = "./test_work_dir"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

    print("==================================================")
    print("🚀 阶段一：数据预处理 (带 SeqGroup 的终极测试)")
    print("==================================================")

    raw_df = pl.DataFrame({
        "user_id": ["u1", "u2", "u1", "u3"],
        "item_id": ["i1", "i2", "i3", "i1"],
        "tags": ["action,sci-fi", "comedy", "action,romance", "horror"],
        "time": [2, 6, 7, 3],
    }).lazy()

    # 1. 抽取序列历史 (映射 item_id 和 tags 两个特征)
    raw_df = extract_history_sequences(
        raw_df, max_seq_len=10, user_col="user_id", time_col="time",
        feature_mapping={
            "item_id": "item_id_seq",
            "tags": "tags_seq"  # ✨ 新增 tags 历史序列
        },
        seq_len_col="seq_len"
    )

    # 2. 声明基础图纸
    item_setting = SparseEmbSetting("item_id", FeatureSource.ITEM, embedding_dim=8)
    tags_setting = SparseSetEmbSetting("tags", FeatureSource.ITEM, embedding_dim=12, max_len=3, is_string_format=True,
                                       separator=",")

    # 3. 声明巨无霸 Group 图纸
    seq_group = PatchedSeqGroupEmbSetting(
        group_name="hist_seq_group",
        seq_len_field_name="seq_len",
        target_dict={
            "item_id_seq": item_setting,
            "tags_seq": tags_setting
        },
        max_len=5,  # 序列截断长度
        is_string_format=True,
        separator=","
    )

    settings_list = [
        SparseEmbSetting("user_id", FeatureSource.USER, embedding_dim=8),
        item_setting,
        tags_setting,
        seq_group
    ]

    manager_v1 = SchemaManager(settings_list, work_dir=work_dir)
    manager_v1.prepare_data(raw_df, output_dir=work_dir, redo=True)
    print("[+] 阶段一完成！字典与数据已固化至硬盘。\n")

    print("==================================================")
    print("🚀 阶段二：加载模型并测试 3D 融合前向传播")
    print("==================================================")

    # 重新声明一样的图纸并加载状态
    manager_v2 = SchemaManager(settings_list, work_dir=work_dir)
    manager_v2.load_schema()
    omni_layer = OmniEmbLayer(manager=manager_v2)

    parquet_path = f"{work_dir}/{manager_v1.WHOLE_DATA_NAME}"
    batch_df = pl.read_parquet(parquet_path)
    print(batch_df.select(["user_id", "item_id_seq", "tags_seq"]).head(3))

    # 组装 Interaction 字典
    interaction = {}
    for col in batch_df.columns:
        series = batch_df[col]
        # 如果列没在 setting 里注册(比如 time, seq_len)，默认不处理或者强转即可
        setting = manager_v2.get_setting(col)

        if series.dtype == pl.List:
            if col == "tags_seq":
                # ✨ 针对 3D 序列使用专属 Padding
                arr = pad_nested_list_column(series.to_list(), max_seq_len=5, max_tag_len=3)
            else:
                # ✨ 针对 2D 序列常规 Padding
                arr = pad_list_column(series.to_list(), max_len=5)
            interaction[col] = torch.tensor(arr, dtype=torch.long)

        else:
            try:
                interaction[col] = torch.tensor(series.to_numpy().astype(np.int64), dtype=torch.long)
            except:
                pass  # 忽略无关列

    # 前向传播
    print("\n[*] 向 OmniEmbLayer 喂入张量数据...")
    output_dict = omni_layer(interaction, split_by="name")

    print("\n[+] --- 核心结果见证奇迹 ---")
    print(f" - Tags 序列 (输入端): 形状 {interaction['tags_seq'].shape} -> [Batch, SeqLen, MaxTagLen]")
    print(f" - hist_seq_group 融合特征: 形状 {output_dict['hist_seq_group'].shape}")
    print("   (理论值: [4, 5, 20]，因为 item_id[8维] + tags[12维被Sum池化后] = 20维)")

    assert output_dict['hist_seq_group'].shape[-1] == 20, "组特征拼接维度错误！"
    print("\n✨ SeqGroup 测试圆满成功！")


if __name__ == "__main__":
    test_pipeline()