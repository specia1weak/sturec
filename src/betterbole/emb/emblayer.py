from typing import Iterable, List, Dict, Union

import numpy as np
import pandas as pd
import torch
from src.betterbole.emb.schema import EmbType, EmbSetting, IdSeqEmbSetting
from src.betterbole.enum_type import FeatureSource
from torch import nn

from src.betterbole.interaction import Interaction


# 将settings中的col分成用户侧和物品侧，BoleEmbLayer会把宽表Interaction 变成对应的Dict Embedding


class BoleEmbLayer(nn.Module):
    def __init__(self, emb_settings: Iterable[EmbSetting]):
        super(BoleEmbLayer, self).__init__()
        self.settings: List[EmbSetting] = list(emb_settings)
        self.emb_modules = nn.ModuleDict()
        self.source2settings: Dict[FeatureSource, List[EmbSetting]] = {}

        for setting in self.settings:
            if self.source2settings.get(setting.source) is None:
                self.source2settings[setting.source] = []
            self.source2settings[setting.source].append(setting)

            if setting.emb_type in (EmbType.DENSE,):
                continue
            # 所有的特征 (不管是数值分箱还是离散ID)，现在都可以统一用 nn.Embedding
            self.emb_modules[setting.field_name] = nn.Embedding(
                num_embeddings=setting.num_embeddings,
                embedding_dim=setting.embedding_size,
                padding_idx={True: 0, False: None}[setting.padding_zero]
            )

    def forward(self, interaction) -> Dict[FeatureSource, torch.Tensor]:
        output_embs = {}
        for source, settings in self.source2settings.items():
            if not settings:
                continue
            source_emb_list = []
            for setting in settings:
                idx_tensor = interaction[setting.field_name]
                # 数字不需要emb
                if setting.emb_type in (EmbType.DENSE,):
                    emb = idx_tensor.unsqueeze(-1)
                    source_emb_list.append(emb)
                    continue

                emb = self.emb_modules[setting.field_name](idx_tensor)
                # 序列特征的 Pooling
                if setting.emb_type in (EmbType.SPARSE_SEQ,):
                    emb = torch.sum(emb, dim=-2)

                source_emb_list.append(emb)

            if source_emb_list:
                # 同一个 Source 下的特征直接 Concat
                output_embs[source] = torch.cat(source_emb_list, dim=-1)
        return output_embs


class SideEmb(nn.Module):
    def __init__(self, all_settings: Iterable[EmbSetting],
                 target_sources: Union[FeatureSource, Iterable[FeatureSource]]=None):
        super(SideEmb, self).__init__()

        # 处理单 Source 和多 Source 的兼容

        if target_sources is not None:
            if not isinstance(target_sources, Iterable):
                target_sources = [target_sources]
            # 核心逻辑：过滤出属于自己这一侧的 settings
            side_settings = [s for s in all_settings if s.source in target_sources]
        else:
            side_settings = all_settings

        self.target_sources = target_sources  # 转为 tuple 方便后续固定顺序遍历
        self.embedding = BoleEmbLayer(side_settings)
        self.embedding_size = sum([s.embedding_size for s in side_settings])

    def forward(self, interaction, flat2tensor=False) -> Union[torch.Tensor, Dict[FeatureSource, torch.Tensor]]:
        # 直接把大宽表吐出的 interaction 传给底层
        emb_dict = self.embedding(interaction)

        if flat2tensor:
            tensors_to_cat = [emb_dict.get(src) for src in self.target_sources if src in emb_dict]
            if tensors_to_cat:
                return torch.cat(tensors_to_cat, dim=-1)
            return None
        else:
            return emb_dict


# 具体化实现，极其干净优雅
class UserSideEmb(SideEmb):
    VALID_SOURCE = (FeatureSource.USER_ID, FeatureSource.USER)

    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings, self.VALID_SOURCE)


class ItemSideEmb(SideEmb):
    VALID_SOURCE = (FeatureSource.ITEM_ID, FeatureSource.ITEM)

    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings, self.VALID_SOURCE)


class InterSideEmb(SideEmb):
    VALID_SOURCE = (FeatureSource.INTERACTION,)

    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings, self.VALID_SOURCE)

# 无任何约束
class CustomSideEmb(SideEmb):
    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings)


class ProfileEncoder(nn.Module):
    """
    全自动静态画像编码器：告别硬编码，完全数据驱动！
    """

    def __init__(self, settings: List['EmbSetting'], profile_path: str,
                 id_source: FeatureSource, feature_source: FeatureSource):
        super().__init__()

        # 1. 解析图纸：找出谁是主键 (ID)，谁是属性 (Features)
        self.id_setting = next((s for s in settings if s.source == id_source), None)
        if not self.id_setting:
            raise ValueError(f"在 settings 中未找到主键源: {id_source}")

        self.feature_settings = [s for s in settings if s.source == feature_source]
        self.feature_names = [s.field_name for s in self.feature_settings]

        # 2. 初始化底层 Embedding 层 (直接复用你写好的 SideEmb)
        self.side_emb = SideEmb(settings, target_sources=[id_source, feature_source])

        # 3. 读取静态宽表
        df = pd.read_parquet(profile_path)

        # 4. 构建地基 (最大 ID 决定了查找表的行数)
        id_col = self.id_setting.field_name
        num_entities = df[id_col].max() + 1
        entity_ids = torch.tensor(df[id_col].astype(np.int64).values, dtype=torch.long)

        # 5. 全自动装配流水线
        for setting in self.feature_settings:
            field = setting.field_name

            # --- 分支 A：处理变长序列特征 (必须 Padding) ---
            if setting.emb_type == EmbType.SPARSE_SEQ:
                max_len = getattr(setting, 'max_len', 50)  # 从 setting 获取最大长度
                padded_matrix = self._pad_sequences(df[field], max_len)

                # 初始化空矩阵 [num_entities, max_len]
                buffer_tensor = torch.zeros((num_entities, max_len), dtype=torch.long)
                vals = torch.tensor(padded_matrix, dtype=torch.long)

            # --- 分支 B：处理单值特征 ---
            else:
                # 初始化一维向量 [num_entities]
                buffer_tensor = torch.zeros(num_entities, dtype=torch.long)
                clean_series = df[field].fillna(0).astype(np.int64)
                vals = torch.tensor(clean_series.values, dtype=torch.long)

            # 散列赋值：自动对齐坑位 (完美解决残缺错位问题)
            buffer_tensor[entity_ids] = vals

            # 动态注册为模型的 Buffer (名字比如: buf_tag, buf_author_id)
            self.register_buffer(f"buf_{field}", buffer_tensor)

    def _pad_sequences(self, series: pd.Series, max_len: int, pad_val: int = 0) -> np.ndarray:
        """高性能 Numpy 序列补齐工具，自带数据清洗"""
        padded = np.full((len(series), max_len), pad_val, dtype=np.int64)
        for i, seq in enumerate(series):
            # 防御由于 left join 产生的 NaN/None
            if seq is None or (isinstance(seq, float) and np.isnan(seq)) or len(seq) == 0:
                continue

            valid_len = min(len(seq), max_len)
            # 【核心修复】不仅截断，还要强制清洗内部元素的类型为 int64
            padded[i, :valid_len] = np.array(seq[:valid_len], dtype=np.int64)

        return padded

    def forward(self, query_ids: torch.Tensor, flat2tensor: bool = False):
        """
        前向传播：根据传入的 query_ids (比如一串 item_id)，
        瞬间从缓存中拉出所有特征，组装成 interaction 字典，喂给 SideEmb。
        """
        # 1. 构建虚拟 Interaction 字典
        interaction = {self.id_setting.field_name: query_ids}

        # 2. 动态查表
        for field in self.feature_names:
            # 取出注册好的 Buffer 矩阵
            buffer_tensor = getattr(self, f"buf_{field}")
            # 高级索引查表，支持 query_ids 是一维 (当前目标) 或二维 (历史序列)
            interaction[field] = buffer_tensor[query_ids]

        # 3. 完美接入你的 SideEmb
        return self.side_emb(interaction, flat2tensor=flat2tensor)

class UserProfileEncoder(ProfileEncoder):
    ID_SOURCE = FeatureSource.USER_ID
    FEATURE_SOURCE = FeatureSource.USER
    def __init__(self, settings: List['EmbSetting'], profile_path: str,):
        super().__init__(settings, profile_path,
                         id_source=self.ID_SOURCE, feature_source=self.FEATURE_SOURCE)

class ItemProfileEncoder(ProfileEncoder):
    ID_SOURCE = FeatureSource.ITEM_ID
    FEATURE_SOURCE = FeatureSource.ITEM
    def __init__(self, settings: List['EmbSetting'], profile_path: str,):
        super().__init__(settings, profile_path,
                         id_source=self.ID_SOURCE, feature_source=self.FEATURE_SOURCE)


class SeqEmbedder(nn.Module):
    def __init__(self, seq_field_name, settings: List[EmbSetting], profile_encoder: ProfileEncoder):
        super().__init__()
        self.settings = settings
        filtered_list = [setting for setting in settings if setting.field_name == seq_field_name]
        if len(filtered_list) != 1:
            raise ValueError
        self.id_setting = filtered_list[0]
        if not isinstance(self.id_setting, IdSeqEmbSetting):
            raise ValueError
        self.id_setting: IdSeqEmbSetting = self.id_setting
        self.seq_field_name = seq_field_name
        self.seq_len_field_name = self.id_setting.seq_len_field_name

        self.profile_encoder = profile_encoder

    def forward(self, interaction: Interaction, flat2tensor=False):
        inter_dict = interaction.interaction
        seq = inter_dict[self.seq_field_name]
        seq_len = inter_dict[self.seq_len_field_name]
        emb_seq = self.profile_encoder(seq)
        if flat2tensor:
            tensors_to_cat = [emb_seq.get(src) for src in emb_seq]
            emb_seq = torch.cat(tensors_to_cat, dim=-1)
        return emb_seq, seq_len