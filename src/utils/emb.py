"""
要脱离recbole自己写Embedding的话比较麻烦
首先一个EmbeddingLayer有自动拼接操作，他必须管理每一个Embedding的设定


为每一个field设定一个 EmbSetting抽象类, 传入dataset
name, embedding_size, embedding_type

子类SparseEmbSetting -> SPARSE
设定size

子类AbsRangeEmbSetting -> ABS_RANGE 绝对值区间数值Embedding
设定
子类QuantileEmbSetting -> QUANTILE 传入dataset，等频分箱


然后我有一个SuperEmbeddingLayer, 根据传来的东西我来处理 user_id -> user侧   item_id/seq -> item侧   直接给interaction -> user/item/inter 三个给出
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, Dict, List, Union

import numpy as np
import torch
from recbole.data import Interaction
from torch import nn
from recbole.data.dataset import Dataset
from recbole.utils import FeatureSource
from torch.nn.functional import embedding


class EmbType(Enum):
    UNKNOWN = None
    SPARSE = "sparse"
    ABS_RANGE = "abs_range"
    QUANTILE = "quantile"
    SPARSE_SEQ = "sparse_seq"

class EmbSetting(ABC):
    embedding_type = EmbType.UNKNOWN
    def __init__(self, dataset: Dataset, field_name, embedding_size=0):
        self.dataset = dataset
        self.field_name = field_name
        self.embedding_size = embedding_size
        self.source = dataset.field2source[self.field_name]

    @property
    @abstractmethod
    def num_embeddings(self) -> int:
        """返回该特征对应的 Embedding 词表大小"""
        pass

    @abstractmethod
    def feature2index(self, feature: torch.Tensor):
        raise NotImplementedError

class SparseEmbSetting(EmbSetting):
    embedding_type = EmbType.SPARSE
    def __init__(self, dataset, field_name, embedding_size=16):
        super().__init__(dataset, field_name, embedding_size)
        self._num_embeddings = dataset.num(self.field_name)
    @property
    def num_embeddings(self) -> int:
        return self._num_embeddings
    def feature2index(self, feature: torch.Tensor):
        return feature.clone().to(torch.long)

class AbsRangeEmbSetting(EmbSetting):
    embedding_type = EmbType.ABS_RANGE
    def __init__(self, dataset, field_name, abs_values, embedding_size=16):
        super().__init__(dataset, field_name, embedding_size)
        self.abs_values = abs_values
        self.boundaries = self.register_boundaries(abs_values)
        self.bucket_count = len(self.boundaries) + 1
    def register_boundaries(self, values):
        return torch.tensor(sorted(values), dtype=torch.float32)
    @property
    def num_embeddings(self) -> int:
        # 边界数 + 1 就是分桶的数量，对应 embedding 的词表大小
        return len(self.boundaries) + 1
    def feature2index(self, feature: torch.Tensor):
        return torch.bucketize(feature.detach(), self.boundaries, right=False).to(torch.long)

class SparseSeqEmbSetting(EmbSetting):
    embedding_type = EmbType.SPARSE_SEQ
    def __init__(self, dataset, field_name, embedding_size=16):
        super().__init__(dataset, field_name, embedding_size)
        self._num_embeddings = dataset.num(self.field_name)
    @property
    def num_embeddings(self) -> int:
        return self._num_embeddings
    def feature2index(self, feature: torch.Tensor):
        return feature.clone().to(torch.long)

class QuantileEmbSetting(EmbSetting):
    embedding_type = EmbType.QUANTILE
    def __init__(self, dataset, field_name, bucket_count, embedding_size=16):
        super().__init__(dataset, field_name, embedding_size)
        self.bucket_count = bucket_count
        self.boundaries = self._statistics_from_dataset(dataset)

    def _statistics_from_dataset(self, dataset: Dataset):
        field_col = None
        if self.source == FeatureSource.USER:
            field_col = dataset.user_feat[self.field_name]
        elif self.source == FeatureSource.ITEM:
            field_col = dataset.item_feat[self.field_name]
        elif self.source == FeatureSource.INTERACTION:
            field_col = dataset.inter_feat[self.field_name]
        else:
            raise ValueError(f"未知的 FeatureSource: {self.source}")

        if field_col is None or len(field_col) == 0:
            return torch.tensor([], dtype=torch.float32)

            # RecBole 的 interaction 返回的通常是 tensor，但保险起见做一次强转
        if not isinstance(field_col, torch.Tensor):
            field_col = torch.tensor(field_col)

            # torch.quantile 要求输入必须是浮点型 (float32 或 float64)
        field_col = field_col.float()
        q = torch.linspace(0, 1, steps=self.bucket_count + 1)[1:-1]
        boundaries = torch.quantile(field_col, q)
        boundaries = torch.unique(boundaries)

        return boundaries

    @property
    def num_embeddings(self) -> int:
        # 因为经过 unique 去重，必须以实际生成的 boundaries 长度为准
        if self.boundaries is None or self.boundaries.numel() == 0:
            return 1  # 退化情况：所有值都映射到 index 0
        return len(self.boundaries) + 1

    def feature2index(self, feature: torch.Tensor):
        if self.boundaries is None or self.boundaries.numel() == 0:
            return torch.zeros_like(feature, dtype=torch.long)
        # bucketize 根据边界把数值映射到 0 ~ len(boundaries) 的索引上
        return torch.bucketize(feature.detach(), self.boundaries, right=False).to(torch.long)

class ItemSeqEmbSetting(EmbSetting):
    embedding_type = EmbType.SPARSE
    def __init__(self, field_name, item_emb_setting: EmbSetting):
        super().__init__(item_emb_setting.dataset, field_name, item_emb_setting.embedding_size)
        self.item_emb_setting = item_emb_setting
    def num_embeddings(self) -> int:
        return self.item_emb_setting.num_embeddings
    def feature2index(self, feature: torch.Tensor):
        return self.item_emb_setting.feature2index(feature)

class BoleEmbLayer(nn.Module):
    def __init__(self, emb_settings: Iterable[EmbSetting]):
        super(BoleEmbLayer, self).__init__()
        self.settings: List[EmbSetting] = list(emb_settings)
        self.emb_modules = nn.ModuleDict()
        self.source2settings = {}

        for setting in self.settings:
            if self.source2settings.get(setting.source) is None:
                self.source2settings[setting.source] = []
            self.source2settings[setting.source].append(setting)

            # 核心改进：直接调用 setting.num_embeddings，无需关心具体的 embedding_type
            self.emb_modules[setting.field_name] = nn.Embedding(
                num_embeddings=setting.num_embeddings,
                embedding_dim=setting.embedding_size,
                padding_idx=0
            )
    def forward_with_interaction(self, interaction):
        output_embs = {}
        for source, settings in self.source2settings.items():
            if not settings:
                continue
            source_emb_list = []
            for setting in settings:
                raw_feature = interaction[setting.field_name]
                idx = setting.feature2index(raw_feature)
                emb = self.emb_modules[setting.field_name](idx)
                if setting.embedding_type in (EmbType.SPARSE_SEQ, ):
                    emb = torch.sum(emb, dim=-2)
                source_emb_list.append(emb)
            if source_emb_list:
                output_embs[source] = torch.cat(source_emb_list, dim=-1)
        return output_embs

    def forward(self, x: Interaction) -> Dict[FeatureSource, torch.Tensor]:
        return self.forward_with_interaction(x)


class UserSideEmb(nn.Module):
    VALID_SOURCE = (FeatureSource.USER_ID, FeatureSource.USER)
    def __init__(self, user_side_settings: List[EmbSetting]):
        super(UserSideEmb, self).__init__()
        user_side_settings = [setting for setting in user_side_settings if setting.source in self.VALID_SOURCE] # 保证他们的Source来自User和User_id
        self.embedding_size = sum([setting.embedding_size for setting in user_side_settings])
        self.dataset = user_side_settings[0].dataset
        self.user_feat = self.dataset.get_user_feature()
        self.embedding = BoleEmbLayer(user_side_settings)
        self.USER_ID = self.dataset.uid_field
    def forward(self, uid: Union[torch.Tensor, Interaction], flat2tensor=False)->Union[torch.Tensor,Dict[FeatureSource, torch.Tensor]]:
        if isinstance(uid, Interaction):
            uid = uid.interaction[self.USER_ID]
        if self.user_feat.interaction[self.USER_ID].device != uid.device:
            self.user_feat = self.user_feat.to(uid.device)
        interaction = self.user_feat[uid]
        emb_dict = self.embedding(interaction)
        if flat2tensor:
            return torch.cat([emb_dict.get(k) for k in self.VALID_SOURCE if k in emb_dict], dim=-1) # dim=1 的cat
        else:
            return emb_dict

class ItemSideEmb(nn.Module):
    VALID_SOURCE = (FeatureSource.ITEM_ID, FeatureSource.ITEM)
    def __init__(self, item_side_settings: List[EmbSetting]):
        super(ItemSideEmb, self).__init__()
        item_side_settings = [setting for setting in item_side_settings if setting.source in self.VALID_SOURCE]  # 保证他们的Source来自User和User_id
        self.embedding_size = sum([setting.embedding_size for setting in item_side_settings])
        self.dataset = item_side_settings[0].dataset
        self.item_feat = self.dataset.get_item_feature()
        self.embedding = BoleEmbLayer(item_side_settings)
        self.ITEM_ID = self.dataset.iid_field

    def forward(self, iid: Union[torch.Tensor, Interaction], flat2tensor=False)->Union[torch.Tensor,Dict[FeatureSource, torch.Tensor]]:
        if isinstance(iid, Interaction):
            iid = iid.interaction[self.ITEM_ID]
        if self.item_feat.interaction[self.ITEM_ID].device != iid.device:
            self.item_feat = self.item_feat.to(iid.device)
        interaction = self.item_feat[iid]
        emb_dict = self.embedding(interaction)
        if flat2tensor:
            return torch.cat([emb_dict.get(k) for k in self.VALID_SOURCE if k in emb_dict], dim=-1)  # dim=1 的cat
        else:
            return emb_dict