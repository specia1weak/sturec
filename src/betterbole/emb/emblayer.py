from typing import Iterable, List, Dict, Union, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from betterbole.emb.schema import (
    BaseSequenceSetting,
    EmbSetting,
    EmbType,
    IdSeqEmbSetting,
    SparseSetEmbSetting,
)
from betterbole.core.enum_type import FeatureSource
from torch import nn

from betterbole.core.interaction import Interaction

if TYPE_CHECKING:
    from betterbole.emb.manager import SchemaManager


# 将settings中的col分成用户侧和物品侧，BoleEmbLayer会把宽表Interaction 变成对应的Dict Embedding


SPLIT_METHODS = Literal["source", "name", "none", None]


class RecEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, init_method='normal', init_std=1e-4):
        super(RecEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.init_method = init_method
        self.init_std = init_std
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_method == 'normal':
            nn.init.normal_(self.embedding.weight, mean=0.0, std=self.init_std)
        elif self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight.data)
        else:
            pass
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    def forward(self, x):
        return self.embedding(x)


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

            if setting.requires_embedding_module:
                self.emb_modules[setting.field_name] = RecEmbedding(
                    num_embeddings=setting.num_embeddings,
                    embedding_dim=setting.embedding_dim,
                    padding_idx={True: 0, False: None}[setting.padding_zero]
                )

    def forward(self, interaction, split_by: SPLIT_METHODS="none") -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor]:
        split_by = split_by.lower() if isinstance(split_by, str) else None
        if split_by not in ("source", "name", "none", None):
            raise ValueError

        output_embs = {}
        for source, settings in self.source2settings.items():
            if not settings:
                continue
            emb_list = []
            for setting in settings:
                emb = setting.compute_tensor(interaction, self.emb_modules)
                emb_list.append([setting.field_name, emb])

            if emb_list:
                if split_by == "source":
                    output_embs[source] = torch.cat([emb for name, emb in emb_list], dim=-1)
                elif split_by == "name":
                    for name, emb in emb_list:
                        output_embs[name] = emb
                else:
                    output_embs["all"] = output_embs.get("all", [])
                    output_embs["all"].extend([emb for name, emb in emb_list])

        if split_by is None or split_by == "none":
            if output_embs.get("all", None) is None:
                return None
            else:
                return torch.cat(output_embs.get("all"), dim=-1)
        else:
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
        self.embedding_size = sum([s.embedding_dim for s in side_settings])

    def forward(self, interaction, split_by: SPLIT_METHODS="none") -> Union[torch.Tensor, Dict[FeatureSource, torch.Tensor]]:
        # 直接把大宽表吐出的 interaction 传给底层
        emb_dict = self.embedding(interaction, split_by)
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

"""已废弃"""
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
            if setting.emb_type in (EmbType.SPARSE_SET, ):
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

    def forward(self, query_ids: torch.Tensor, split_by: SPLIT_METHODS= "none"):
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
            interaction[field] = buffer_tensor[query_ids]

        # 3. 完美接入你的 SideEmb
        return self.side_emb(interaction, split_by=split_by)

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

    def forward(self, interaction: Interaction, split_by: SPLIT_METHODS="none"):
        inter_dict = interaction.interaction
        seq = inter_dict[self.seq_field_name]
        seq_len = inter_dict[self.seq_len_field_name]
        emb_seq = self.profile_encoder.forward(seq, split_by)
        return emb_seq, seq_len


class EmbView:
    """
    轻量级特征视图：
    本身没有任何参数，只负责将请求转发给底层的 OmniEmbLayer，并自动绑定 target_sources。
    """
    def __init__(self,
                 omni_layer: 'OmniEmbLayer',
                 target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
                 include_fields: Iterable[str] = None,
                 exclude_fields: Iterable[str] = None):
        super().__init__()
        # 引用同一个巨无霸底座
        self.omni = omni_layer
        self.target_sources = target_sources
        self.include_fields = tuple(include_fields) if include_fields is not None else None
        self.exclude_fields = tuple(exclude_fields) if exclude_fields is not None else None

    def forward(self, interaction, split_by: SPLIT_METHODS = "none") -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
        return self.omni(
            interaction,
            split_by=split_by,
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    def __call__(self, interaction, split_by: SPLIT_METHODS = "none") -> Union[
        Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
        return self.omni(
            interaction,
            split_by=split_by,
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    @property
    def embedding_dim(self) -> int:
        return self.omni.get_output_dim(
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )


class SeqGroupView(EmbView):
    """
    专为序列组设计的强化版视图：
    不仅能获取序列特征，还能自动获取严格对齐的 Target 特征和序列长度。
    """

    def __init__(self, omni_layer, group_name: str, include_fields: Iterable[str]):
        super().__init__(omni_layer, include_fields=include_fields)
        self.group_name = group_name

        # 预加载对齐的 target 设置与 seq_len 字段
        self.target_settings = []
        self.seq_len_field = None
        self._target_forward_valid = True

        for field in self.include_fields:
            setting = next((s for s in self.omni.settings if s.field_name == field), None)
            if setting is None:
                continue

            if hasattr(setting, 'target_setting'):
                self.target_settings.append(setting.target_setting)
            else:
                self._target_forward_valid = False

            if self.seq_len_field is None and hasattr(setting, 'seq_len_field_name'):
                self.seq_len_field = setting.seq_len_field_name

    def forward_target(self, interaction, split_by: str = "none"):
        # 实际上所有setting必须拥有target_setting forward_target才不会有歧义
        if split_by == "source":
            raise ValueError("Target sequence view strictly supports split_by='none' or 'name'.")

        computed_results = []
        for target_s in self.target_settings:
            emb = target_s.compute_tensor(interaction, self.omni.emb_modules)
            computed_results.append((target_s, emb))
        return self.omni.format_output(computed_results, split_by)

    def fetch_all(self, interaction):
        """一键打包返回：(序列特征拼接, 目标特征拼接, 序列长度)"""
        seq_emb = self.forward(interaction, split_by="none")
        tar_emb = self.forward_target(interaction, split_by="none") if self._target_forward_valid else ValueError("存在部分setting没有target_setting，函数中断")
        seq_len = interaction[self.seq_len_field] if self.seq_len_field else None

        return seq_emb, tar_emb, seq_len

class OmniEmbLayer(nn.Module):
    """
    巨无霸统一 Embedding 层：
    内部纳管所有 settings。forward 时可通过 target_sources 动态指定范围。
    彻底取代 SideEmb、UserSideEmb、ItemSideEmb 等一众子类。
    """
    def __init__(self, emb_settings: Iterable[EmbSetting] = None, manager: 'SchemaManager' = None):
        super(OmniEmbLayer, self).__init__()
        if manager is not None:
            emb_settings = manager.settings
        if emb_settings is None:
            raise ValueError("OmniEmbLayer requires manager or emb_settings.")

        self.manager = manager
        self.settings: List[EmbSetting] = list(emb_settings)
        self.emb_modules = nn.ModuleDict()
        self.source2settings: Dict[FeatureSource, List[EmbSetting]] = {}

        self.group2fields = {}
        self._init_embedding()

        self.seq_groups = {}
        for g_name, fields in self.group2fields.items():
            self.seq_groups[g_name] = SeqGroupView(self, group_name=g_name, include_fields=fields)

        self.whole = EmbView(self, target_sources=(FeatureSource.USER_ID, FeatureSource.USER,
                                                   FeatureSource.ITEM_ID, FeatureSource.ITEM,
                                                   FeatureSource.INTERACTION))
        self.user_all = EmbView(self, target_sources=(FeatureSource.USER_ID, FeatureSource.USER))
        self.item_all = EmbView(self, target_sources=(FeatureSource.ITEM_ID, FeatureSource.ITEM))
        self.inter = EmbView(self, target_sources=(FeatureSource.INTERACTION,))

        self.user_id = EmbView(self, target_sources=(FeatureSource.USER_ID,))
        self.item_id = EmbView(self, target_sources=(FeatureSource.ITEM_ID,))


        domain_fields = tuple(manager.domain_fields) if manager is not None else tuple()
        domain_field = manager.domain_field if manager is not None else None
        self.domain = EmbView(self, include_fields=domain_fields)
        self.domain_id = EmbView(
            self,
            include_fields=((domain_field,) if domain_field is not None else tuple()),
        )
        self.whole_without_domain = EmbView(
            self,
            target_sources=(FeatureSource.USER_ID, FeatureSource.USER,
                            FeatureSource.ITEM_ID, FeatureSource.ITEM,
                            FeatureSource.INTERACTION),
            exclude_fields=domain_fields,
        )
        self.inter_without_domain = EmbView(
            self,
            target_sources=(FeatureSource.INTERACTION,),
            exclude_fields=domain_fields,
        )

    def _init_embedding(self):
        for setting in self.settings:
            # A. 维护 Source 映射
            self.source2settings.setdefault(setting.source, []).append(setting)

            # B. 确定哪些特征需要物理权重 (nn.Embedding)
            if setting.requires_embedding_module:
                self.emb_modules[setting.field_name] = RecEmbedding(
                    num_embeddings=setting.num_embeddings,
                    embedding_dim=setting.embedding_dim,
                    padding_idx={True: 0, False: None}[setting.padding_zero]
                )

            # C. 自动收集带有组名标签的序列特征
            if hasattr(setting, 'group_name') and setting.group_name:
                self.group2fields.setdefault(setting.group_name, []).append(setting.field_name)


    def forward(
            self,
            interaction,
            split_by: SPLIT_METHODS = "none",
            target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
            include_fields: Iterable[str] = None,
            exclude_fields: Iterable[str] = None,
    ) -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:

        split_by = split_by.lower() if isinstance(split_by, str) else None
        if split_by not in ("source", "name", "none", None):
            raise ValueError(f"Invalid split_by: {split_by}")

        # 1. 路由过滤：找出本次 forward 需要处理的所有 Setting
        valid_settings = self._filter_settings(target_sources, include_fields, exclude_fields, split_by)
        computed = [(s, s.compute_tensor(interaction, self.emb_modules)) for s in valid_settings]
        return self.format_output(computed, split_by)


    def _filter_settings(self, target_sources, include_fields, exclude_fields, split_by) -> List[EmbSetting]:
        """职责一：专门负责解析 target_sources 和 include/exclude_fields，返回合法的 Setting 列表"""
        target_sources = set(target_sources) if target_sources else None
        include_fields = set(include_fields) if include_fields else None
        exclude_fields = set(exclude_fields) if exclude_fields else None
        valid_settings = []
        for source, settings in self.source2settings.items():
            if target_sources and source not in target_sources:
                continue
            for setting in settings:
                if include_fields and setting.field_name not in include_fields:
                    continue
                if exclude_fields and setting.field_name in exclude_fields:
                    continue

                is_sequence_type = isinstance(setting, BaseSequenceSetting) and not isinstance(setting, SparseSetEmbSetting)
                if (split_by == "none" or split_by is None) and is_sequence_type:
                    if not (include_fields and setting.field_name in include_fields):
                        continue

                valid_settings.append(setting)
        return valid_settings

    def format_output(self, computed_results: List[tuple], split_by: str):
        """职责三：根据 split_by 策略打包最终输出的 Tensor 或 Dict"""
        if not computed_results:
            return None

        if split_by == "name":
            return {setting.field_name: emb for setting, emb in computed_results}

        if split_by == "source":
            output_dict = {}
            for setting, emb in computed_results:
                output_dict.setdefault(setting.source, []).append(emb)
            return {source: torch.cat(embs, dim=-1) for source, embs in output_dict.items()}

        all_embs = [emb for _, emb in computed_results]
        return torch.cat(all_embs, dim=-1)

    def get_output_dim(self,
                       target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
                       include_fields: Iterable[str] = None,
                       exclude_fields: Iterable[str] = None) -> int:
        """
        获取指定 target_sources 下拼接后 Embedding 的总维度大小。
        这是为了弥补移除 SideEmb 后，下游网络无法直接获取 self.embedding_size 的问题。
        """
        valid_settings = self._filter_settings(
            target_sources=target_sources,
            include_fields=include_fields,
            exclude_fields=exclude_fields,
            split_by="none",
        )
        return sum(setting.embedding_dim for setting in valid_settings)
