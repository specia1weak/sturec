from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import torch

from betterbole.core.enum_type import FeatureSource
from betterbole.core.interaction import Interaction
from betterbole.emb.schema import SparseEmbSetting
from betterbole.models.msr.sharedbottom import SharedBottomModel


def dummy_input_multi_domain(num_domains, batch_size=10, emb_size=128):
    x = torch.randn(batch_size, emb_size)
    domain_ids = torch.randint(0, num_domains, size=(batch_size, ))
    return x, domain_ids


def _build_fitted_sparse_setting(
    field_name: str,
    source: FeatureSource,
    embedding_dim: int,
    values: range,
    *,
    padding_zero: bool = False,
    use_oov: bool = False,
) -> SparseEmbSetting:
    setting = SparseEmbSetting(
        field_name,
        source,
        embedding_dim,
        padding_zero=padding_zero,
        use_oov=use_oov,
    )
    setting._build_vocab_indices([str(value) for value in values])
    return setting


@dataclass(frozen=True)
class DummySchemaManager:
    settings: tuple[SparseEmbSetting, ...]
    uid_field: str
    iid_field: str
    _label_fields: tuple[str, ...]
    _domain_fields: tuple[str, ...]
    time_field: Union[str, None] = None

    @property
    def label_field(self) -> Union[str, None]:
        return next(iter(self._label_fields), None)

    @property
    def label_fields(self) -> tuple[str, ...]:
        return self._label_fields

    @property
    def domain_field(self) -> Union[str, None]:
        return next(iter(self._domain_fields), None)

    @property
    def domain_fields(self) -> tuple[str, ...]:
        return self._domain_fields

    def get_setting(self, field_name: str) -> Union[SparseEmbSetting, None]:
        for setting in self.settings:
            if setting.field_name == field_name:
                return setting
        return None

    def fields(self) -> list[str]:
        fields = []
        for setting in self.settings:
            for output_name in setting.get_output_field_names():
                if output_name not in fields:
                    fields.append(output_name)

        for ctx_field in (self.time_field, *self.label_fields, *self.domain_fields):
            if ctx_field is not None and ctx_field not in fields:
                fields.append(ctx_field)
        return fields


class DummyCls:
    USER_ID = "user_id"
    ITEM_ID = "video_id"
    DOMAIN = "domain"
    LABEL = "label"

    NUM_USERS = 32
    NUM_ITEMS = 64
    NUM_DOMAINS = 3

    EMB_DIM = 8
    DEFAULT_BATCH_SIZE = 4

    INTERACTION_CLS = Interaction
    MODEL_CLS = SharedBottomModel

    SETTINGS_LIST = (
        _build_fitted_sparse_setting(USER_ID, FeatureSource.USER_ID, EMB_DIM, range(NUM_USERS)),
        _build_fitted_sparse_setting(ITEM_ID, FeatureSource.ITEM_ID, EMB_DIM, range(NUM_ITEMS)),
        # domain 字段既参与 embedding，又被 tower 当作 0-based 路由 id 直接使用。
        _build_fitted_sparse_setting(DOMAIN, FeatureSource.INTERACTION, EMB_DIM, range(NUM_DOMAINS)),
    )

    MANAGER = DummySchemaManager(
        settings=SETTINGS_LIST,
        uid_field=USER_ID,
        iid_field=ITEM_ID,
        _label_fields=(LABEL,),
        _domain_fields=(DOMAIN,),
    )

    @classmethod
    def build_model(cls, model_cls=None, **kwargs):
        model_cls = model_cls or cls.MODEL_CLS
        return model_cls(manager=cls.MANAGER, num_domains=cls.NUM_DOMAINS, **kwargs)

    @classmethod
    def make_interaction(
        cls,
        batch_size: Union[int, None] = None,
        *,
        device: Union[torch.device, str, None] = None,
        **overrides: Any,
    ) -> Interaction:
        batch_size = batch_size or cls._infer_batch_size(overrides) or cls.DEFAULT_BATCH_SIZE
        device = torch.device(device) if device is not None else None

        payload = {
            cls.USER_ID: cls._coerce_tensor(
                overrides.pop(cls.USER_ID, cls._randint(0, cls.NUM_USERS, batch_size, device)),
                device=device,
            ),
            cls.ITEM_ID: cls._coerce_tensor(
                overrides.pop(cls.ITEM_ID, cls._randint(0, cls.NUM_ITEMS, batch_size, device)),
                device=device,
            ),
            cls.DOMAIN: cls._coerce_tensor(
                overrides.pop(cls.DOMAIN, cls._randint(0, cls.NUM_DOMAINS, batch_size, device)),
                device=device,
            ),
            cls.LABEL: cls._coerce_tensor(
                overrides.pop(cls.LABEL, cls._randint(0, 2, batch_size, device)),
                device=device,
                dtype=torch.float32,
            ),
        }

        for field_name, value in overrides.items():
            payload[field_name] = cls._coerce_tensor(value, device=device)

        return cls.INTERACTION_CLS(payload)

    @staticmethod
    def _infer_batch_size(overrides: dict[str, Any]) -> Union[int, None]:
        for value in overrides.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        return None

    @staticmethod
    def _randint(
        low: int,
        high: int,
        batch_size: int,
        device: Union[torch.device, None],
    ) -> torch.Tensor:
        return torch.randint(low=low, high=high, size=(batch_size,), dtype=torch.long, device=device)

    @staticmethod
    def _coerce_tensor(
        value: Any,
        *,
        device: Union[torch.device, None],
        dtype: Union[torch.dtype, None] = None,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device) if device is not None else value
            return tensor.to(dtype=dtype) if dtype is not None else tensor

        tensor = torch.as_tensor(value, device=device)
        if dtype is not None:
            return tensor.to(dtype=dtype)
        if tensor.dtype.is_floating_point:
            return tensor.float()
        return tensor.long()
