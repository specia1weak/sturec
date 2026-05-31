from __future__ import annotations

from typing import Dict, Type

from betterbole.models.msr.base import MSRModel
from custom_models.ple_adv import PLEAdversarialModel
from custom_models.ple_adv_dualemb_v1 import PLEAdversarialDualEmbV1Model
from custom_models.ple_adv_transfer_v1 import PLEAdversarialTransferV1Model
from custom_models.ple_balanced_v1 import PLEBalancedV1Model
from custom_models.ple_balanced_v2 import PLEBalancedV2Model
from custom_models.ple_balanced_v3 import PLEBalancedV3Model
from custom_models.ple_ob import PLEObservatoryModel, PLEObservatorySharpModel
from custom_models.ple_shavq_v1 import PLESHAVQV1Model
from custom_models.ple_shavq_v2 import PLESHAVQV2Model
from custom_models.ple_shavq_v3 import PLESHAVQV3Model
from custom_models.shavq_v1 import SHAVQV1Model
from custom_models.shavq_v2 import SHAVQV2Model
from custom_models.shavq_v3 import SHAVQV3Model
from custom_models.shavq_v3b import SHAVQV3BModel
from custom_models.shavq_v3c import SHAVQV3CModel
from custom_models.shavq_v4 import SHAVQV4Model
from custom_models.vq_share import VQShareModel


CUSTOM_MODEL_REGISTRY: Dict[str, Type[MSRModel]] = {
    "ple_adv": PLEAdversarialModel,
    "ple_adv_dualemb_v1": PLEAdversarialDualEmbV1Model,
    "ple_adv_transfer_v1": PLEAdversarialTransferV1Model,
    "ple_ob": PLEObservatoryModel,
    "ple_ob_sharp": PLEObservatorySharpModel,
    "ple_balanced_v1": PLEBalancedV1Model,
    "ple_balanced_v2": PLEBalancedV2Model,
    "ple_balanced_v3": PLEBalancedV3Model,
    "ple_shavq_v1": PLESHAVQV1Model,
    "ple_shavq_v2": PLESHAVQV2Model,
    "ple_shavq_v3": PLESHAVQV3Model,
    "shavq_v1": SHAVQV1Model,
    "shavq_v2": SHAVQV2Model,
    "shavq_v3": SHAVQV3Model,
    "shavq_v3b": SHAVQV3BModel,
    "shavq_v3c": SHAVQV3CModel,
    "shavq_v4": SHAVQV4Model,
    "vq_share": VQShareModel,
}
