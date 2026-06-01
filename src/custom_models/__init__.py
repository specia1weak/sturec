from __future__ import annotations

from typing import Dict, Type

from betterbole.models.msr.base import MSRModel
from custom_models.curriculum_ssim import CurriculumSSIMModel
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
from custom_models.shavq_h2_ste_v1 import SHAVQH2STEV1Model
from custom_models.shavq_md_ste_v2 import SHAVQMDSTEV2Model
from custom_models.shavq_md_ste_v3 import SHAVQMDSTEV3Model
from custom_models.shavq_md_v1 import SHAVQMDV1Model
from custom_models.shavq_md_ste_v1 import SHAVQMDSTEV1Model
from custom_models.vq_share import VQShareModel
from custom_models.vq_share_gate_v1 import VQShareGateV1Model


CUSTOM_MODEL_REGISTRY: Dict[str, Type[MSRModel]] = {
    "curriculum_ssim": CurriculumSSIMModel,
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
    "shavq_h2_ste_v1": SHAVQH2STEV1Model,
    "shavq_h2_ste_v1_r001": SHAVQH2STEV1Model,
    "shavq_h2_ste_v1_r001_b050": SHAVQH2STEV1Model,
    "shavq_h2_ste_v1_r001_fuse": SHAVQH2STEV1Model,
    "shavq_md_v1": SHAVQMDV1Model,
    "shavq_md_ste_v1": SHAVQMDSTEV1Model,
    "shavq_md_ste_v2": SHAVQMDSTEV2Model,
    "shavq_md_ste_v2_s090_f000": SHAVQMDSTEV2Model,
    "shavq_md_ste_v2_s090_f005": SHAVQMDSTEV2Model,
    "shavq_md_ste_v2_s095_f000": SHAVQMDSTEV2Model,
    "shavq_md_ste_v2_s095_f005": SHAVQMDSTEV2Model,
    "shavq_md_ste_v3": SHAVQMDSTEV3Model,
    "shavq_md_ste_v3_r001": SHAVQMDSTEV3Model,
    "shavq_md_ste_v3_r005": SHAVQMDSTEV3Model,
    "shavq_md_ste_v3_r001_k128": SHAVQMDSTEV3Model,
    "vq_share": VQShareModel,
    "vq_share_gate_v1": VQShareGateV1Model,
}
