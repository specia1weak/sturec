"""
Legacy compatibility layer for training-time tensor monitoring.

The real implementation now lives in `betterbole.utils.observatory`.
Models can keep importing `ExplicitFeatureRecorder` from this module,
while new development should prefer the observatory package directly.
"""

from betterbole.utils.observatory import (
    IndividualReLURecorder,
    RelationOptions,
    TensorDisplayConfig,
    TensorMonitorOptions,
    TensorObservatory,
    TensorSketchConfig,
)


class ExplicitFeatureRecorder(TensorObservatory):
    pass


__all__ = [
    "ExplicitFeatureRecorder",
    "IndividualReLURecorder",
    "RelationOptions",
    "TensorDisplayConfig",
    "TensorMonitorOptions",
    "TensorObservatory",
    "TensorSketchConfig",
]
