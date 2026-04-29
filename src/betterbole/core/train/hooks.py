from typing import Any, Optional, Protocol, runtime_checkable

from betterbole.core.train.context import TrainContext


@runtime_checkable
class CustomTrainStepProtocol(Protocol):
    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        ...


@runtime_checkable
class TrainerHooksProtocol(Protocol):
    def on_train_epoch_start(self, ctx: TrainContext) -> None:
        ...

    def on_train_epoch_end(self, ctx: TrainContext) -> None:
        ...

    def on_eval_epoch_end(
            self,
            metrics: Optional[dict[str, Any]],
            ctx: TrainContext,
    ) -> None:
        ...
