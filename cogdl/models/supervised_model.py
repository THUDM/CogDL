from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from typing import TYPE_CHECKING

from cogdl.models import BaseModel

if TYPE_CHECKING:
    # trick for resolve circular import
    from cogdl.trainers.supervised_trainer import (
        SupervisedHomogeneousNodeClassificationTrainer,
        SupervisedHeterogeneousNodeClassificationTrainer,
    )


class SupervisedModel(BaseModel, ABC):
    @abstractmethod
    def loss(self, data: Any) -> Any:
        raise NotImplemented


class SupervisedHeterogeneousNodeClassificationModel(BaseModel, ABC):
    @abstractmethod
    def loss(self, data: Any) -> Any:
        raise NotImplemented

    def evaluate(self, data: Any, nodes: Any, targets: Any) -> Any:
        raise NotImplemented

    @staticmethod
    def get_trainer(
        taskType: Any, args: Any
    ) -> "Optional[Type[SupervisedHeterogeneousNodeClassificationTrainer]]":
        return None


class SupervisedHomogeneousNodeClassificationModel(BaseModel, ABC):
    @abstractmethod
    def loss(self, data: Any) -> Any:
        raise NotImplemented

    @abstractmethod
    def predict(self, data: Any) -> Any:
        raise NotImplemented

    @staticmethod
    def get_trainer(
        taskType: Any, args: Any,
    ) -> "Optional[Type[SupervisedHomogeneousNodeClassificationTrainer]]":
        return None
