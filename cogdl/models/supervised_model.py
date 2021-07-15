from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from typing import TYPE_CHECKING

from cogdl.models.base_model import BaseModel

if TYPE_CHECKING:
    # trick for resolve circular import
    from cogdl.trainers.supervised_model_trainer import (
        SupervisedHomogeneousNodeClassificationTrainer,
        SupervisedHeterogeneousNodeClassificationTrainer,
    )


class SupervisedModel(BaseModel, ABC):
    @abstractmethod
    def loss(self, data: Any) -> Any:
        raise NotImplementedError


class SupervisedHeterogeneousNodeClassificationModel(BaseModel, ABC):
    @abstractmethod
    def loss(self, data: Any) -> Any:
        raise NotImplementedError

    def evaluate(self, data: Any, nodes: Any, targets: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def get_trainer(args: Any = None) -> "Optional[Type[SupervisedHeterogeneousNodeClassificationTrainer]]":
        return None


class SupervisedHomogeneousNodeClassificationModel(BaseModel, ABC):
    @abstractmethod
    def loss(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def get_trainer(args: Any = None) -> "Optional[Type[SupervisedHomogeneousNodeClassificationTrainer]]":
        return None
