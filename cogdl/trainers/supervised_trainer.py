from abc import ABC, abstractmethod
from typing import Any

from cogdl.data import Dataset
from cogdl.models.supervised_model import (
    SupervisedHeterogeneousNodeClassificationModel,
    SupervisedHomogeneousNodeClassificationModel,
)
from cogdl.trainers.base_trainer import BaseTrainer


class SupervisedTrainer(BaseTrainer, ABC):
    @abstractmethod
    def fit(self) -> None:
        raise NotImplemented

    @abstractmethod
    def predict(self) -> Any:
        raise NotImplemented


class SupervisedHeterogeneousNodeClassificationTrainer(BaseTrainer, ABC):
    @abstractmethod
    def fit(
        self, model: SupervisedHeterogeneousNodeClassificationModel, dataset: Dataset
    ) -> None:
        raise NotImplemented

    # @abstractmethod
    # def evaluate(self, data: Any, nodes: Any, targets: Any) -> Any:
    #     raise NotImplemented


class SupervisedHomogeneousNodeClassificationTrainer(BaseTrainer, ABC):
    @abstractmethod
    def fit(
        self, model: SupervisedHomogeneousNodeClassificationModel, dataset: Dataset
    ) -> None:
        raise NotImplemented

    # @abstractmethod
    # def predictAll(self) -> Any:
    #     raise NotImplemented
