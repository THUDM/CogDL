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
        raise NotImplementedError

    @abstractmethod
    def predict(self) -> Any:
        raise NotImplementedError


class SupervisedHeterogeneousNodeClassificationTrainer(BaseTrainer, ABC):
    @abstractmethod
    def fit(self, model: SupervisedHeterogeneousNodeClassificationModel, dataset: Dataset) -> None:
        raise NotImplementedError

    # @abstractmethod
    # def evaluate(self, data: Any, nodes: Any, targets: Any) -> Any:
    #     raise NotImplementedError


class SupervisedHomogeneousNodeClassificationTrainer(BaseTrainer, ABC):
    @abstractmethod
    def fit(self, model: SupervisedHomogeneousNodeClassificationModel, dataset: Dataset) -> None:
        raise NotImplementedError

    # @abstractmethod
    # def predictAll(self) -> Any:
    #     raise NotImplementedError
