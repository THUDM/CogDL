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
    def __init__(
        self, model: SupervisedHeterogeneousNodeClassificationModel, dataset: Dataset
    ):
        self.model = model
        self.dataset = dataset

    @abstractmethod
    def fit(self) -> None:
        raise NotImplemented

    @abstractmethod
    def evaluate(self, data: Any, nodes: Any, targets: Any) -> Any:
        raise NotImplemented


class SupervisedHomogeneousNodeClassificationTrainer(BaseTrainer, ABC):
    def __init__(
        self, model: SupervisedHomogeneousNodeClassificationModel, dataset: Dataset
    ):
        self.model = model
        self.dataset = dataset

    @abstractmethod
    def fit(self) -> None:
        raise NotImplemented

    @abstractmethod
    def predictAll(self) -> Any:
        raise NotImplemented
