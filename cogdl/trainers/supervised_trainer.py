from abc import ABC, abstractmethod

from cogdl.data import Dataset
from cogdl.models.supervised_model import (
    SupervisedModel,
    SupervisedHeterogeneousNodeClassificationModel,
    SupervisedHomogeneousNodeClassificationModel,
)
from cogdl.trainers.base_trainer import BaseTrainer


class SupervisedTrainer(BaseTrainer, ABC):
    @abstractmethod
    def fit(self, model: SupervisedModel, dataset) -> None:
        raise NotImplementedError


class SupervisedHeterogeneousNodeClassificationTrainer(BaseTrainer, ABC):
    @abstractmethod
    def fit(self, model: SupervisedHeterogeneousNodeClassificationModel, dataset: Dataset) -> None:
        raise NotImplementedError


class SupervisedHomogeneousNodeClassificationTrainer(BaseTrainer, ABC):
    @abstractmethod
    def fit(self, model: SupervisedHomogeneousNodeClassificationModel, dataset: Dataset) -> None:
        raise NotImplementedError

    # @abstractmethod
    # def predictAll(self) -> Any:
    #     raise NotImplementedError
