from abc import abstractmethod, ABC
from .base_trainer import BaseTrainer

from cogdl.data import Dataset
from cogdl.models.supervised_model import (
    SupervisedModel,
    SupervisedHomogeneousNodeClassificationModel,
    SupervisedHeterogeneousNodeClassificationModel,
)


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
