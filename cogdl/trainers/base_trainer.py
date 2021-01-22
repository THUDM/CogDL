from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @classmethod
    @abstractmethod
    def build_trainer_from_args(cls, args):
        """Build a new trainer instance."""
        raise NotImplementedError("Trainers must implement the build_trainer_from_args method")

    @abstractmethod
    def fit(self, model, dataset):
        raise NotImplementedError
