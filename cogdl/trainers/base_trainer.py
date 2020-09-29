from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseTrainer(object, ABC):
    @classmethod
    @abstractmethod
    def build_trainer_from_args(cls, args):
        """Build a new trainer instance."""
        raise NotImplementedError(
            "Trainers must implement the build_trainer_from_args method"
        )
