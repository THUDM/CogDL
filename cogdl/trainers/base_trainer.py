from argparse import ArgumentParser

from cogdl.models import BaseModel
from cogdl.data import Dataset


class BaseTrainer(object):
    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add task-specific arguments to the parser."""
        pass

    def __init__(self, model: BaseModel, dataset: Dataset):
        self.model = model
        self.dataset = dataset
