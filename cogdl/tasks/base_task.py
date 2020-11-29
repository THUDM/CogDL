from abc import ABC
import argparse


class BaseTask(ABC):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        pass

    def __init__(self, args):
        pass

    def train(self, num_epoch: int):
        raise NotImplementedError
