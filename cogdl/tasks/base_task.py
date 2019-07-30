import torch
import torch.nn as nn


class BaseTask(object):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        pass

    # @classmethod
    # def build_task_from_args(cls, args):
    #     """Build a new task instance."""
    #     raise NotImplementedError(
    #         "Tasks must implement the build_task_from_args method"
    #     )

    def __init__(self, args):
        pass

    def train(self, num_epoch):
        raise NotImplementedError
