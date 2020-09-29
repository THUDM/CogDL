from argparse import ArgumentParser


class BaseTrainer(object):
    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add task-specific arguments to the parser."""
        pass
