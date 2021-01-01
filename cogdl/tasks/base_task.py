from abc import ABC, ABCMeta
import argparse
import atexit
import os
import torch


class LoadFrom(ABCMeta):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.load_from_pretrained()
        return obj


class BaseTask(ABC, metaclass=LoadFrom):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        pass

    def __init__(self, args):
        super(BaseTask, self).__init__()
        os.makedirs("./checkpoints", exist_ok=True)
        self.load_from_checkpoint = hasattr(args, "checkpoint") and args.checkpoint
        self._checkpoint = os.path.join("checkpoints", f"{args.model}_{args.dataset}.pt")
        atexit.register(self.save_checkpoint)

    def train(self):
        raise NotImplementedError

    def load_from_pretrained(self):
        if self.load_from_checkpoint:
            try:
                ck_pt = torch.load(self._checkpoint)
                self.model.load_state_dict(ck_pt)
            except FileNotFoundError:
                print(f"'{self._checkpoint}' doesn't exists")
        return self.model

    def save_checkpoint(self):
        if self._checkpoint is not None and issubclass(self.model.__class__, torch.nn.Module):
            torch.save(self.model.state_dict(), self._checkpoint)
