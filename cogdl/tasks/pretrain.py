import argparse
import torch

from . import register_task, BaseTask
from cogdl.models import build_model


@register_task("pretrain")
class PretrainTask(BaseTask):
    @staticmethod
    def add_args(_: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument("--num-features", type=int)
        # fmt: on

    def __init__(self, args):
        super(PretrainTask, self).__init__(args)

        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        self.model = build_model(args)
        self.model = self.model.to(self.device)

    def train(self):
        return self.model.trainer.fit()
