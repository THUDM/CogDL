from typing import Optional, Type, Any

import torch.nn as nn
import torch.nn.functional as F

from cogdl.trainers.base_trainer import BaseTrainer


class BaseModel(nn.Module):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model_from_args(cls, args):
        """Build a new model instance."""
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = ""

    def _forward_unimplemented(self, *input: Any) -> None:  # abc warning
        pass

    def forward(self, *args):
        raise NotImplementedError

    def node_classification_loss(self, data):
        pred = self.forward(data.x, data.edge_index)
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(
            pred[data.train_mask],
            data.y[data.train_mask],
        )

    @staticmethod
    def get_trainer(taskType: Any, args: Any) -> Optional[Type[BaseTrainer]]:
        return None

    def set_device(self, device):
        self.device = device
