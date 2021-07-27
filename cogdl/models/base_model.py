from typing import Optional, Type, Any
import torch.nn as nn

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
        self.loss_fn = None
        self.evaluator = None

    def _forward_unimplemented(self, *input: Any) -> None:  # abc warning
        pass

    def forward(self, *args):
        raise NotImplementedError

    def predict(self, data):
        return self.forward(data)

    def node_classification_loss(self, data, mask=None):
        if mask is None:
            mask = data.train_mask
        pred = self.forward(data)
        return self.loss_fn(pred[mask], data.y[mask])

    def graph_classification_loss(self, batch):
        pred = self.forward(batch)
        return self.loss_fn(pred, batch.y)

    @staticmethod
    def get_trainer(args=None) -> Optional[Type[BaseTrainer]]:
        return None

    def set_device(self, device):
        self.device = device

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
