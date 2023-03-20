from typing import Optional, Type, Any
from jittor import nn, Module


class BaseModel(Module):
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
        self.model_name = self.__class__.__name__
        self.loss_fn = None
        self.evaluator = None

    def _forward_unimplemented(self, *input: Any) -> None:  # abc warning
        pass

    def execute(self, *args):
        raise NotImplementedError

    def predict(self, data):
        return self.forward(data)

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
