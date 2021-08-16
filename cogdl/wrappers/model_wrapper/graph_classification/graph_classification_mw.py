import torch

from .. import register_model_wrapper, ModelWrapper
from cogdl.wrappers.tools.wrapper_utils import pre_evaluation_index


@register_model_wrapper("graph_classification_mw")
class GraphClassificationModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg):
        super(GraphClassificationModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg

    def train_step(self, batch):
        pred = self.model(batch)
        y = batch.y
        loss = self.default_loss_fn(pred, y)
        return loss

    def val_step(self, batch):
        pred = self.model(batch)
        y = batch.y
        val_loss = self.default_loss_fn(pred, y)

        self.note("val_loss", val_loss)
        self.note("val_eval_index", pre_evaluation_index(pred, y))

    def test_step(self, batch):
        pred = self.model(batch)
        y = batch.y
        test_loss = self.default_loss_fn(pred, y)

        self.note("test_loss", test_loss)
        self.note("test_eval_index", pre_evaluation_index(pred, y))

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
