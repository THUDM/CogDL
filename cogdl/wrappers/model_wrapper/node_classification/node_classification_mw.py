import torch
from cogdl.wrappers.model_wrapper import ModelWrapper, register_model_wrapper
from cogdl.wrappers.wrapper_utils import pre_evaluation_index


@register_model_wrapper("node_classification_mw")
class NodeClfModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_config):
        super(NodeClfModelWrapper, self).__init__()
        self.optimizer_config = optimizer_config
        self.model = model

    def train_step(self, batch):
        graph = batch
        pred = self.model(graph)
        train_mask = batch.train_mask
        loss = self.default_loss_fn(pred[train_mask], batch.y[train_mask])
        return loss

    def val_step(self, batch):
        graph = batch
        pred = self.model(graph)
        y = batch.y
        val_mask = batch.val_mask
        loss = self.default_loss_fn(pred[val_mask], y[val_mask])
        return dict(val_loss=loss, val_eval_index=pre_evaluation_index(pred[val_mask], y[val_mask]))

    def test_step(self, batch):
        graph = batch
        pred = self.model(graph)
        test_mask = batch.test_mask
        loss = self.default_loss_fn(pred[test_mask], batch.y[test_mask])
        return dict(test_loss=loss, test_eval_index=pre_evaluation_index(pred[test_mask], batch.y[test_mask]))

    def setup_optimizer(self):
        cfg = self.optimizer_config
        if hasattr(self.model, "get_optimizer"):
            model_spec_optim = self.model.get_optimizer(cfg)
            if model_spec_optim is not None:
                return model_spec_optim
        return torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
