import torch
from cogdl.wrappers.model_wrapper import ModelWrapper, register_model_wrapper
from cogdl.wrappers.wrapper_utils import pre_evaluation_index


@register_model_wrapper("pprgo_mw")
class PPRGoModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_config):
        super(PPRGoModelWrapper, self).__init__()
        self.optimizer_config = optimizer_config
        self.model = model

    def train_step(self, batch):
        x, targets, ppr_scores, y = batch
        pred = self.model(x, targets, ppr_scores)
        loss = self.default_loss_fn(pred, y)
        return loss

    def val_step(self, batch):
        graph = batch
        pred = self.model.predict(graph)

        y = graph.y[graph.val_mask]
        pred = pred[graph.val_mask]

        loss = self.default_loss_fn(pred, y)
        return dict(val_loss=loss, val_eval_index=pre_evaluation_index(pred, y))

    def test_step(self, batch):
        graph = batch
        pred = self.model.predict(graph)
        test_mask = batch.test_mask
        pred = pred[test_mask]
        y = graph.y[test_mask]
        loss = self.default_loss_fn(pred, y)
        return dict(test_loss=loss, test_eval_index=pre_evaluation_index(pred, y))

    def setup_optimizer(self):
        cfg = self.optimizer_config
        return torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
