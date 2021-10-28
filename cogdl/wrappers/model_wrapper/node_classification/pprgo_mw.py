import torch
from cogdl.wrappers.model_wrapper import ModelWrapper


class PPRGoModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg):
        super(PPRGoModelWrapper, self).__init__()
        self.optimizer_cfg = optimizer_cfg
        self.model = model

    def train_step(self, batch):
        x, targets, ppr_scores, y = batch
        pred = self.model(x, targets, ppr_scores)
        loss = self.default_loss_fn(pred, y)
        return loss

    def val_step(self, batch):
        graph = batch
        if isinstance(batch, list):
            x, targets, ppr_scores, y = batch
            pred = self.model(x, targets, ppr_scores)
        else:
            pred = self.model.predict(graph)

            y = graph.y[graph.val_mask]
            pred = pred[graph.val_mask]

        loss = self.default_loss_fn(pred, y)

        metric = self.evaluate(pred, y, metric="auto")

        self.note("val_loss", loss.item())
        self.note("val_metric", metric)

    def test_step(self, batch):
        graph = batch

        if isinstance(batch, list):
            x, targets, ppr_scores, y = batch
            pred = self.model(x, targets, ppr_scores)
        else:
            pred = self.model.predict(graph)
            test_mask = batch.test_mask

            pred = pred[test_mask]
            y = graph.y[test_mask]

        loss = self.default_loss_fn(pred, y)

        self.note("test_loss", loss.item())
        self.note("test_metric", self.evaluate(pred, y))

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
