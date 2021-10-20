import torch

from .. import ModelWrapper


class GraphSAGEModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg):
        super(GraphSAGEModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg

    def train_step(self, batch):
        x_src, y, adjs = batch
        pred = self.model(x_src, adjs)
        loss = self.default_loss_fn(pred, y)
        return loss

    def val_step(self, batch):
        x_src, y, adjs = batch
        pred = self.model(x_src, adjs)
        loss = self.default_loss_fn(pred, y)

        metric = self.evaluate(pred, y, metric="auto")

        self.note("val_loss", loss.item())
        self.note("val_metric", metric)

    def test_step(self, batch):
        dataset, test_loader = batch
        graph = dataset.data
        if hasattr(self.model, "inference"):
            pred = self.model.inference(graph.x, test_loader)
        else:
            pred = self.model(graph)
        pred = pred[graph.test_mask]
        y = graph.y[graph.test_mask]

        metric = self.evaluate(pred, y, metric="auto")
        self.note("test_loss", self.default_loss_fn(pred, y))
        self.note("test_metric", metric)

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
