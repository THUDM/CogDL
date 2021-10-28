import torch
from cogdl.wrappers.model_wrapper import ModelWrapper


class HeterogeneousGNNModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg):
        super(HeterogeneousGNNModelWrapper, self).__init__()
        self.optimizer_cfg = optimizer_cfg
        self.model = model

    def train_step(self, batch):
        graph = batch.data
        pred = self.model(graph)
        train_mask = graph.train_node
        loss = self.default_loss_fn(pred[train_mask], graph.y[train_mask])
        return loss

    def val_step(self, batch):
        graph = batch.data
        pred = self.model(graph)
        val_mask = graph.valid_node
        loss = self.default_loss_fn(pred[val_mask], graph.y[val_mask])
        metric = self.evaluate(pred[val_mask], graph.y[val_mask], metric="auto")
        self.note("val_loss", loss.item())
        self.note("val_metric", metric)

    def test_step(self, batch):
        graph = batch.data
        pred = self.model(graph)
        test_mask = graph.test_node
        loss = self.default_loss_fn(pred[test_mask], graph.y[test_mask])
        metric = self.evaluate(pred[test_mask], graph.y[test_mask], metric="auto")
        self.note("test_loss", loss.item())
        self.note("test_metric", metric)

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        if hasattr(self.model, "get_optimizer"):
            model_spec_optim = self.model.get_optimizer(cfg)
            if model_spec_optim is not None:
                return model_spec_optim
        return torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
