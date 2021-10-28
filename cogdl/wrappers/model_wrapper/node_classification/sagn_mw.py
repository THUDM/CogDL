import torch

from .. import ModelWrapper


class SAGNModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg):
        super(SAGNModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg

    def train_step(self, batch):
        batch_x, batch_y_emb, y = batch
        pred = self.model(batch_x, batch_y_emb)
        loss = self.default_loss_fn(pred, y)
        return loss

    def val_step(self, batch):
        batch_x, batch_y_emb, y = batch
        # print(batch_x.device, batch_y_emb.devce, y.device, next(self.parameters()).device)
        pred = self.model(batch_x, batch_y_emb)

        metric = self.evaluate(pred, y, metric="auto")

        self.note("val_loss", self.default_loss_fn(pred, y))
        self.note("val_metric", metric)

    def test_step(self, batch):
        batch_x, batch_y_emb, y = batch
        pred = self.model(batch_x, batch_y_emb)

        metric = self.evaluate(pred, y, metric="auto")

        self.note("test_loss", self.default_loss_fn(pred, y))
        self.note("test_metric", metric)

    def pre_stage(self, stage, data_w):
        device = next(self.model.parameters()).device
        if stage == 0:
            return None

        self.model.eval()
        preds = []

        eval_loader = data_w.post_stage_wrapper()
        with torch.no_grad():
            for batch in eval_loader:
                batch_x, batch_y_emb, _ = data_w.pre_stage_transform(batch)
                batch_x = batch_x.to(device)
                batch_y_emb = batch_y_emb.to(device) if batch_y_emb is not None else batch_y_emb
                pred = self.model(batch_x, batch_y_emb)
                preds.append(pred.to("cpu"))
        probs = torch.cat(preds, dim=0)
        return probs

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
