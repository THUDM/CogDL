# from cogdl import function as BF
# from cogdl.backend import BACKEND
# if BACKEND == 'jittor':
#     import jittor as tj
# elif BACKEND == 'torch':
#     import torch as tj

# from .. import ModelWrapper


# class SAGNModelWrapper(ModelWrapper):
#     def __init__(self, model, optimizer_cfg):
#         super(SAGNModelWrapper, self).__init__()
#         self.model = model
#         self.optimizer_cfg = optimizer_cfg

#     def train_step(self, batch):
#         batch_x, batch_y_emb, y = batch
#         pred = self.model(batch_x, batch_y_emb)
#         loss = self.default_loss_fn(pred, y)
#         return loss

#     def val_step(self, batch):
#         batch_x, batch_y_emb, y = batch
#         # print(batch_x.device, batch_y_emb.devce, y.device, next(self.parameters()).device)
#         pred = self.model(batch_x, batch_y_emb)

#         metric = self.evaluate(pred, y, metric="auto")

#         self.note("val_loss", self.default_loss_fn(pred, y))
#         self.note("val_metric", metric)

#     def test_step(self, batch):
#         batch_x, batch_y_emb, y = batch
#         pred = self.model(batch_x, batch_y_emb)

#         metric = self.evaluate(pred, y, metric="auto")

#         self.note("test_loss", self.default_loss_fn(pred, y))
#         self.note("test_metric", metric)

#     def pre_stage(self, stage, data_w):
#         device = BF.device(self.model.parameters()[0])
#         if stage == 0:
#             return None

#         self.model.eval()
#         preds = []

#         eval_loader = data_w.post_stage_wrapper()
#         with tj.no_grad():
#             for batch in eval_loader:
#                 batch_x, batch_y_emb, _ = data_w.pre_stage_transform(batch)
#                 batch_x = batch_x.to(device)
#                 batch_y_emb = batch_y_emb.to(device) if batch_y_emb is not None else batch_y_emb
#                 pred = self.model(batch_x, batch_y_emb)
#                 preds.append(pred.to("cpu"))
#         probs = BF.cat(preds, dim=0)
#         return probs

#     def setup_optimizer(self):
#         cfg = self.optimizer_cfg
#         return tj.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
from cogdl import function as BF
from cogdl.backend import BACKEND
if BACKEND == 'jittor':
    import jittor as tj
elif BACKEND == 'torch':
    import torch as tj

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
        if BACKEND == 'jittor':
            device=None
        else:
            device = BF.device(next(self.model.parameters())[0])
        if stage == 0:
            return None

        self.model.eval()
        preds = []

        eval_loader = data_w.post_stage_wrapper()
        with tj.no_grad():
            for batch in eval_loader:
                batch_x, batch_y_emb, _ = data_w.pre_stage_transform(batch)
                batch_x = BF.to(batch_x, device)
                batch_y_emb = BF.to(batch_y_emb, device) if batch_y_emb is not None else batch_y_emb
                pred = self.model(batch_x, batch_y_emb)
                preds.append(BF.to(pred, "cpu"))
        probs = BF.cat(preds, dim=0)
        return probs

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return tj.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
