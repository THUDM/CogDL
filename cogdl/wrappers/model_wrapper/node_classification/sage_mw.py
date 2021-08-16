import torch

from .. import ModelWrapper, register_model_wrapper
from cogdl.wrappers.tools.wrapper_utils import pre_evaluation_index


@register_model_wrapper("sagn_mw")
class SagnModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg):
        super(SagnModelWrapper, self).__init__()
        self.model = model

    def train_step(self, batch):
        batch_x, batch_y_emb, y = batch
        pred = self.model(batch_x, batch_y_emb)
        loss = self.default_loss_fn(pred, y[batch])
        return loss

    def val_step(self, batch):
        batch_x, batch_y_emb, y = batch
        pred = self.model(batch_x, batch_y_emb)

        self.note("val_loss", self.default_loss_fn(pred, y))
        self.note("val_eval_index", pre_evaluation_index(pred, y))

    def test_step(self, batch):
        pass

    def post_stage(self, stage, data_w):
        device = next(self.model.parameters()).device

        self.model.eval()
        preds = []

        eval_loader = data_w.post_stage_wrapper()
        with torch.no_grad():
            for batch in eval_loader:
                batch_x, batch_y_emb = data_w.post_stage_transform(batch)
                batch_x = batch_x.to(device)
                batch_y_emb = batch_y_emb.to(device) if batch_y_emb is not None else batch_y_emb
                pred = self.model(batch_x, batch_y_emb)
                preds.append(pred.to(self.data_device))
        probs = torch.cat(preds, dim=0)
        return probs
