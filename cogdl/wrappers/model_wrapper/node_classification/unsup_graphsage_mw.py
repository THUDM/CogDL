from cogdl.backend import BACKEND

if BACKEND == "jittor":
    import jittor as tj
elif BACKEND == "torch":
    import torch as tj
import numpy as np
from cogdl.utils import RandomWalker
from cogdl.wrappers.tools.wrapper_utils import evaluate_node_embeddings_using_logreg
from .. import UnsupervisedModelWrapper


class UnsupGraphSAGEModelWrapper(UnsupervisedModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--walk-length", type=int, default=10)
        parser.add_argument("--negative-samples", type=int, default=30)
        # fmt: on

    def __init__(self, model, optimizer_cfg, walk_length, negative_samples):
        super(UnsupGraphSAGEModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.walk_length = walk_length
        self.num_negative_samples = negative_samples

    def train_step(self, batch):
        x_src, adjs = batch
        out = self.model(x_src, adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = tj.log(tj.sigmoid((out * pos_out).sum(-1)).mean())
        neg_loss = tj.log(tj.sigmoid(-(out * neg_out).sum(-1)).mean())
        loss = -pos_loss - neg_loss
        return loss

    def test_step(self, batch):
        dataset, test_loader = batch
        graph = dataset.data
        if hasattr(self.model, "inference"):
            pred = self.model.inference(graph.x, test_loader)
        else:
            pred = self.model(graph)
        pred = pred.split(pred.size(0) // 3, dim=0)[0]
        pred = pred[graph.test_mask]
        y = graph.y[graph.test_mask]

        metric = self.evaluate(pred, y, metric="auto")
        self.note("test_loss", self.default_loss_fn(pred, y))
        self.note("test_metric", metric)

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return tj.optim.Adam(self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
