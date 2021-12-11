import torch
import torch.nn as nn

from .. import UnsupervisedModelWrapper
from cogdl.wrappers.tools.wrapper_utils import evaluate_node_embeddings_using_logreg


class MVGRLModelWrapper(UnsupervisedModelWrapper):
    def __init__(self, model, optimizer_cfg):
        super(MVGRLModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.loss_f = nn.BCEWithLogitsLoss()

    def train_step(self, subgraph):
        graph = subgraph
        logits = self.model(graph)
        labels = torch.zeros_like(logits)
        num_outs = logits.shape[1]
        labels[:, : num_outs // 2] = 1
        loss = self.loss_f(logits, labels)
        return loss

    def test_step(self, graph):
        with torch.no_grad():
            pred = self.model(graph)
        y = graph.y
        result = evaluate_node_embeddings_using_logreg(pred, y, graph.train_mask, graph.test_mask)
        self.note("test_acc", result)

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
