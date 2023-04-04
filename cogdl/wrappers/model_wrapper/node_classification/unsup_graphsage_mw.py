import torch

import numpy as np
from cogdl.wrappers.tools.wrapper_utils import evaluate_node_embeddings_using_liblinear
from .. import UnsupervisedModelWrapper
from torch.nn import functional as F

class UnsupGraphSAGEModelWrapper(UnsupervisedModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-shuffle", type=int, default=1)
        parser.add_argument("--training-percents", default=[0.2], type=float, nargs="+")
        parser.add_argument("--walk-length", type=int, default=10)
        parser.add_argument("--negative-samples", type=int, default=30)
        # fmt: on

    def __init__(self, model, optimizer_cfg, walk_length, negative_samples, num_shuffle=1, training_percents=[0.1]):
        super(UnsupGraphSAGEModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.walk_length = walk_length
        self.num_negative_samples = negative_samples
        self.num_shuffle = num_shuffle
        self.training_percents = training_percents


    def train_step(self, batch):
        x_src, adjs = batch
        out = self.model(x_src,adjs)  
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        return loss

    def test_step(self, graph):
        dataset, test_loader = graph
        graph = dataset.data
        with torch.no_grad():
            if hasattr(self.model, "inference"):
                pred = self.model.inference(graph.x, test_loader)
            else:
                pred = self.model(graph)
        if len(graph.y.shape) > 1:
            self.label_matrix = graph.y.numpy()
        else:
            self.label_matrix = np.zeros((graph.num_nodes, graph.num_classes), dtype=int)
            self.label_matrix[range(graph.num_nodes), graph.y.numpy()] = 1
        return evaluate_node_embeddings_using_liblinear(pred, self.label_matrix, self.num_shuffle, self.training_percents)


    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])