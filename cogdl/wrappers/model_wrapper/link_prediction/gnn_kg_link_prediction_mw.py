import torch
import torch.nn as nn

from .. import ModelWrapper
from cogdl.utils.link_prediction_utils import cal_mrr, DistMultLayer, ConvELayer


class GNNKGLinkPredictionModelWrapper(ModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--score-func", type=str, default="distmult")
        # fmt: on

    def __init__(self, model, optimizer_cfg, score_func):
        super(GNNKGLinkPredictionModelWrapper, self).__init__()

        self.model = model
        self.optimizer_cfg = optimizer_cfg
        hidden_size = optimizer_cfg["hidden_size"]

        self.score_func = score_func
        if score_func == "distmult":
            self.scoring = DistMultLayer()
        elif score_func == "conve":
            self.scoring = ConvELayer(hidden_size)
        else:
            raise NotImplementedError

    def train_step(self, subgraph):
        graph = subgraph
        mask = graph.train_mask
        edge_index = torch.stack(graph.edge_index)
        edge_index, edge_types = edge_index[:, mask], graph.edge_attr[mask]

        with graph.local_graph():
            graph.edge_index = edge_index
            graph.edge_attr = edge_types
            loss = self.model.loss(graph, self.scoring)
        return loss

    def val_step(self, subgraph):
        train_mask = subgraph.train_mask
        eval_mask = subgraph.val_mask
        return self.eval_step(subgraph, train_mask, eval_mask)

    def test_step(self, subgraph):
        infer_mask = subgraph.train_mask | subgraph.val_mask
        eval_mask = subgraph.test_mask
        return self.eval_step(subgraph, infer_mask, eval_mask)

    def eval_step(self, graph, mask1, mask2):
        row, col = graph.edge_index
        edge_types = graph.edge_attr

        with graph.local_graph():
            graph.edge_index = (row[mask1], col[mask1])
            graph.edge_attr = edge_types[mask1]
            output, rel_weight = self.model.predict(graph)

        mrr, hits = cal_mrr(
            output,
            rel_weight,
            (row[mask2], col[mask2]),
            edge_types[mask2],
            scoring=self.scoring,
            protocol="raw",
            batch_size=500,
            hits=[1, 3, 10],
        )

        return dict(mrr=mrr, hits1=hits[0], hits3=hits[1], hits10=hits[2])

    def setup_optimizer(self):
        lr, weight_decay = self.optimizer_cfg["lr"], self.optimizer_cfg["weight_decay"]
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def set_early_stopping(self):
        return "mrr", ">"
