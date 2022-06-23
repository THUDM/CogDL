import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.data import Graph
from .. import UnsupervisedModelWrapper
from cogdl.wrappers.tools.wrapper_utils import evaluate_node_embeddings_using_logreg
from cogdl.utils import dropout_adj, dropout_features


class GRACEModelWrapper(UnsupervisedModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--tau", type=float, default=0.5)
        parser.add_argument("--drop-feature-rates", type=float, nargs="+", default=[0.3, 0.4])
        parser.add_argument("--drop-edge-rates", type=float, nargs="+", default=[0.2, 0.4])
        parser.add_argument("--batch-fwd", type=int, default=-1)
        parser.add_argument("--proj-hidden-size", type=int, default=128)
        # fmt: on

    def __init__(self, model, optimizer_cfg, tau, drop_feature_rates, drop_edge_rates, batch_fwd, proj_hidden_size):
        super(GRACEModelWrapper, self).__init__()
        self.tau = tau
        self.drop_feature_rates = drop_feature_rates
        self.drop_edge_rates = drop_edge_rates
        self.batch_size = batch_fwd

        self.model = model
        hidden_size = optimizer_cfg["hidden_size"]
        self.project_head = nn.Sequential(
            nn.Linear(hidden_size, proj_hidden_size), nn.ELU(), nn.Linear(proj_hidden_size, hidden_size)
        )
        self.optimizer_cfg = optimizer_cfg

    def train_step(self, subgraph):
        graph = subgraph
        z1 = self.prop(graph, graph.x, self.drop_feature_rates[0], self.drop_edge_rates[0])
        z2 = self.prop(graph, graph.x, self.drop_feature_rates[1], self.drop_edge_rates[1])

        z1 = self.project_head(z1)
        z2 = self.project_head(z2)

        if self.batch_size > 0:
            return 0.5 * (self.batched_loss(z1, z2, self.batch_size) + self.batched_loss(z2, z1, self.batch_size))
        else:
            return 0.5 * (self.contrastive_loss(z1, z2) + self.contrastive_loss(z2, z1))

    def test_step(self, graph):
        with torch.no_grad():
            pred = self.model(graph)
        y = graph.y
        result = evaluate_node_embeddings_using_logreg(pred, y, graph.train_mask, graph.test_mask)
        self.note("test_acc", result)

    def prop(
        self, graph: Graph, x: torch.Tensor, drop_feature_rate: float = 0.0, drop_edge_rate: float = 0.0,
    ):
        x = dropout_features(x, drop_feature_rate)
        with graph.local_graph():
            graph.edge_index, graph.edge_weight = dropout_adj(graph.edge_index, graph.edge_weight, drop_edge_rate)
            return self.model.forward(graph, x)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        def score_func(emb1, emb2):
            scores = torch.matmul(emb1, emb2.t())
            scores = torch.exp(scores / self.tau)
            return scores

        intro_scores = score_func(z1, z1)
        inter_scores = score_func(z1, z2)

        _loss = -torch.log(intro_scores.diag() / (intro_scores.sum(1) - intro_scores.diag() + inter_scores.sum(1)))
        return torch.mean(_loss)

    def batched_loss(
        self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int,
    ):
        num_nodes = z1.shape[0]
        num_batches = (num_nodes - 1) // batch_size + 1

        losses = []
        indices = torch.arange(num_nodes).to(z1.device)
        for i in range(num_batches):
            train_indices = indices[i * batch_size : (i + 1) * batch_size]
            _loss = self.contrastive_loss(z1[train_indices], z2)
            losses.append(_loss)
        return sum(losses) / len(losses)

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
