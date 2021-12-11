import torch

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
        self.random_walker = RandomWalker()

    def train_step(self, batch):
        data = batch
        x = self.model(data)
        device = x.device

        self.random_walker.build_up(data.edge_index, data.x.shape[0])
        walk_res = self.random_walker.walk(
            start=torch.arange(0, x.shape[0]).to(device), walk_length=self.walk_length + 1
        )
        self.walk_res = torch.as_tensor(walk_res)[:, 1:]

        self.num_nodes = max(data.edge_index[0].max(), data.edge_index[1].max()).item() + 1

        self.negative_samples = torch.from_numpy(
            np.random.choice(self.num_nodes, (self.num_nodes, self.num_negative_samples))
        ).to(device)

        pos_loss = -torch.log(
            torch.sigmoid(torch.sum(x.unsqueeze(1).repeat(1, self.walk_length, 1) * x[self.walk_res], dim=-1))
        ).mean()
        neg_loss = -torch.log(
            torch.sigmoid(
                -torch.sum(x.unsqueeze(1).repeat(1, self.num_negative_samples, 1) * x[self.negative_samples], dim=-1)
            )
        ).mean()
        return (pos_loss + neg_loss) / 2

    def test_step(self, graph):
        with torch.no_grad():
            pred = self.model(graph)
        y = graph.y
        result = evaluate_node_embeddings_using_logreg(pred, y, graph.train_mask, graph.test_mask)
        self.note("test_acc", result)

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
