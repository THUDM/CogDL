import torch
import torch.nn.functional as F

from .. import ModelWrapper
from cogdl.wrappers.tools.wrapper_utils import evaluate_clustering


class GAEModelWrapper(ModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-clusters", type=int, default=7)
        parser.add_argument("--cluster-method", type=str, default="kmeans", help="option: kmeans or spectral")
        parser.add_argument("--evaluation", type=str, default="full", help="option: full or NMI")
        # fmt: on

    def __init__(self, model, optimizer_cfg, num_clusters, cluster_method="kmeans", evaluation="full"):
        super(GAEModelWrapper, self).__init__()
        self.model = model
        self.num_clusters = num_clusters
        self.optimizer_cfg = optimizer_cfg
        self.cluster_method = cluster_method
        self.full = evaluation == "full"

    def train_step(self, subgraph):
        graph = subgraph
        loss = self.model.make_loss(graph, graph.adj_mx)
        return loss

    def test_step(self, subgraph):
        graph = subgraph
        features_matrix = self.model(graph)
        features_matrix = features_matrix.detach().cpu().numpy()
        return evaluate_clustering(
            features_matrix, graph.y, self.cluster_method, self.num_clusters, graph.num_nodes, self.full
        )

    def pre_stage(self, stage, data_w):
        if stage == 0:
            data = data_w.get_dataset().data
            adj_mx = torch.sparse_coo_tensor(
                torch.stack(data.edge_index),
                torch.ones(data.edge_index[0].shape[0]),
                torch.Size([data.x.shape[0], data.x.shape[0]]),
            ).to_dense()
            data.adj_mx = adj_mx

    def setup_optimizer(self):
        lr, wd = self.optimizer_cfg["lr"], self.optimizer_cfg["weight_decay"]
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
