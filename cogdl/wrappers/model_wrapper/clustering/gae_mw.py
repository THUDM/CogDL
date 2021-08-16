import torch
import torch.nn.functional as F

from .. import register_model_wrapper, ModelWrapper
from cogdl.wrappers.tools.wrapper_utils import evaluate_clustering


@register_model_wrapper("gat_mw")
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
        mean, log_var = self.model.encode(graph)
        z = self.reparameterization(mean, log_var)
        mat = self.model.decode(z)
        recon_loss = F.binary_cross_entropy(mat, graph.adj_mx, reduction="sum")
        var = torch.exp(log_var)
        kl_loss = 0.5 * torch.mean(torch.sum(mean * mean + var - log_var - 1, dim=1))
        # print("recon_loss = %.3f, kl_loss = %.3f" % (recon_loss, kl_loss))
        return recon_loss + kl_loss

    def evaluate(self, dataset):
        data = dataset.data
        features_matrix = self.model(data)
        features_matrix = features_matrix.cpu().numpy()
        return evaluate_clustering(
            features_matrix, data.y, self.cluster_method, self.num_clusters, data.num_nodes, self.full
        )

    @staticmethod
    def reparameterization(mean, log_var):
        sigma = torch.exp(log_var)
        z = mean + torch.randn_like(log_var) * sigma
        return z

    def pre_stage(self, stage, data_w):
        if stage == 0:
            data = data_w.get_dataset().data
            adj_mx = torch.sparse_coo_tensor(
                torch.stack(data.edge_index),
                torch.ones(data.edge_index[0].shape[0]),
                torch.Size([data.x.shape[0], data.x.shape[0]]),
            ).to_dense()
            data.adj_mx = adj_mx
