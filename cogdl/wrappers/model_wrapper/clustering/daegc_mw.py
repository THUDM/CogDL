import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

from .. import ModelWrapper
from cogdl.wrappers.tools.wrapper_utils import evaluate_clustering


class DAEGCModelWrapper(ModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-clusters", type=int, default=7)
        parser.add_argument("--cluster-method", type=str, default="kmeans", help="option: kmeans or spectral")
        parser.add_argument("--evaluation", type=str, default="full", help="option: full or NMI")
        parser.add_argument("--T", type=int, default=5)
        # fmt: on

    def __init__(self, model, optimizer_cfg, num_clusters, cluster_method="kmeans", evaluation="full", T=5):
        super(DAEGCModelWrapper, self).__init__()
        self.model = model
        self.num_clusters = num_clusters
        self.optimizer_cfg = optimizer_cfg
        self.cluster_method = cluster_method
        self.full = evaluation == "full"
        self.t = T

        self.stage = 0
        self.count = 0

    def train_step(self, subgraph):
        graph = subgraph
        if self.stage == 0:
            z = self.model(graph)
            loss = self.recon_loss(z, graph.adj_mx)
        else:
            cluster_center = self.model.get_cluster_center()
            z = self.model(graph)
            Q = self.getQ(z, cluster_center)
            self.count += 1
            if self.count % self.t == 0:
                P = self.getP(Q).detach()
            loss = self.recon_loss(z, graph.adj_mx) + self.gamma * self.cluster_loss(P, Q)
        return loss

    def test_step(self, subgraph):
        graph = subgraph
        features_matrix = self.model(graph)
        features_matrix = features_matrix.detach().cpu().numpy()
        return evaluate_clustering(
            features_matrix, graph.y, self.cluster_method, self.num_clusters, graph.num_nodes, self.full
        )

    def recon_loss(self, z, adj):
        return F.binary_cross_entropy(F.softmax(torch.mm(z, z.t())), adj, reduction="sum")

    def cluster_loss(self, P, Q):
        return torch.nn.KLDivLoss(reduce=True, size_average=False)(P.log(), Q)

    def setup_optimizer(self):
        lr, wd = self.optimizer_cfg["lr"], self.optimizer_cfg["weight_decay"]
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def pre_stage(self, stage, data_w):
        self.stage = stage
        if stage == 0:
            data = data_w.get_dataset().data
            data.add_remaining_self_loops()

            data.store("edge_index")

            data.adj_mx = torch.sparse_coo_tensor(
                torch.stack(data.edge_index),
                torch.ones(data.edge_index[0].shape[0]),
                torch.Size([data.x.shape[0], data.x.shape[0]]),
            ).to_dense()
            edge_index_2hop = data.edge_index
            data.edge_index = edge_index_2hop

    def post_stage(self, stage, data_w):
        if stage == 0:
            data = data_w.get_dataset().data
            data.restore("edge_index")
            data.to(self.device)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(self.model(data).detach().cpu().numpy())
            self.model.set_cluster_center(torch.tensor(kmeans.cluster_centers_, device=self.device))

    def getQ(self, z, cluster_center):
        Q = None
        for i in range(z.shape[0]):
            dis = torch.sum((z[i].repeat(self.num_clusters, 1) - cluster_center) ** 2, dim=1)
            t = 1 / (1 + dis)
            t = t / torch.sum(t)
            if Q is None:
                Q = t.clone().unsqueeze(0)
            else:
                Q = torch.cat((Q, t.unsqueeze(0)), 0)
        return Q

    def getP(self, Q):
        P = torch.sum(Q, dim=0).repeat(Q.shape[0], 1)
        P = Q ** 2 / P
        P = P / (torch.ones(1, self.num_clusters, device=self.device) * torch.sum(P, dim=1).unsqueeze(-1))
        # print("P=", P)
        return P
