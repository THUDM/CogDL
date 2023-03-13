import networkx as nx

import torch
import torch.nn.functional as F
from .. import BaseModel
from cogdl.layers import GATLayer


class DAEGC(BaseModel):
    r"""The DAEGC model from the `"Attributed Graph Clustering: A Deep Attentional Embedding Approach"
    <https://arxiv.org/abs/1906.06532>`_ paper

    Args:
        num_clusters (int) : Number of clusters.
        T (int) : Number of iterations to recalculate P and Q
        gamma (float) : Hyperparameter that controls two parts of the loss.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--hidden-size", type=int, default=256)
        parser.add_argument("--embedding-size", type=int, default=16)
        parser.add_argument("--num-heads", type=int, default=1)
        parser.add_argument("--dropout", type=float, default=0)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.001)
        # parser.add_argument("--T", type=int, default=5)
        parser.add_argument("--gamma", type=float, default=10)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features, args.hidden_size, args.embedding_size, args.num_heads, args.dropout, args.num_clusters
        )

    def __init__(self, num_features, hidden_size, embedding_size, num_heads, dropout, num_clusters):
        super(DAEGC, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.num_clusters = num_clusters
        self.att1 = GATLayer(num_features, hidden_size, attn_drop=dropout, alpha=0.2, nhead=num_heads)
        self.att2 = GATLayer(hidden_size * num_heads, embedding_size, attn_drop=dropout, alpha=0.2, nhead=1)
        self.cluster_center = torch.nn.Parameter(torch.FloatTensor(self.num_clusters))

    def set_cluster_center(self, center):
        self.cluster_center.data = center

    def get_cluster_center(self):
        return self.cluster_center.data.detach()

    def forward(self, graph):
        x = graph.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.att1(graph, x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.att2(graph, x))
        return F.normalize(x, p=2, dim=1)

    def get_2hop(self, edge_index):
        r"""add 2-hop neighbors as new edges"""
        G = nx.Graph()
        edge_index = torch.stack(edge_index)
        G.add_edges_from(edge_index.t().tolist())
        H = nx.Graph()
        for i in range(G.number_of_nodes()):
            layers = dict(nx.bfs_successors(G, source=i, depth_limit=2))
            for succ in layers:
                for idx in layers[succ]:
                    H.add_edge(i, idx)
        return torch.tensor(list(H.edges())).t()

    def get_features(self, data):
        return self.forward(data).detach()

    def recon_loss(self, z, adj):
        # print(torch.mm(z, z.t()), adj)
        return F.binary_cross_entropy(F.sigmoid(torch.mm(z, z.t())), adj, reduction="sum")
