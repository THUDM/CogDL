import torch
import numpy as np
from sklearn.cluster import SpectralClustering

from cogdl.utils import spmm
from .. import BaseModel


class AGC(BaseModel):
    r"""The AGC model from the `"Attributed Graph Clustering via Adaptive Graph Convolution"
    <https://arxiv.org/abs/1906.01210>`_ paper

    Args:
        num_clusters (int) : Number of clusters.
        max_iter     (int) : Max iteration to increase k
    """

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-clusters", type=int, default=7)
        parser.add_argument("--max-iter", type=int, default=10)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_clusters, args.max_iter, args.cpu)

    def __init__(self, num_clusters, max_iter, cpu):
        super(AGC, self).__init__()

        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.cpu = cpu

    def forward(self, data):
        device = "cuda" if torch.cuda.is_available() and not self.cpu else "cpu"
        data = data.to(device)
        self.num_nodes = data.x.shape[0]
        graph = data
        graph.add_remaining_self_loops()

        graph.sym_norm()
        graph.edge_weight = data.edge_weight * 0.5

        pre_intra = 1e27
        pre_feat = None
        for t in range(1, self.max_iter + 1):
            x = data.x
            for i in range(t):
                x = spmm(graph, x)
            k = torch.mm(x, x.t())
            w = (torch.abs(k) + torch.abs(k.t())) / 2
            clustering = SpectralClustering(
                n_clusters=self.num_clusters, assign_labels="discretize", random_state=0
            ).fit(w.detach().cpu())
            clusters = clustering.labels_
            intra = self.compute_intra(x.cpu().numpy(), clusters)
            print("iter #%d, intra = %.4lf" % (t, intra))
            if intra > pre_intra:
                features_matrix = pre_feat
                return features_matrix
            pre_intra = intra
            pre_feat = w
        features_matrix = w
        return features_matrix.cpu()

    def compute_intra(self, x, clusters):
        num_nodes = x.shape[0]
        intra = np.zeros(self.num_clusters)
        num_per_cluster = np.zeros(self.num_clusters)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if clusters[i] == clusters[j]:
                    intra[clusters[i]] += np.sum((x[i] - x[j]) ** 2) ** 0.5
                    num_per_cluster[clusters[i]] += 1
        intra = np.array(list(filter(lambda x: x > 0, intra)))
        num_per_cluster = np.array(list(filter(lambda x: x > 0, num_per_cluster)))
        return np.mean(intra / num_per_cluster)
