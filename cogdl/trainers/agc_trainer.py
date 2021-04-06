import torch
import torch.nn
import torch.sparse
import numpy as np
from sklearn.cluster import SpectralClustering

from cogdl.utils import spmm
from .base_trainer import BaseTrainer


class AGCTrainer(BaseTrainer):
    def __init__(self, args):
        self.num_clusters = args.num_clusters
        self.max_iter = args.max_iter
        self.device = args.device_id[0] if not args.cpu else "cpu"

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def fit(self, model, data):
        model = model.to(self.device)
        data.to(self.device)
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
                model.features_matrix = pre_feat
                model.k = t - 1
                return model.cpu()
            pre_intra = intra
            pre_feat = w
        model.features_matrix = w
        model.k = t - 1
        return model.cpu()

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
