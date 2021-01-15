import torch
import torch.nn
import torch.nn.functional as F
import torch.sparse
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score

from cogdl.utils import add_remaining_self_loops, spmm
from .base_trainer import BaseTrainer


class AGCTrainer(BaseTrainer):
    def __init__(self, args):
        self.num_clusters = args.num_clusters
        self.max_iter = args.max_iter
        self.device = args.device_id[0] if not args.cpu else "cpu"

    @staticmethod
    def build_trainer_from_args(args):
        pass

    def fit(self, model, data):
        model = model.to(self.device)
        data.apply(lambda x: x.to(self.device))
        self.num_nodes = data.x.shape[0]

        adj = data.edge_index
        adj_values = torch.ones(adj.shape[1]).to(self.device)
        deg = spmm(adj, adj_values, torch.ones(data.x.shape[0], 1).to(self.device)).squeeze()
        deg_sqrt = deg.pow(-1 / 2)
        adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
        adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, data.x.shape[0])
        adj_values = 0.5 * adj_values
        adj = torch.sparse_coo_tensor(adj, adj_values, torch.Size([data.x.shape[0], data.x.shape[0]])).to(self.device)

        pre_intra = 1e27
        pre_feat = None
        for t in range(1, self.max_iter + 1):
            x = data.x
            for i in range(t):
                x = torch.spmm(adj, x)
            k = torch.mm(x, x.t())
            w = (torch.abs(k) + torch.abs(k.t())) / 2
            clustering = SpectralClustering(
                n_clusters=self.num_clusters, assign_labels="discretize", random_state=0
            ).fit(w.detach().cpu())
            clusters = clustering.labels_
            intra = self.compute_intra(x.cpu().numpy(), clusters)
            print("iter #%d, intra = %.4lf" % (t, intra))
            # self.evaluate(clusters, data.y.cpu().numpy())
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
        # print(intra / num_per_cluster)
        return np.mean(intra / num_per_cluster)

    def evaluate(self, clusters, truth):
        print("Evaluating...")
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if clusters[i] == clusters[j] and truth[i] == truth[j]:
                    TP += 1
                if clusters[i] != clusters[j] and truth[i] == truth[j]:
                    FP += 1
                if clusters[i] == clusters[j] and truth[i] != truth[j]:
                    FN += 1
                if clusters[i] != clusters[j] and truth[i] != truth[j]:
                    TN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        print("TP", TP, "FP", FP, "TN", TN, "FN", FN)
        micro_f1 = 2 * (precision * recall) / (precision + recall)
        print(
            "Accuracy = ", precision, "NMI = ", normalized_mutual_info_score(clusters, truth), "Micro_F1 = ", micro_f1
        )
