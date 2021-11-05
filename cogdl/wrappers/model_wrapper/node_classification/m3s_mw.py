import copy

import torch
import numpy as np
from sklearn.cluster import KMeans

from cogdl.wrappers.data_wrapper import DataWrapper

from .node_classification_mw import NodeClfModelWrapper


class M3SModelWrapper(NodeClfModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--n-cluster", type=int, default=10)
        parser.add_argument("--num-new-labels", type=int, default=10)
        # fmt: on

    def __init__(self, model, optimizer_cfg, n_cluster, num_new_labels):
        super(M3SModelWrapper, self).__init__(model, optimizer_cfg)
        self.model = model
        self.num_clusters = n_cluster
        self.hidden_size = optimizer_cfg["hidden_size"]
        self.num_new_labels = num_new_labels
        self.optimizer_cfg = optimizer_cfg

    def pre_stage(self, stage, data_w: DataWrapper):
        if stage > 0:
            graph = data_w.get_dataset().data
            graph.store("y")

            y = copy.deepcopy(graph.y)

            num_classes = graph.num_classes
            num_nodes = graph.num_nodes

            with torch.no_grad():
                emb = self.model.embed(graph)

            confidence_ranking = np.zeros([num_classes, num_nodes], dtype=int)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(emb)
            clusters = kmeans.labels_

            # Compute centroids μ_m of each class m in labeled data and v_l of each cluster l in unlabeled data.
            labeled_centroid = np.zeros([num_classes, self.hidden_size])
            unlabeled_centroid = np.zeros([self.num_clusters, self.hidden_size])
            for i in range(num_nodes):
                if graph.train_mask[i]:
                    labeled_centroid[y[i]] += emb[i]
                else:
                    unlabeled_centroid[clusters[i]] += emb[i]

            # Align labels for each cluster
            align = np.zeros(self.num_clusters, dtype=int)
            for i in range(self.num_clusters):
                for j in range(num_classes):
                    if np.linalg.norm(unlabeled_centroid[i] - labeled_centroid[j]) < np.linalg.norm(
                        unlabeled_centroid[i] - labeled_centroid[align[i]]
                    ):
                        align[i] = j

            # Add new labels
            for i in range(num_classes):
                t = self.num_new_labels
                for j in range(num_nodes):
                    idx = confidence_ranking[i][j]
                    if not graph.train_mask[idx]:
                        if t <= 0:
                            break
                        t -= 1
                        if align[clusters[idx]] == i:
                            graph.train_mask[idx] = True
                            y[idx] = i
            return y
