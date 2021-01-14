import argparse
from typing import Dict
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn.functional as F
from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task

@register_task("attributed_graph_clustering")
class AttributedGraphClustering(BaseTask):
    """Attributed graph clustring task."""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-clusters", type=int, default=7)
        parser.add_argument("--cluster-method", type=str, default="kmeans")
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--model-type", type=str, default="content")
        # fmt: on

    def __init__(
        self,
        args,
        dataset=None,
        _=None,
    ):
        super(AttributedGraphClustering, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        if dataset is None:
            dataset = build_dataset(args)
        self.dataset = dataset
        self.data = dataset[0]
        self.num_nodes = self.data.y.shape[0]

        if args.model == "prone":
            self.hidden_size = args.hidden_size = args.num_features = 13
        else:
            self.hidden_size = args.hidden_size = args.hidden_size
            args.num_features = dataset.num_features
        self.model = build_model(args)
        
        self.num_clusters = args.num_clusters
        if args.cluster_method not in ["kmeans", "spectral"]:
            raise Exception("cluster method must be kmeans or spectral")
        if args.model_type not in ["content", "spectral", "both"]:
            raise Exception("model type must be content, spectral or both")
        self.cluster_method = args.cluster_method
        self.model_type = args.model_type
        self.is_weighted = self.data.edge_attr is not None
        self.enhance = args.enhance

    def train(self) -> Dict[str, float]:
        if self.model_type == "content":
            features_matrix = self.data.x
        elif self.model_type == "spectral":
            G = nx.Graph()
            if self.is_weighted:
                edges, weight = (
                    self.data.edge_index.t().tolist(),
                    self.data.edge_attr.tolist(),
                )
                G.add_weighted_edges_from([(edges[i][0], edges[i][1], weight[i][0]) for i in range(len(edges))])
            else:
                G.add_edges_from(self.data.edge_index.t().tolist())
            embeddings = self.model.train(G)
            if self.enhance is not None:
                embeddings = self.enhance_emb(G, embeddings)
            # Map node2id
            features_matrix = np.zeros((self.num_nodes, self.hidden_size))
            for vid, node in enumerate(G.nodes()):
                features_matrix[node] = embeddings[vid]
            features_matrix = torch.tensor(features_matrix)
            features_matrix = F.normalize(features_matrix, p=2, dim=1)
            #features_matrix = self.momentum * node_attr + (1 - self.momentum) * features_matrix
        else:
            trainer = self.model.get_trainer(AttributedGraphClustering, self.args)(self.args)
            self.model = trainer.fit(self.model, self.data)
            features_matrix = self.model.get_features(self.data)
            
        features_matrix = features_matrix.cpu().numpy()
        print("Clustering...")
        if self.cluster_method == "kmeans":
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(features_matrix)
            clusters = kmeans.labels_
        else:
            # features_matrix = np.dot(features_matrix, features_matrix.transpose())
            # features_matrix = 0.5 * (np.abs(features_matrix) + np.abs(features_matrix.transpose()))
            clustering = SpectralClustering(n_clusters=self.num_clusters, assign_labels="discretize", random_state=0).fit(
                features_matrix
            )
            clusters = clustering.labels_
        return self.__evaluate(clusters)

    def __evaluate(self, clusters) -> Dict[str, float]:
        print("Evaluating...")
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        truth = self.data.y.cpu().numpy()
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
        _ = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        print("TP", TP, "FP", FP, "TN", TN, "FN", FN)
        micro_f1 = 2 * (precision * recall) / (precision + recall)
        return dict(Accuracy=precision, NMI=normalized_mutual_info_score(clusters, truth), Micro_F1=micro_f1)
