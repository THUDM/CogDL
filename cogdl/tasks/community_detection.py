import copy
import random
import warnings
from collections import defaultdict

import networkx as nx
from networkx.algorithms import community
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score

from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.data import Dataset, InMemoryDataset
from cogdl.models import build_model

from . import BaseTask, register_task


@register_task("community_detection")
class CommunityDetection(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--num-shuffle", type=int, default=5)
        # fmt: on

    def __init__(self, args):
        super(CommunityDetection, self).__init__(args)
        dataset = build_dataset(args)
        self.data = dataset[0]
        if issubclass(dataset.__class__.__bases__[0], InMemoryDataset):
            self.num_nodes = self.data.y.shape[0]
            self.num_classes = dataset.num_classes
        else:
            self.num_nodes, self.num_classes = self.data.y.shape

        self.label = np.argmax(self.data.y, axis=1)
        self.model = build_model(args)
        self.hidden_size = args.hidden_size
        self.num_shuffle = args.num_shuffle
        self.is_weighted = self.data.edge_attr is not None

    def train(self):
        G = nx.Graph()
        if self.is_weighted:
            edges, weight = (
                self.data.edge_index.t().tolist(),
                self.data.edge_attr.tolist(),
            )
            G.add_weighted_edges_from(
                [(edges[i][0], edges[i][1], weight[0][i]) for i in range(len(edges))]
            )
        else:
            G.add_edges_from(self.data.edge_index.t().tolist())
        partition = community.greedy_modularity_communities(G)
        base_label = [0] * G.number_of_nodes()
        for i, node_set in enumerate(partition):
            for node in node_set:
                base_label[node] = i
        nmi_score = normalized_mutual_info_score(self.label, base_label)
        print("NMI score of greedy modularity optimize algorithm: ", nmi_score)
        
        embeddings = self.model.train(G)

        # Map node2id
        features_matrix = np.zeros((self.num_nodes, self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]
            
        return self._evaluate(features_matrix)

    def _evaluate(self, features_matrix):
        clusters = [30, 40, 50, 60, 70]
        all_results = defaultdict(list)
        for num_cluster in clusters:
            for _ in range(self.num_shuffle):
                model = KMeans(n_clusters=num_cluster).fit(features_matrix)
                # sc_score = silhouette_score(features_matrix, model.labels_, metric='euclidean')
                nmi_score = normalized_mutual_info_score(self.label, model.labels_)
                all_results[num_cluster].append(nmi_score)
                
        return dict(
            (
                f"normalized_mutual_info_score {num_cluster}",
                sum(all_results[num_cluster]) / len(all_results[num_cluster]),
            )
            for num_cluster in sorted(all_results.keys())
        )