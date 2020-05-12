import copy
import random
import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.data import Dataset, InMemoryDataset
from cogdl.models import build_model

from . import BaseTask, register_task
from queue import PriorityQueue as PQueue



@register_task("influence_maximization")
class InfluenceMaximization(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--num-seed", type=int, default=20)
        parser.add_argument("--num-simulation", type=int, default=1)
        parser.add_argument("--decay", type=float, default=0.5)

    
        # fmt: on

    def __init__(self, args):
        super(InfluenceMaximization, self).__init__(args)
        dataset = build_dataset(args)
        self.data = dataset[0]
        if issubclass(dataset.__class__.__bases__[0], InMemoryDataset):
            self.num_nodes = self.data.y.shape[0]
            self.num_classes = dataset.num_classes
        else:
            self.num_nodes, self.num_classes = self.data.y.shape

        self.label = np.argmax(self.data.y, axis=1)
        self.model = build_model(args)
        self.is_weighted = self.data.edge_attr is not None
        self.hidden_size = args.hidden_size
        self.num_simulation= args.num_simulation
        self.num_seed = args.num_seed
        self.decay = args.decay


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
        
        embeddings = self.model.train(G)

        # Map node2id
        features_matrix = np.zeros((self.num_nodes, self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]
            
        return self._evaluate(G, features_matrix)

    def _evaluate(self, G, features_matrix):
        thresholds = [0.7, 0.8]
        all_results = defaultdict(list)
        for threshold in thresholds:
            print("influence maximization with threshold :", threshold)
            influence_score, seed_list = self._influence_maximazation(G, features_matrix, threshold)
            all_results[threshold].append(influence_score)
                
        return dict(
            (
                f"influence score {threshold}",
                sum(all_results[threshold]) / len(all_results[threshold]),
            )
            for threshold in sorted(all_results.keys())
        )
        
    def _influence_maximazation(self, G, features_matrix, threshold):
        num_node = G.number_of_nodes()
        Q = PQueue()
        seed_list = []
        
        for node in G.nodes():
            act_num = self._simulation(G, features_matrix, node, threshold)
            Q.put((-1* act_num, [node, 1]))
            # print("Inside PriorityQueue: ", Q.queue) 
        
        total_num = 0.0
        for i in range(self.num_seed):
            while not Q.empty():
                tp = Q.get()
                act_num, node, k = -tp[0], tp[1][0], tp[1][1]
                if k == i:
                    total_num += act_num
                    seed_list.append(node)
                    break
                else:
                    act_num = self._simulation(G, features_matrix, node, threshold)
                    Q.put((-1*(act_num - total_num), [node, i]))
        
        return total_num / num_node, seed_list
    
    
    def _simulation(self, G, features_matrix, seed, threshold):
        res = 0
        num_node = G.number_of_nodes()
        for _ in range(self.num_simulation):
            active_list, active_flag = [], [False] * num_node  
            active_flag[seed] = True
            active_list.append(seed)
            
            i = 0
            while i < len(active_list):
                v = active_list[i]
                for target in list(G.neighbors(v)):
                    if active_flag[target] is True: continue 
                    vec1, vec2 = features_matrix[v], features_matrix[target]
                    # prob = 1.0 / G.degree(target)
                    prob = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    # if np.random.rand() <= prob  * self.decay:
                    #     active_flag[target] = True
                    #     active_list.append(target)
                    if prob >= threshold:
                        active_flag[target] = True
                        active_list.append(target)
                i += 1
            res += len(active_list) - 1
            # print("number of influence node", len(active_list))

        return res / self.num_simulation