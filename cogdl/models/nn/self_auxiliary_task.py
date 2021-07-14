import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import networkx as nx
import random
from tqdm import tqdm

from .. import register_model
from cogdl.data import Dataset
from cogdl.utils import dropout_adj
from cogdl.models.nn.gcn import TKipfGCN
from cogdl.models.self_supervised_model import SelfSupervisedGenerativeModel
from cogdl.trainers.self_supervised_trainer import SelfSupervisedJointTrainer


class SSLTask:
    def __init__(self, graph, device):
        self.graph = graph
        self.edge_index = graph.edge_index_train if hasattr(graph, "edge_index_train") else graph.edge_index
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.num_edges
        self.features = graph.x
        self.device = device
        self.cached_edges = None

    def transform_data(self):
        raise NotImplementedError

    def make_loss(self, embeddings):
        raise NotImplementedError


class EdgeMask(SSLTask):
    def __init__(self, graph, hidden_size, mask_ratio, device):
        super().__init__(graph, device)
        self.linear = nn.Linear(hidden_size, 2).to(device)
        self.mask_ratio = mask_ratio

    def transform_data(self):
        if self.cached_edges is None:
            row, col = self.edge_index
            edges = torch.stack([row, col])
            perm = np.random.permutation(self.num_edges)
            preserve_nnz = int(self.num_edges * (1 - self.mask_ratio))
            masked = perm[preserve_nnz:]
            preserved = perm[:preserve_nnz]
            self.masked_edges = edges[:, masked]
            self.cached_edges = edges[:, preserved]
            mask_num = len(masked)
            self.neg_edges = self.neg_sample(mask_num)
            self.pseudo_labels = torch.cat([torch.ones(mask_num), torch.zeros(mask_num)]).long().to(self.device)
            self.node_pairs = torch.cat([self.masked_edges, self.neg_edges], 1).to(self.device)
            self.graph.edge_index = self.cached_edges

        return self.graph.to(self.device)

    def make_loss(self, embeddings):
        embeddings = self.linear(torch.abs(embeddings[self.node_pairs[0]] - embeddings[self.node_pairs[1]]))
        output = F.log_softmax(embeddings, dim=1)
        return F.nll_loss(output, self.pseudo_labels)

    def neg_sample(self, edge_num):
        edges = torch.stack(self.edge_index).t().cpu().numpy()
        exclude = set([(_[0], _[1]) for _ in list(edges)])
        itr = self.sample(exclude)
        sampled = [next(itr) for _ in range(edge_num)]
        return torch.tensor(sampled).t()

    def sample(self, exclude):
        while True:
            t = tuple(np.random.randint(0, self.num_nodes, 2))
            if t[0] != t[1] and t not in exclude:
                exclude.add(t)
                exclude.add((t[1], t[0]))
                yield t


class AttributeMask(SSLTask):
    def __init__(self, graph, hidden_size, train_mask, mask_ratio, device):
        super().__init__(graph, device)
        self.linear = nn.Linear(hidden_size, graph.x.shape[1]).to(device)
        self.unlabeled = np.array([i for i in range(self.num_nodes) if not train_mask[i]])
        self.cached_features = None
        self.mask_ratio = mask_ratio

    def transform_data(self):
        if self.cached_features is None:
            perm = np.random.permutation(self.unlabeled)
            mask_nnz = int(self.num_nodes * self.mask_ratio)
            self.masked_nodes = perm[:mask_nnz]
            self.cached_features = self.features.clone()
            self.cached_features[self.masked_nodes] = torch.zeros(self.features.shape[1])
            self.pseudo_labels = self.features[self.masked_nodes].to(self.device)
            self.graph.features = self.cached_features

        return self.graph.to(self.device)

    def make_loss(self, embeddings):
        embeddings = self.linear(embeddings[self.masked_nodes])
        loss = F.mse_loss(embeddings, self.pseudo_labels, reduction="mean")
        return loss


class PairwiseDistance(SSLTask):
    def __init__(self, graph, hidden_size, class_split, sampling, dropedge_rate, num_centers, device):
        super().__init__(graph, device)
        self.nclass = len(class_split) + 1
        self.class_split = class_split
        self.max_distance = self.class_split[self.nclass - 2][1]
        self.sampling = sampling
        self.dropedge_rate = dropedge_rate
        self.num_centers = num_centers
        self.linear = nn.Linear(hidden_size, self.nclass).to(device)
        self.get_distance()

    def get_distance(self):
        if self.sampling:
            self.dis_node_pairs = [[] for i in range(self.nclass)]
            node_idx = random.sample(range(self.num_nodes), self.num_centers)
            adj = sp.coo_matrix(
                (np.ones(self.num_edges), (self.edge_index[0].cpu().numpy(), self.edge_index[1].cpu().numpy())),
                shape=(self.num_nodes, self.num_nodes),
            ).tocsr()

            num_samples = tqdm(range(self.num_centers))
            for i in num_samples:
                num_samples.set_description(f"Generating node pairs {i:03d}")
                idx = node_idx[i]
                queue = [idx]
                dis = -np.ones(self.num_nodes)
                dis[idx] = 0
                head = 0
                tail = 0
                cur_class = 0
                stack = []
                # bfs algorithm
                while head <= tail:
                    u = queue[head]
                    if cur_class != self.nclass - 1 and dis[u] >= self.class_split[cur_class][1]:
                        sampled = random.sample(stack, 1024) if len(stack) > 1024 else stack
                        if self.dis_node_pairs[cur_class] == []:
                            self.dis_node_pairs[cur_class] = np.array([[idx] * len(sampled), sampled]).transpose()
                        else:
                            self.dis_node_pairs[cur_class] = np.concatenate(
                                (self.dis_node_pairs[cur_class], np.array([[idx] * len(sampled), sampled]).transpose()),
                                axis=0,
                            )
                        cur_class += 1
                        if cur_class == self.nclass - 1:
                            break
                        stack = []
                    if u != idx:
                        stack.append(u)
                    head += 1
                    i_s = adj.indptr[u]
                    i_e = adj.indptr[u + 1]
                    for i in range(i_s, i_e):
                        v = adj.indices[i]
                        if dis[v] == -1:
                            dis[v] = dis[u] + 1
                            tail += 1
                            queue.append(v)
                remain = list(np.where(dis == -1)[0])
                sampled = random.sample(remain, 1024) if len(remain) > 1024 else remain
                if self.dis_node_pairs[cur_class] == []:
                    self.dis_node_pairs[cur_class] = np.array([[idx] * len(sampled), sampled]).transpose()
                else:
                    self.dis_node_pairs[cur_class] = np.concatenate(
                        (self.dis_node_pairs[cur_class], np.array([[idx] * len(sampled), sampled]).transpose()), axis=0
                    )
            if self.class_split[0][1] == 2:
                self.dis_node_pairs[0] = torch.stack(self.edge_index).cpu().numpy().transpose()
            num_per_class = np.min(np.array([len(dis) for dis in self.dis_node_pairs]))
            for i in range(self.nclass):
                sampled = np.random.choice(np.arange(len(self.dis_node_pairs[i])), num_per_class, replace=False)
                self.dis_node_pairs[i] = self.dis_node_pairs[i][sampled]
        else:
            G = nx.Graph()
            G.add_edges_from(torch.stack(self.edge_index).cpu().numpy().transpose())

            path_length = dict(nx.all_pairs_shortest_path_length(G, cutoff=self.max_distance))
            distance = -np.ones((self.num_nodes, self.num_nodes), dtype=np.int)
            for u, p in path_length.items():
                for v, d in p.items():
                    distance[u][v] = d - 1
            self.distance = distance

            self.dis_node_pairs = []
            for i in range(self.nclass - 1):
                tmp = np.array(
                    np.where((distance >= self.class_split[i][0]) * (distance < self.class_split[i][1]))
                ).transpose()
                np.random.shuffle(tmp)
                self.dis_node_pairs.append(tmp)
            tmp = np.array(np.where(distance == -1)).transpose()
            np.random.shuffle(tmp)
            self.dis_node_pairs.append(tmp)

    def transform_data(self):
        self.graph.edge_index, _ = dropout_adj(edge_index=self.edge_index, drop_rate=self.dropedge_rate)
        return self.graph.to(self.device)

    def make_loss(self, embeddings, sample=True, k=4000):
        node_pairs, pseudo_labels = self.sample(sample, k)
        embeddings = self.linear(torch.abs(embeddings[node_pairs[0]] - embeddings[node_pairs[1]]))
        output = F.log_softmax(embeddings, dim=1)
        return F.nll_loss(output, pseudo_labels)

    def sample(self, sample, k):
        sampled = torch.tensor([]).long()
        pseudo_labels = torch.tensor([]).long()
        for i in range(self.nclass):
            tmp = self.dis_node_pairs[i]
            if sample:
                x = int(random.random() * (len(tmp) - k))
                sampled = torch.cat([sampled, torch.tensor(tmp[x : x + k]).long().t()], 1)
                """
                indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
                sampled = torch.cat([sampled, torch.tensor(tmp[indices]).long().t()], 1)
                """
                pseudo_labels = torch.cat([pseudo_labels, torch.ones(k).long() * i])
            else:
                sampled = torch.cat([sampled, torch.tensor(tmp).long().t()], 1)
                pseudo_labels = torch.cat([pseudo_labels, torch.ones(len(tmp)).long() * i])
        return sampled.to(self.device), pseudo_labels.to(self.device)


class Distance2Clusters(SSLTask):
    def __init__(self, graph, hidden_size, num_clusters, device):
        super().__init__(graph, device)
        self.num_clusters = num_clusters
        self.linear = nn.Linear(hidden_size, num_clusters).to(device)
        self.gen_cluster_info()

    def transform_data(self):
        return self.graph.to(self.device)

    def gen_cluster_info(self, use_metis=False):
        G = nx.Graph()
        G.add_edges_from(torch.stack(self.edge_index).cpu().numpy().transpose())
        if use_metis:
            import metis

            _, parts = metis.part_graph(G, self.num_clusters)
        else:
            from sklearn.cluster import KMeans

            clustering = KMeans(n_clusters=self.num_clusters, random_state=0).fit(self.features.cpu())
            parts = clustering.labels_

        node_clusters = [[] for i in range(self.num_clusters)]
        for i, p in enumerate(parts):
            node_clusters[p].append(i)
        self.central_nodes = np.array([])
        self.distance_vec = np.zeros((self.num_nodes, self.num_clusters))
        for i in range(self.num_clusters):
            subgraph = G.subgraph(node_clusters[i])
            center = None
            for node in subgraph.nodes:
                if center is None or subgraph.degree[node] > subgraph.degree[center]:
                    center = node
            np.append(self.central_nodes, center)
            distance = dict(nx.shortest_path_length(G, source=center))
            for node in distance:
                self.distance_vec[node][i] = distance[node]
        self.distance_vec = torch.tensor(self.distance_vec).float().to(self.device)

    def make_loss(self, embeddings):
        output = self.linear(embeddings)
        return F.mse_loss(output, self.distance_vec, reduction="mean")


class PairwiseAttrSim(SSLTask):
    def __init__(self, graph, hidden_size, k, device):
        super().__init__(graph, device)
        self.k = k
        self.linear = nn.Linear(hidden_size, 1).to(self.device)
        self.get_attr_sim()

    def get_avg_distance(self, idx_sorted, k, sampled):
        self.G = nx.Graph()
        self.G.add_edges_from(torch.stack(self.edge_index).cpu().numpy().transpose())
        avg_min = 0
        avg_max = 0
        avg_sampled = 0
        for i in range(self.num_nodes):
            distance = dict(nx.shortest_path_length(self.G, source=i))
            sum = 0
            num = 0
            for node in idx_sorted[i, :k]:
                if node in distance:
                    sum += distance[node]
                    num += 1
            if num:
                avg_min += sum / num / self.num_nodes
            sum = 0
            num = 0
            for node in idx_sorted[i, -k - 1 :]:
                if node in distance:
                    sum += distance[node]
                    num += 1
            if num:
                avg_max += sum / num / self.num_nodes
            sum = 0
            num = 0
            for node in idx_sorted[i, sampled]:
                if node in distance:
                    sum += distance[node]
                    num += 1
            if num:
                avg_sampled += sum / num / self.num_nodes
        return avg_min, avg_max, avg_sampled

    def get_attr_sim(self):
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self.features.numpy())
        idx_sorted = sims.argsort(1)
        self.node_pairs = None
        self.pseudo_labels = None
        sampled = self.sample(self.k, self.num_nodes)
        for i in range(self.num_nodes):
            for node in np.hstack((idx_sorted[i, : self.k], idx_sorted[i, -self.k - 1 :], idx_sorted[i, sampled])):
                pair = torch.tensor([[i, node]])
                sim = torch.tensor([sims[i][node]])
                self.node_pairs = pair if self.node_pairs is None else torch.cat([self.node_pairs, pair], 0)
                self.pseudo_labels = sim if self.pseudo_labels is None else torch.cat([self.pseudo_labels, sim])
        print(
            "max k avg distance: {%.4f}, min k avg distance: {%.4f}, sampled k avg distance: {%.4f}"
            % (self.get_avg_distance(idx_sorted, self.k, sampled))
        )
        self.node_pairs = self.node_pairs.long().to(self.device)
        self.pseudo_labels = self.pseudo_labels.float().to(self.device)

    def sample(self, k, num_nodes):
        sampled = []
        for i in range(k):
            sampled.append(int(random.random() * (self.num_nodes - self.k * 2)) + self.k)
        return np.array(sampled)

    def transform_data(self):
        return self.graph.to(self.device)

    def make_loss(self, embeddings):
        node_pairs = self.node_pairs
        output = self.linear(torch.abs(embeddings[node_pairs[0]] - embeddings[node_pairs[1]]))
        return F.mse_loss(output, self.pseudo_labels, reduction="mean")


@register_model("self_auxiliary_task")
class SelfAuxiliaryTask(SelfSupervisedGenerativeModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--auxiliary-task", type=str)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--sampling", action="store_true")
        parser.add_argument("--dropedge-rate", type=float, default=0.0)
        parser.add_argument("--mask-ratio", type=float, default=0.1)
        # fmt: on

    def __init__(self, args):
        super().__init__()
        self.device = args.device_id[0] if not args.cpu else "cpu"
        self.auxiliary_task = args.auxiliary_task
        self.hidden_size = args.hidden_size
        self.sampling = args.sampling
        self.dropedge_rate = args.dropedge_rate
        self.mask_ratio = args.mask_ratio
        self.gcn = TKipfGCN(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.activation,
            args.residual,
            args.norm,
        )
        self.agent = None

    def generate_virtual_labels(self, data):
        if self.auxiliary_task == "edgemask":
            self.agent = EdgeMask(data, self.hidden_size, self.mask_ratio, self.device)
        elif self.auxiliary_task == "attributemask":
            self.agent = AttributeMask(data, self.hidden_size, data.train_mask, self.mask_ratio, self.device)
        elif self.auxiliary_task == "pairwise-distance":
            self.agent = PairwiseDistance(
                data,
                self.hidden_size,
                [(1, 2), (2, 3), (3, 5)],
                self.sampling,
                self.dropedge_rate,
                256,
                self.device,
            )
        elif self.auxiliary_task == "distance2clusters":
            self.agent = Distance2Clusters(data, self.hidden_size, 30, self.device)
        elif self.auxiliary_task == "pairwise-attr-sim":
            self.agent = PairwiseAttrSim(data, self.hidden_size, 5, self.device)
        else:
            raise Exception(
                "auxiliary task must be edgemask, pairwise-distance, distance2clusters, distance2clusters++ or pairwise-attr-sim"
            )

    def transform_data(self):
        return self.agent.transform_data()

    def self_supervised_loss(self, data):
        embed = self.gcn.embed(data)
        return self.agent.make_loss(embed)

    def forward(self, data):
        return self.gcn.forward(data)

    def embed(self, data):
        return self.gcn.embed(data)

    def get_parameters(self):
        return list(self.gcn.parameters()) + list(self.agent.linear.parameters())

    @staticmethod
    def get_trainer(args):
        return SelfSupervisedJointTrainer
