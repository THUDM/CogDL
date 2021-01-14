import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp
import numpy as np
import networkx as nx
import random


class SSLTask:
    def __init__(self, edge_index, features, device):
        self.edge_index = edge_index
        self.num_nodes = torch.max(edge_index).cpu() + 1
        self.num_edges = edge_index.shape[1]
        self.features = features
        self.device = device
        self.cached_edges = None

    def transform_data(self):
        raise NotImplementedError

    def make_loss(self, embeddings):
        raise NotImplementedError


class EdgeMask(SSLTask):
    def __init__(self, edge_index, features, hidden_size, device):
        super().__init__(edge_index, features, device)
        self.linear = nn.Linear(hidden_size, 2).to(device)

    def transform_data(self, mask_ratio=0.1):
        if self.cached_edges is None:
            edges = self.edge_index.t()
            perm = np.random.permutation(self.num_edges)
            preserve_nnz = int(self.num_edges * (1 - mask_ratio))
            masked = perm[preserve_nnz:]
            preserved = perm[:preserve_nnz]
            self.masked_edges = edges[masked].t()
            self.cached_edges = edges[preserved].t()
            mask_num = len(masked)
            self.neg_edges = self.neg_sample(mask_num)
            self.pseudo_labels = torch.cat([torch.ones(mask_num), torch.zeros(mask_num)]).long().to(self.device)
            self.node_pairs = torch.cat([self.masked_edges, self.neg_edges], 1)

        return self.cached_edges, self.features

    def make_loss(self, embeddings):
        embeddings = self.linear(torch.abs(embeddings[self.node_pairs[0]] - embeddings[self.node_pairs[1]]))
        output = F.log_softmax(embeddings, dim=1)
        return F.nll_loss(output, self.pseudo_labels)

    def neg_sample(self, edge_num):
        edges = self.edge_index.t().cpu().numpy()
        exclude = set([(_[0], _[1]) for _ in list(edges)])
        itr = self.sample(exclude)
        sampled = [next(itr) for _ in range(edge_num)]
        return torch.tensor(sampled).t().to(self.device)

    def sample(self, exclude):
        while True:
            t = tuple(np.random.randint(0, self.num_nodes, 2))
            if t[0] != t[1] and t not in exclude:
                exclude.add(t)
                exclude.add((t[1], t[0]))
                yield t


class PairwiseDistance(SSLTask):
    def __init__(self, edge_index, features, hidden_size, num_class, device):
        super().__init__(edge_index, features, device)
        self.linear = nn.Linear(hidden_size, num_class).to(device)
        self.nclass = num_class
        self.get_distance()

    def get_distance(self):
        G = nx.Graph()
        G.add_edges_from(self.edge_index.cpu().t().numpy())

        path_length = dict(nx.all_pairs_shortest_path_length(G, cutoff=self.nclass - 1))
        distance = -np.ones((self.num_nodes, self.num_nodes)).astype(int)
        for u, p in path_length.items():
            for v, d in p.items():
                distance[u][v] = d - 1
        distance[distance == -1] = self.nclass - 1
        self.distance = distance

        self.dis_node_pairs = []
        for i in range(self.nclass):
            tmp = np.array(np.where(distance == i)).transpose()
            self.dis_node_pairs.append(tmp)

    def transform_data(self):
        return self.edge_index, self.features

    def make_loss(self, embeddings, k=4000):
        node_pairs, pseudo_labels = self.sample(k)
        # print(node_pairs, pseudo_labels)
        embeddings = self.linear(torch.abs(embeddings[node_pairs[0]] - embeddings[node_pairs[1]]))
        output = F.log_softmax(embeddings, dim=1)
        return F.nll_loss(output, pseudo_labels)

    def sample(self, k):
        sampled = torch.tensor([]).long()
        pseudo_labels = torch.tensor([]).long()
        for i in range(self.nclass):
            tmp = self.dis_node_pairs[i]
            indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
            sampled = torch.cat([sampled, torch.tensor(tmp[indices]).t()], 1)
            pseudo_labels = torch.cat([pseudo_labels, torch.ones(k).long() * i])
        return sampled.to(self.device), pseudo_labels.to(self.device)


class Distance2Clusters(SSLTask):
    def __init__(self, edge_index, features, hidden_size, num_clusters, device):
        super().__init__(edge_index, features, device)
        self.num_clusters = num_clusters
        self.linear = nn.Linear(hidden_size, num_clusters).to(device)
        self.gen_cluster_info()

    def transform_data(self):
        return self.edge_index, self.features

    def gen_cluster_info(self, use_metis=False):
        G = nx.Graph()
        G.add_edges_from(self.edge_index.cpu().t().numpy())
        if use_metis:
            import metis

            _, parts = metis.part_graph(G, self.num_clusters)
        else:
            from sklearn.cluster import SpectralClustering

            clustering = SpectralClustering(
                n_clusters=self.num_clusters, assign_labels="discretize", random_state=0
            ).fit(self.features.cpu())
            parts = clustering.labels_

        node_clusters = [[] for i in range(self.num_clusters)]
        for i, p in enumerate(parts):
            node_clusters[p].append(i)
        self.central_nodes = np.array([])
        self.distance_vec = np.zeros((self.num_nodes, self.num_clusters))
        for i in range(self.num_clusters):
            subgraph = G.subgraph(node_clusters[i])
            center = None
            # print(subgraph.nodes)
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
    def __init__(self, edge_index, features, hidden_size, k, device):
        super().__init__(edge_index, features, device)
        self.k = k
        self.linear = nn.Linear(hidden_size, 1).to(self.device)
        self.get_attr_sim()

    def get_avg_distance(self, idx_sorted, k, sampled):
        self.G = nx.Graph()
        self.G.add_edges_from(self.edge_index.cpu().t().numpy())
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
                avg_min += sum / num / self.num_nodes.item()
            sum = 0
            num = 0
            for node in idx_sorted[i, -k - 1 :]:
                if node in distance:
                    sum += distance[node]
                    num += 1
            if num:
                avg_max += sum / num / self.num_nodes.item()
            sum = 0
            num = 0
            for node in idx_sorted[i, sampled]:
                if node in distance:
                    sum += distance[node]
                    num += 1
            if num:
                avg_sampled += sum / num / self.num_nodes.item()
        return avg_min, avg_max, avg_sampled

    def get_attr_sim(self):
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self.features.detach().cpu().numpy())
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
        # print(self.node_pairs, self.pseudo_labels)
        self.node_pairs = self.node_pairs.long().to(self.device)
        self.pseudo_labels = self.pseudo_labels.float().to(self.device)

    def sample(self, k, num_nodes):
        sampled = []
        for i in range(k):
            sampled.append(int(random.random() * (self.num_nodes - self.k * 2)) + self.k)
        return np.array(sampled)

    def transform_data(self):
        return self.edge_index, self.features

    def make_loss(self, embeddings):
        """
        k = 5000
        sampled = np.random.choice(self.node_pairs.shape[0], k, replace=False)
        node_pairs = self.node_pairs[sampled].t()
        """
        node_pairs = self.node_pairs
        output = self.linear(torch.abs(embeddings[node_pairs[0]] - embeddings[node_pairs[1]]))
        return F.mse_loss(output, self.pseudo_labels, reduction="mean")


class Distance2ClustersPP(SSLTask):
    def __init__(self, edge_index, features, labels, hidden_size, num_clusters, k, device):
        super().__init__(edge_index, features, device)
        self.labels = labels.cpu().numpy()
        self.k = k
        self.num_clusters = num_clusters
        self.clusters = None
        self.linear = nn.Linear(hidden_size, 1).to(self.device)

    def build_graph(self):
        edges = self.edge_index.detach().cpu().numpy()
        edge_attr = np.ones(edges.shape[1])
        inter_label = np.where(self.labels[edges[0]] - self.labels[edges[1]] != 0)
        inter_cluster = np.where(self.clusters[edges[0]] - self.clusters[edges[1]] != 0)
        edge_attr[inter_label] = 2
        edge_attr[inter_cluster] = 2
        self.G = nx.Graph()
        for i in range(edges.shape[1]):
            self.G.add_edge(edges[0][i], edges[1][i], weight=edge_attr[i])

    def update_cluster(self):
        if self.clusters is None:
            from sklearn.cluster import SpectralClustering

            clustering = SpectralClustering(
                n_clusters=self.num_clusters, assign_labels="discretize", random_state=0
            ).fit(self.features.cpu())
            self.clusters = clustering.labels_

        self.build_graph()
        self.node_pairs = None
        self.pseudo_labels = None
        for i in range(self.num_clusters):
            cluster_idx = np.where(self.clusters == i)[0]
            if len(cluster_idx) < self.k:
                continue
            sampled = np.random.choice(cluster_idx, self.k, replace=False)
            for node in sampled:
                distance = dict(nx.shortest_path_length(self.G, source=node))
                for j in range(self.num_nodes):
                    if j == node or j not in distance:
                        continue
                    pair = torch.tensor([[j, node]])
                    sim = torch.tensor([distance[j]])
                    self.node_pairs = pair if self.node_pairs is None else torch.cat([self.node_pairs, pair], 0)
                    self.pseudo_labels = sim if self.pseudo_labels is None else torch.cat([self.pseudo_labels, sim])
        self.node_pairs = self.node_pairs.t().to(self.device)
        self.pseudo_labels = self.pseudo_labels.float().to(self.device)

    def transform_data(self):
        return self.edge_index, self.features

    def make_loss(self, embeddings):
        embeddings = self.linear(embeddings)
        # print(embeddings.shape, self.pseudo_labels.shape, F.mse_loss(embeddings, self.pseudo_labels, reduction="mean"))
        return F.mse_loss(embeddings, self.pseudo_labels, reduction="mean")
