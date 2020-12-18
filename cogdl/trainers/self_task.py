import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp
import numpy as np
import networkx as nx

class SSLTask:
    def __init__(self, edge_index, features, device):
        self.edge_index = edge_index
        self.num_nodes = torch.max(edge_index) + 1
        self.num_edges = edge_index.shape[1]
        self.features = features
        self.device = device
        self.cached_edges = None

    def transform_data(self):
        raise NotImplemented

    def make_loss(self, embeddings):
        raise NotImplemented

class EdgeMask(SSLTask):
    def __init__(self, edge_index, features, hidden_size, device):
        super().__init__(edge_index, features, device)
        self.linear = nn.Linear(hidden_size, 2).to(device)

    def transform_data(self, mask_ratio=0.1):
        if self.cached_edges is None:
            edges = self.edge_index.t()
            perm = np.random.permutation(self.num_edges)
            preserve_nnz = int(self.num_edges * (1 - mask_ratio))
            masked = perm[preserve_nnz :]
            preserved = perm[: preserve_nnz]
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
        distance = - np.ones((self.num_nodes, self.num_nodes)).astype(int)
        for u, p in path_length.items():
            for v, d in p.items():
                distance[u][v] = d - 1
        distance[distance == -1] = self.nclass - 1
        self.distance = distance

        self.dis_node_pairs = []
        for i in range(self.nclass):
            tmp = np.array(np.where(distance==i)).transpose()
            self.dis_node_pairs.append(tmp)

    def transform_data(self):
        return self.edge_index, self.features

    def make_loss(self, embeddings, k=4000):
        node_pairs, pseudo_labels = self.sample(k)
        #print(node_pairs, pseudo_labels)
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

    def gen_cluster_info(self):
        import metis
        G = nx.Graph()
        G.add_edges_from(self.edge_index.cpu().t().numpy())

        _, parts = metis.part_graph(G, self.num_clusters)
        node_clusters = [[] for i in range(self.num_clusters)]
        for i, p in enumerate(parts):
            node_clusters[p].append(i)
        self.central_nodes = np.array([])
        self.distance_vec = np.zeros((self.num_nodes, self.num_clusters))
        for i in range(self.num_clusters):
            subgraph = G.subgraph(node_clusters[i])
            center = None
            #print(subgraph.nodes)
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
        self.linear = nn.Linear(hidden_size, 1)
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
            for node in idx_sorted[i, -k-1:]:
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
        sampled = np.random.choice(np.arange(self.k, self.num_nodes - self.k), self.k, replace=False)
        for i in range(self.num_nodes):
            for node in np.hstack((idx_sorted[i, :self.k], idx_sorted[i, -self.k - 1:], idx_sorted[i, sampled])):
                pair = torch.tensor([[i, node]])
                sim = torch.tensor([sims[i][node]])
                self.node_pairs = pair if self.node_pairs is None else torch.cat([self.node_pairs, pair], 0)
                self.pseudo_labels = sim if self.pseudo_labels is None else torch.cat([self.pseudo_labels, sim])
        #print("max k avg distance: {%.4f}, min k avg distance: {%.4f}, sampled k avg distance: {%.4f}" % (self.get_avg_distance(idx_sorted, self.k, sampled)))
        #print(self.node_pairs, self.pseudo_labels)
        self.node_pairs = self.node_pairs.long().to(self.device)
        self.pseudo_labels = self.pseudo_labels.float().to(self.device)

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
        return F.mse_loss(output, self.pseudo_labels, reduction='mean')

class Distance2ClustersPP(SSLTask):
    def __init__(self, edge_index, features, labels, hidden_size, num_clusters, k, device):
        super().__init__(edge_index, features, device)
        self.labels = labels.numpy()
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
            clustering = SpectralClustering(n_clusters=self.num_clusters, assign_labels="discretize", random_state=0).fit(self.features)
            self.clusters = clustering.labels_

        self.build_graph()
        self.node_pairs = None
        self.pseudo_labels = None
        for i in range(self.num_clusters):
            cluster_idx = np.where(self.clusters==i)[0]
            if len(cluster_idx) < self.k:
                continue
            sampled = np.random.choice(cluster_idx, self.k, replace=False)
            for node in sampled:
                distance = dict(nx.shortest_path_length(self.G, source=node))
                for j in range(self.num_nodes):
                    if j == node or not j in distance:
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
        #print(embeddings.shape, self.pseudo_labels.shape, F.mse_loss(embeddings, self.pseudo_labels, reduction="mean"))
        return F.mse_loss(embeddings, self.pseudo_labels, reduction="mean")

class ScalableDistancePred(SSLTask):
    def __init__(self, edge_index, features, labels, hidden_size, num_clusters, k, device):
        super().__init__(edge_index, features, device)
        self.labels = labels.numpy()
        self.k = k
        self.num_clusters = num_clusters
        self.num_class1 = 10
        self.num_class2 = 3
        self.linear1 = nn.Linear(hidden_size, self.num_class1).to(self.device)
        self.linear2 = nn.Linear(hidden_size, self.num_class2).to(self.device)
        self.get_clusters()
        self.G = nx.Graph()
        self.G.add_edges_from(self.edge_index.cpu().t().numpy())

    def get_clusters(self):
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(n_clusters=self.num_clusters, assign_labels="discretize", random_state=0).fit(self.features)
        self.clusters = clustering.labels_

    def get_class(self, dis):
        if dis <= 1:
            return dis
        else:
            return 2

    def adv_hop(self, u, v):
        return self.labels[u] != self.labels[v] and self.clusters[u] != self.clusters[v]

    def update(self):
        class_node_pairs = [[] for _ in range(self.num_class1)]
        class_adv_distance = [[] for _ in range(self.num_class1)]
        # 1, 2 hop neighbor
        for i in range(self.num_nodes):
            layers = dict(nx.bfs_successors(self.G, source=i, depth_limit=2))
            for succ in layers:
                for node in layers[succ]:
                    if node < i:
                        continue
                    if succ == i:
                        class_node_pairs[0].append([i, node])
                        class_adv_distance[0].append(self.get_class(self.adv_hop(i, node)))
                    else:
                        class_node_pairs[1].append([i, node])
                        class_adv_distance[1].append(self.get_class(self.adv_hop(i, succ) + self.adv_hop(succ, node)))

        # 3+ hop neighbor
        for i in range(self.num_clusters):
            cluster_idx = np.where(self.clusters==i)[0]
            if len(cluster_idx) < self.k:
                continue
            sampled = np.random.choice(cluster_idx, self.k, replace=False)
            for node in sampled:
                l = 0
                r = 0
                q = [node]
                dis = np.zeros(self.num_nodes)
                adv_dis = np.zeros(self.num_nodes)
                while l <= r:
                    u = q[l]
                    l += 1
                    for e in self.G.edges(u):
                        v = e[1]
                        if dis[v] == 0 and v != node:
                            dis[v] = dis[u] + 1
                            adv_dis[v] = adv_dis[u] + self.adv_hop(u, v)
                            q.append(v)
                            r += 1
                        elif dis[v] == dis[u] + 1 and adv_dis[v] > adv_dis[u] + self.adv_hop(u, v):
                            adv_dis[v] = adv_dis[u] + self.adv_hop(u, v)
                for i in range(self.num_nodes):
                    if dis[i] > 2 and dis[i] < self.num_class1:
                        class_node_pairs[int(dis[i] - 1)].append([node, i])
                        class_adv_distance[int(dis[i] - 1)].append(self.get_class(adv_dis[i]))
                    elif dis[i] >= self.num_class1:
                        class_node_pairs[self.num_class1 - 1].append([node, i])
                        class_adv_distance[self.num_class1 - 1].append(self.get_class(adv_dis[i]))

        print([len(i) for i in class_adv_distance])
        """
        path_length = dict(nx.all_pairs_shortest_path_length(self.G, cutoff=self.num_class1 - 1))
        distance = - np.ones((len(self.G), len(self.G))).astype(int)
        for u, p in path_length.items():
            for v, d in p.items():
                distance[u][v] = d - 1
        distance[distance == -1] = self.num_class1 - 1
        for i in range(self.num_class1):
            for fuck in class_node_pairs[i]:
                assert distance[fuck[0]][fuck[1]] == i
        """
        num_per_class = np.min(np.array([len(i) for i in class_adv_distance]))
        self.node_pairs = torch.tensor([]).long()
        self.pseudo_labels1 = torch.tensor([]).long()
        self.pseudo_labels2 = torch.tensor([]).long()
        for i in range(0, self.num_class1):
            indices = np.random.choice(np.arange(len(class_adv_distance[i])), num_per_class, replace=False)
            self.node_pairs = torch.cat([self.node_pairs, torch.tensor(np.array(class_node_pairs[i])[indices]).long().t()], 1)
            self.pseudo_labels1 = torch.cat([self.pseudo_labels1, torch.ones(num_per_class).long() * i])
            self.pseudo_labels2 = torch.cat([self.pseudo_labels2, torch.tensor(np.array(class_adv_distance[i])[indices]).long()])
        print(self.node_pairs, self.pseudo_labels1)

    def transform_data(self):
        return self.edge_index, self.features

    def make_loss(self, embeddings):
        diff = torch.abs(embeddings[self.node_pairs[0]] - embeddings[self.node_pairs[1]])
        embeddings1 = self.linear1(diff)
        embeddings2 = self.linear2(diff)
        output1 = F.log_softmax(embeddings1, dim=1)
        output2 = F.log_softmax(embeddings2, dim=1)
        return F.nll_loss(output1, self.pseudo_labels1) #+ F.nll_loss(output2, self.pseudo_labels2)