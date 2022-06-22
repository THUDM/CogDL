import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.utils.transform import dropout_adj
from cogdl.wrappers.tools.wrapper_utils import evaluate_node_embeddings_using_logreg
from tqdm import tqdm

from .. import UnsupervisedModelWrapper


class SelfAuxiliaryModelWrapper(UnsupervisedModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--auxiliary-task", type=str, default="edge_mask",
                            help="Option: edge_mask, attribute_mask, distance2clusters,"
                                 " pairwise_distance, pairwise_attr_sim")
        parser.add_argument("--dropedge-rate", type=float, default=0.0)
        parser.add_argument("--mask-ratio", type=float, default=0.1)
        parser.add_argument("--sampling", action="store_true")
        # fmt: on

    def __init__(self, model, optimizer_cfg, auxiliary_task, dropedge_rate, mask_ratio, sampling):
        super(SelfAuxiliaryModelWrapper, self).__init__()
        self.auxiliary_task = auxiliary_task
        self.optimizer_cfg = optimizer_cfg
        self.hidden_size = optimizer_cfg["hidden_size"]
        self.dropedge_rate = dropedge_rate
        self.mask_ratio = mask_ratio
        self.sampling = sampling
        self.model = model

        self.agent = None

    def train_step(self, subgraph):
        graph = subgraph
        with graph.local_graph():
            graph = self.agent.transform_data(graph)
            pred = self.model(graph)
        sup_loss = self.default_loss_fn(pred, graph.y)
        pred = self.model.embed(graph)
        ssl_loss = self.agent.make_loss(pred)
        return sup_loss + ssl_loss

    def test_step(self, graph):
        self.model.eval()
        with torch.no_grad():
            pred = self.model.embed(graph)
        y = graph.y
        result = evaluate_node_embeddings_using_logreg(pred, y, graph.train_mask, graph.test_mask)
        self.note("test_acc", result)

    def pre_stage(self, stage, data_w):
        if stage == 0:
            data = data_w.get_dataset().data
            self.generate_virtual_labels(data)

    def generate_virtual_labels(self, data):
        if self.auxiliary_task == "edge_mask":
            self.agent = EdgeMask(self.hidden_size, self.mask_ratio, self.device)
        elif self.auxiliary_task == "attribute_mask":
            self.agent = AttributeMask(data, self.hidden_size, data.train_mask, self.mask_ratio, self.device)
        elif self.auxiliary_task == "pairwise_distance":
            self.agent = PairwiseDistance(
                self.hidden_size, [(1, 2), (2, 3), (3, 5)], self.sampling, self.dropedge_rate, 256, self.device,
            )
        elif self.auxiliary_task == "distance2clusters":
            self.agent = Distance2Clusters(self.hidden_size, 30, self.device)
        elif self.auxiliary_task == "pairwise_attr_sim":
            self.agent = PairwiseAttrSim(self.hidden_size, 5, self.device)
        else:
            raise Exception(
                "auxiliary task must be edge_mask, attribute_mask, pairwise_distance, distance2clusters,"
                "or pairwise_attr_sim"
            )

    def setup_optimizer(self):
        lr, wd = self.optimizer_cfg["lr"], self.optimizer_cfg["weight_decay"]
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)


class SSLTask:
    def __init__(self, device):
        self.device = device
        self.cached_edges = None

    def transform_data(self, graph):
        raise NotImplementedError

    def make_loss(self, embeddings):
        raise NotImplementedError


class EdgeMask(SSLTask):
    def __init__(self, hidden_size, mask_ratio, device):
        super().__init__(device)
        self.linear = nn.Linear(hidden_size, 2).to(device)
        self.mask_ratio = mask_ratio

    def transform_data(self, graph):
        device = graph.x.device
        num_edges = graph.num_edges
        # if self.cached_edges is None:
        row, col = graph.edge_index
        edges = torch.stack([row, col])
        perm = np.random.permutation(num_edges)
        preserve_nnz = int(num_edges * (1 - self.mask_ratio))
        masked = perm[preserve_nnz:]
        preserved = perm[:preserve_nnz]
        self.masked_edges = edges[:, masked]
        self.cached_edges = edges[:, preserved]
        mask_num = len(masked)
        self.neg_edges = self.neg_sample(mask_num, graph).to(self.masked_edges.device)
        self.pseudo_labels = torch.cat([torch.ones(mask_num), torch.zeros(mask_num)]).long().to(device)
        self.node_pairs = torch.cat([self.masked_edges, self.neg_edges], 1).to(device)

        graph.edge_index = self.cached_edges
        return graph

    def make_loss(self, embeddings):
        embeddings = self.linear(torch.abs(embeddings[self.node_pairs[0]] - embeddings[self.node_pairs[1]]))
        output = F.log_softmax(embeddings, dim=1)
        return F.nll_loss(output, self.pseudo_labels)

    def neg_sample(self, edge_num, graph):
        edge_index = graph.edge_index
        num_nodes = graph.num_nodes
        edges = torch.stack(edge_index).t().cpu().numpy()
        exclude = set([(_[0], _[1]) for _ in list(edges)])
        itr = self.sample(exclude, num_nodes)
        sampled = [next(itr) for _ in range(edge_num)]
        return torch.tensor(sampled).t()

    def sample(self, exclude, num_nodes):
        while True:
            t = tuple(np.random.randint(0, num_nodes, 2))
            if t[0] != t[1] and t not in exclude:
                exclude.add(t)
                exclude.add((t[1], t[0]))
                yield t


class AttributeMask(SSLTask):
    def __init__(self, graph, hidden_size, train_mask, mask_ratio, device):
        super().__init__(device)
        self.linear = nn.Linear(hidden_size, graph.x.shape[1]).to(device)
        self.cached_features = None
        self.mask_ratio = mask_ratio

    def transform_data(self, graph):
        # if self.cached_features is None:
        device = graph.x.device
        x_feat = graph.x

        num_nodes = graph.num_nodes
        unlabelled = torch.where(~graph.train_mask)[0]
        perm = np.random.permutation(unlabelled.cpu().numpy())
        mask_nnz = int(num_nodes * self.mask_ratio)
        self.masked_nodes = perm[:mask_nnz]
        x_feat[self.masked_nodes] = 0
        self.pseudo_labels = x_feat[self.masked_nodes].to(device)
        graph.x = x_feat
        return graph

    def make_loss(self, embeddings):
        embeddings = self.linear(embeddings[self.masked_nodes])
        loss = F.mse_loss(embeddings, self.pseudo_labels, reduction="mean")
        return loss


class PairwiseDistance(SSLTask):
    def __init__(self, hidden_size, class_split, sampling, dropedge_rate, num_centers, device):
        super().__init__(device)
        self.nclass = len(class_split) + 1
        self.class_split = class_split
        self.max_distance = self.class_split[self.nclass - 2][1]
        self.sampling = sampling
        self.dropedge_rate = dropedge_rate
        self.num_centers = num_centers
        self.linear = nn.Linear(hidden_size, self.nclass).to(device)
        self.get_distance_cache = False

    def get_distance(self, graph):
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        edge_index = graph.edge_index

        if self.sampling:
            self.dis_node_pairs = [[] for i in range(self.nclass)]
            node_idx = random.sample(range(num_nodes), self.num_centers)
            adj = sp.coo_matrix(
                (np.ones(num_edges), (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())),
                shape=(num_nodes, num_nodes),
            ).tocsr()

            num_samples = tqdm(range(self.num_centers))
            for i in num_samples:
                num_samples.set_description(f"Generating node pairs {i:03d}")
                idx = node_idx[i]
                queue = [idx]
                dis = -np.ones(num_nodes)
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
                self.dis_node_pairs[0] = torch.stack(edge_index).cpu().numpy().transpose()
            num_per_class = np.min(np.array([len(dis) for dis in self.dis_node_pairs]))
            for i in range(self.nclass):
                sampled = np.random.choice(np.arange(len(self.dis_node_pairs[i])), num_per_class, replace=False)
                self.dis_node_pairs[i] = self.dis_node_pairs[i][sampled]
        else:
            G = nx.Graph()
            G.add_edges_from(torch.stack(edge_index).cpu().numpy().transpose())

            path_length = dict(nx.all_pairs_shortest_path_length(G, cutoff=self.max_distance))
            distance = -np.ones((num_nodes, num_nodes), dtype=np.int)
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

    def transform_data(self, graph):
        if not self.get_distance_cache:
            self.get_distance(graph)
            self.get_distance_cache = True
        graph.edge_index, _ = dropout_adj(edge_index=graph.edge_index, drop_rate=self.dropedge_rate)
        return graph

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
    def __init__(self, hidden_size, num_clusters, device):
        super().__init__(device)
        self.num_clusters = num_clusters
        self.linear = nn.Linear(hidden_size, num_clusters).to(device)
        self.gen_cluster_info_cache = False

    def transform_data(self, graph):
        if not self.gen_cluster_info_cache:
            self.gen_cluster_info(graph)
            self.gen_cluster_info_cache = True

        return graph

    def gen_cluster_info(self, graph, use_metis=False):
        edge_index = graph.edge_index
        num_nodes = graph.num_nodes
        x = graph.x

        G = nx.Graph()
        G.add_edges_from(torch.stack(edge_index).cpu().numpy().transpose())
        if use_metis:
            import metis

            _, parts = metis.part_graph(G, self.num_clusters)
        else:
            from sklearn.cluster import KMeans

            clustering = KMeans(n_clusters=self.num_clusters, random_state=0).fit(x.cpu())
            parts = clustering.labels_

        node_clusters = [[] for i in range(self.num_clusters)]
        for i, p in enumerate(parts):
            node_clusters[p].append(i)
        self.central_nodes = np.array([])
        self.distance_vec = np.zeros((num_nodes, self.num_clusters))
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
    def __init__(self, hidden_size, k, device):
        super().__init__(device)
        self.k = k
        self.linear = nn.Linear(hidden_size, 1).to(self.device)
        self.get_attr_sim_cache = False

    def get_avg_distance(self, graph, idx_sorted, k, sampled):
        edge_index = graph.edge_index
        num_nodes = graph.num_nodes

        self.G = nx.Graph()
        self.G.add_edges_from(torch.stack(edge_index).cpu().numpy().transpose())
        avg_min = 0
        avg_max = 0
        avg_sampled = 0
        for i in range(num_nodes):
            distance = dict(nx.shortest_path_length(self.G, source=i))
            sum = 0
            num = 0
            for node in idx_sorted[i, :k]:
                if node in distance:
                    sum += distance[node]
                    num += 1
            if num:
                avg_min += sum / num / num_nodes
            sum = 0
            num = 0
            for node in idx_sorted[i, -k - 1 :]:
                if node in distance:
                    sum += distance[node]
                    num += 1
            if num:
                avg_max += sum / num / num_nodes
            sum = 0
            num = 0
            for node in idx_sorted[i, sampled]:
                if node in distance:
                    sum += distance[node]
                    num += 1
            if num:
                avg_sampled += sum / num / num_nodes
        return avg_min, avg_max, avg_sampled

    def get_attr_sim(self, graph):
        x = graph.x
        num_nodes = graph.num_nodes

        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(x.cpu().numpy())
        idx_sorted = sims.argsort(1)
        self.node_pairs = None
        self.pseudo_labels = None
        sampled = self.sample(self.k, num_nodes)
        for i in range(num_nodes):
            for node in np.hstack((idx_sorted[i, : self.k], idx_sorted[i, -self.k - 1 :], idx_sorted[i, sampled])):
                pair = torch.tensor([[i, node]])
                sim = torch.tensor([sims[i][node]])
                self.node_pairs = pair if self.node_pairs is None else torch.cat([self.node_pairs, pair], 0)
                self.pseudo_labels = sim if self.pseudo_labels is None else torch.cat([self.pseudo_labels, sim])
        print(
            "max k avg distance: {%.4f}, min k avg distance: {%.4f}, sampled k avg distance: {%.4f}"
            % (self.get_avg_distance(graph, idx_sorted, self.k, sampled))
        )
        self.node_pairs = self.node_pairs.long().to(self.device)
        self.pseudo_labels = self.pseudo_labels.float().to(self.device)

    def sample(self, k, num_nodes):
        sampled = []
        for i in range(k):
            sampled.append(int(random.random() * (num_nodes - self.k * 2)) + self.k)
        return np.array(sampled)

    def transform_data(self, graph):
        if not self.get_attr_sim_cache:
            self.get_attr_sim(graph)
        return graph

    def make_loss(self, embeddings):
        node_pairs = self.node_pairs
        output = self.linear(torch.abs(embeddings[node_pairs[0]] - embeddings[node_pairs[1]]))
        return F.mse_loss(output, self.pseudo_labels, reduction="mean")
