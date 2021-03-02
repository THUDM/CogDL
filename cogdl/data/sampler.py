from typing import List

import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data

from cogdl.data import Data
from cogdl.utils import remove_self_loops, row_normalization


def normalize(adj):
    D = adj.sum(1).flatten()
    norm_diag = sp.dia_matrix((1 / D, 0), shape=adj.shape)
    adj = norm_diag.dot(adj)
    adj.sort_indices()
    return adj


def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))


class Sampler:
    r"""
    Base sampler class.
    Constructs a sampler with data (`torch_geometric.data.Data`), which indicates Graph to be sampled,
    and args_params (Dictionary) which represents args parameters needed by the sampler.
    """

    def __init__(self, data, args_params):
        self.data = data.clone()
        self.full_graph = data.clone()
        self.num_nodes = self.data.x.size()[0]
        self.num_edges = (
            self.data.edge_index_train.size()[1]
            if hasattr(self.data, "edge_index_train")
            else self.data.edge_index.size()[1]
        )

    def sample(self):
        pass


class SAINTSampler(Sampler):
    r"""
    The sampler super-class referenced from GraphSAINT (https://arxiv.org/abs/1907.04931). Any graph sampler is supposed to perform
    the following meta-steps:
     1. [optional] Preprocessing: e.g., for edge sampler, we need to calculate the
            sampling probability for each edge in the training graph. This is to be
            performed only once per phase (or, once throughout the whole training,
            since in most cases, training only consists of a single phase).
            ==> Need to override the `preproc()` in sub-class
     2. Post-processing: upon getting the sampled subgraphs, we need to prepare the
            appropriate information (e.g., subgraph adj with renamed indices) to
            enable the PyTorch trainer.
    """

    def __init__(self, data, args_params):
        super().__init__(data, args_params)

        self.gen_adj()

        self.train_mask = self.data.train_mask.cpu().numpy()
        self.node_train = np.arange(1, self.num_nodes + 1) * self.train_mask
        self.node_train = self.node_train[self.node_train != 0] - 1

        self.sample_coverage = args_params["sample_coverage"]
        self.preprocess()

    def gen_adj(self):
        edge_index = (
            self.data.edge_index_train.cpu().numpy()
            if hasattr(self.data, "edge_index_train")
            else self.data.edge_index.cpu().numpy()
        )

        self.adj = sp.coo_matrix(
            (np.ones(self.num_edges), (edge_index[0], edge_index[1])),
            shape=(self.num_nodes, self.num_nodes),
        ).tocsr()

    def preprocess(self):
        r"""
        estimation of loss / aggregation normalization factors.
        For some special sampler, no need to estimate norm factors, we can calculate
        the node / edge probabilities directly.
        However, for integrity of the framework, we follow the same procedure
        for all samplers:
            1. sample enough number of subgraphs
            2. update the counter for each node / edge in the training graph
            3. estimate norm factor alpha and lambda
        """
        self.subgraph_data = []
        self.subgraph_node_idx = []
        self.subgraph_edge_idx = []

        self.norm_loss_train = np.zeros(self.num_nodes)
        self.norm_aggr_train = np.zeros(self.num_edges)
        self.norm_loss_test = np.ones(self.num_nodes) / self.num_nodes
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))

        num_sampled_nodes = 0
        while True:
            num_sampled_nodes += self.gen_subgraph()
            print(
                "\rGenerating subgraphs %.2lf%%"
                % min(num_sampled_nodes * 100 / self.data.num_nodes / self.sample_coverage, 100),
                end="",
                flush=True,
            )
            if num_sampled_nodes > self.sample_coverage * self.num_nodes:
                break

        num_subg = len(self.subgraph_data)
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraph_edge_idx[i]] += 1
            self.norm_loss_train[self.subgraph_node_idx[i]] += 1
        for v in range(self.data.num_nodes):
            i_s = self.adj.indptr[v]
            i_e = self.adj.indptr[v + 1]
            val = np.clip(self.norm_loss_train[v] / self.norm_aggr_train[i_s:i_e], 0, 1e4)
            val[np.isnan(val)] = 0.1
            self.norm_aggr_train[i_s:i_e] = val
        self.norm_loss_train[np.where(self.norm_loss_train == 0)[0]] = 0.1
        self.norm_loss_train[self.node_train] = num_subg / self.norm_loss_train[self.node_train] / self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))

    def one_batch(self, phase, require_norm=True):
        r"""
        Generate one minibatch for model. In the 'train' mode, one minibatch corresponds
        to one subgraph of the training graph. In the 'val' or 'test' mode, one batch
        corresponds to the full graph (i.e., full-batch rather than minibatch evaluation
        for validation / test sets).

        Inputs:
            phase               str, can be 'train', 'val', 'test'
            require_norm        boolean

        Outputs:
            data                Data object, modeling the sampled subgraph
            data.norm_aggr      aggregation normalization
            data.norm_loss      loss normalization
        """
        if phase in ["val", "test"]:
            data = self.full_graph.clone()
            data.norm_loss = self.norm_loss_test
        else:
            while True:
                if len(self.subgraph_data) == 0:
                    self.gen_subgraph()

                data = self.subgraph_data.pop()
                node_idx = self.subgraph_node_idx.pop()
                edge_idx = self.subgraph_edge_idx.pop()
                if self.exists_train_nodes(node_idx):
                    break

            if require_norm:
                data.norm_aggr = torch.FloatTensor(self.norm_aggr_train[edge_idx][:])
                data.norm_loss = self.norm_loss_train[node_idx]

            data.train_mask = self.data.train_mask[node_idx]
            data.val_mask = self.data.val_mask[node_idx]
            data.test_mask = self.data.test_mask[node_idx]

        # adj = data._build_adj_()
        # adj = normalize(adj).tocoo()
        # data.adj = _coo_scipy2torch(adj)
        edge_weight = row_normalization(data.x.shape[0], data.edge_index)
        data.edge_weight = edge_weight
        # data._eliminate_adj_()
        return data

    def exists_train_nodes(self, node_idx):
        for idx in node_idx:
            if self.train_mask[idx]:
                return True
        return False

    def node_induction(self, node_idx):
        node_idx = np.unique(node_idx)
        node_flags = np.zeros(self.num_nodes)
        for u in node_idx:
            node_flags[u] = 1
        edge_idx = []
        for u in node_idx:
            for e in range(self.adj.indptr[u], self.adj.indptr[u + 1]):
                v = self.adj.indices[e]
                if node_flags[v]:
                    edge_idx.append(e)
        edge_idx = np.array(edge_idx)
        return self.data.subgraph(node_idx), node_idx, edge_idx

    def edge_induction(self, edge_idx):
        return self.data.edge_subgraph(edge_idx, require_idx=True)

    def gen_subgraph(self):
        _data, _node_idx, _edge_idx = self.sample()
        self.subgraph_data.append(_data)
        self.subgraph_node_idx.append(_node_idx)
        self.subgraph_edge_idx.append(_edge_idx)
        return len(_node_idx)

    def sample(self):
        pass


class NodeSampler(SAINTSampler):
    r"""
    randomly select nodes, then adding edges connecting these nodes
    Args:
        sample_coverage (integer):  number of sampled nodes during estimation / number of nodes in graph
        size_subgraph (integer): number of nodes in subgraph
    """

    def __init__(self, data, args_params):
        self.node_num_subgraph = args_params["size_subgraph"]
        super().__init__(data, args_params)

    def sample(self):
        node_idx = np.random.choice(np.arange(self.num_nodes), self.node_num_subgraph)
        return self.node_induction(node_idx)


class EdgeSampler(SAINTSampler):
    r"""
    randomly select edges, then adding nodes connected by these edges
    Args:
        sample_coverage (integer):  number of sampled nodes during estimation / number of nodes in graph
        size_subgraph (integer): number of edges in subgraph
    """

    def __init__(self, data, args_params):
        self.edge_num_subgraph = args_params["size_subgraph"]
        super().__init__(data, args_params)

    def sample(self):
        edge_idx = np.random.choice(np.arange(self.num_edges), self.edge_num_subgraph)
        return self.edge_induction(edge_idx)


class RWSampler(SAINTSampler):
    r"""
    The sampler performs unbiased random walk, by following the steps:
     1. Randomly pick `size_root` number of root nodes from all training nodes;
     2. Perform length `size_depth` random walk from the roots. The current node
            expands the next hop by selecting one of the neighbors uniformly
            at random;
     3. Generate node-induced subgraph from the nodes touched by the random walk.
    Args:
        sample_coverage (integer):  number of sampled nodes during estimation / number of nodes in graph
        num_walks (integer): number of walks
        walk_length (integer): length of the random walk
    """

    def __init__(self, data, args_params):
        self.num_walks = args_params["num_walks"]
        self.walk_length = args_params["walk_length"]
        super().__init__(data, args_params)

    def sample(self):
        node_idx = []
        for walk in range(self.num_walks):
            u = np.random.choice(self.node_train)
            node_idx.append(u)
            for step in range(self.walk_length):
                idx_s = self.adj.indptr[u]
                idx_e = self.adj.indptr[u + 1]
                if idx_s >= idx_e:
                    break
                e = np.random.randint(idx_s, idx_e)
                u = self.adj.indices[e]
                node_idx.append(u)

        return self.node_induction(node_idx)


class MRWSampler(SAINTSampler):
    r"""multidimentional random walk, similar to https://arxiv.org/abs/1002.1751"""

    def __init__(self, data, args_params):
        self.size_frontier = args_params["size_frontier"]
        self.edge_num_subgraph = args_params["size_subgraph"]
        super().__init__(data, args_params)

    def sample(self):
        frontier = np.random.choice(np.arange(self.num_nodes), self.size_frontier)
        deg = self.adj.indptr[frontier + 1] - self.adj.indptr[frontier]
        deg_sum = np.sum(deg)
        edge_idx = []
        for i in range(self.edge_num_subgraph):
            val = np.random.randint(deg_sum)
            id = 0
            while val >= deg[id]:
                val -= deg[id]
                id += 1
            nid = frontier[id]
            idx_s, idx_e = self.adj.indptr[nid], self.adj.indptr[nid + 1]
            if idx_s >= idx_e:
                continue
            e = np.random.randint(idx_s, idx_e)
            edge_idx.append(e)
            v = self.adj.indices[e]
            frontier[id] = v
            deg_sum -= deg[id]
            deg[id] = self.adj.indptr[v + 1] - self.adj.indptr[v]
            deg_sum += deg[id]

        return self.edge_induction(np.array(edge_idx))


class LayerSampler(Sampler):
    def __init__(self, data, model, params_args):
        super().__init__(data, params_args)
        self.model = model
        self.sample_one_layer = self.model._sample_one_layer
        self.sample = self.model.sampling

    def get_batches(self, train_nodes, train_labels, batch_size=64, shuffle=True):
        if shuffle:
            random.shuffle(train_nodes)
        total = train_nodes.shape[0]
        for i in range(0, total, batch_size):
            if i + batch_size <= total:
                cur_nodes = train_nodes[i : i + batch_size]
                cur_labels = train_labels[cur_nodes]
                yield cur_nodes, cur_labels


class NeighborSampler(torch.utils.data.DataLoader):
    def __init__(self, data: Data, sizes: List[int], mask=None, **kwargs):
        self.data = data
        self.sizes = sizes
        node_idx = np.arange(0, data.x.shape[0])
        if mask is not None:
            node_idx = node_idx[mask]
        node_idx = node_idx.tolist()
        super(NeighborSampler, self).__init__(node_idx, collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        """
            Sample a subgraph with neighborhood sampling
        Args:
            batch: torch.Tensor / np.array
                Target nodes
        Returns:
            if `size` is `[-1,]`,
                (
                    source_nodes_id: Tensor,
                    sampled_edges: Tensor,
                    (number_of_source_nodes, number_of_target_nodes): Tuple[int]
                )
            otherwise,
                (
                    target_nodes_id: Tensor
                    all_sampled_nodes_id: Tensor,
                    sampled_adjs: List[Tuple(Tensor, Tensor, Tuple[int]]
                )
        """
        node_id = batch
        adj_list = []
        for size in self.sizes:
            src_id, _edge_index = self.data.sample_adj(node_id, size, replace=False)
            size = (len(src_id), len(node_id))
            adj_list.append((src_id, _edge_index, size))  # src_id, edge_index, (src_size, target_size)
            node_id = src_id
        if self.sizes == [-1]:
            src_id, edge_index, _ = adj_list[0]
            size = (len(src_id), len(batch))
            return src_id, edge_index, size
        else:
            return batch, node_id, adj_list[::-1]


class ClusteredLoader(torch.utils.data.DataLoader):
    partition_tool = None

    def __init__(self, data: Data, n_cluster: int, **kwargs):
        try:
            import metis

            ClusteredLoader.partition_tool = metis
        except Exception as e:
            print(e)
            exit(1)

        self.data = data
        self.clusters = self.preprocess(n_cluster)
        super(ClusteredLoader, self).__init__(list(range(n_cluster)), collate_fn=self.batcher, **kwargs)

    def preprocess(self, n_cluster):
        print("Preprocessing...")
        edges = self.data.edge_index
        edges, _ = remove_self_loops(edges)
        if str(edges.device) != "cpu":
            edges = edges.cpu()
        edges = edges.numpy()
        num_nodes = np.max(edges) + 1
        adj = sp.csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(num_nodes, num_nodes))
        indptr = adj.indptr
        indptr = np.split(adj.indices, indptr[1:])[:-1]
        _, parts = ClusteredLoader.partition_tool.part_graph(indptr, n_cluster, seed=1)
        division = [[] for _ in range(n_cluster)]
        for i, v in enumerate(parts):
            division[v].append(i)
        for k in range(len(division)):
            division[k] = np.array(division[k], dtype=np.int)
        print("Graph clustering over")
        return division

    def batcher(self, batch):
        nodes = np.concatenate([self.clusters[i] for i in batch])
        subgraph = self.data.subgraph(nodes)
        return subgraph
