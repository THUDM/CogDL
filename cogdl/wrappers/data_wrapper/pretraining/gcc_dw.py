import copy
import math

from scipy.sparse import linalg

import numpy as np
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from .. import register_data_wrapper, DataWrapper
from cogdl.data import batch_graphs


@register_data_wrapper("gcc_dw")
class GCCDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # random walk
        parser.add_argument("--rw-hops", type=int, default=256)
        parser.add_argument("--subgraph-size", type=int, default=128)
        parser.add_argument("--restart-prob", type=float, default=0.8)
        parser.add_argument("--positional-embedding-size", type=int, default=128)
        parser.add_argument(
            "--task", type=str, default="node_classification", choices=["node_classification, graph_classification"]
        )

    def __init__(
        self,
        dataset,
        batch_size,
        finetune=False,
        num_workers=4,
        rw_htops=4,
        subgraph_size=128,
        restart_prob=0.8,
        position_embedding_size=128,
        task="node_classification",
    ):
        super(GCCDataWrapper, self).__init__(dataset)

        data = dataset.data
        if task == "node_classification":
            if finetune:
                self.train_dataset = NodeClassificationDatasetLabeled(
                    data, rw_htops, subgraph_size, restart_prob, position_embedding_size
                )
            else:
                self.train_dataset = NodeClassificationDataset(
                    data, rw_htops, subgraph_size, restart_prob, position_embedding_size
                )
        elif task == "graph_classification":
            if finetune:
                pass
            else:
                pass
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.finetune = finetune

        self.rw_hops = rw_htops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob

    def training_wrapper(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=labeled_batcher() if self.finetune else batcher(),
            shuffle=True if self.finetune else False,
            num_workers=self.num_workers,
            worker_init_fn=None,
        )
        return train_loader


def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = batch_graphs(graph_q)
        return graph_q, torch.LongTensor(label)

    return batcher_dev


def batcher():
    def batcher_dev(batch):
        graph_q, graph_k = zip(*batch)
        graph_q, graph_k = batch_graphs(graph_q), batch_graphs(graph_k)
        return graph_q, graph_k

    return batcher_dev


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    with g.local_graph():
        g.sym_norm()
        adj = g.to_scipy_adj()
    laplacian = adj

    # adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    # norm = sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    # laplacian = norm * adj * norm

    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    # g.ndata["pos_undirected"] = x.float()
    g.pos_undirected = x.float()
    return g


def _rwr_trace_to_cogdl_graph(g, seed, trace, positional_embedding_size, entire_graph=False):
    subv = torch.unique(torch.cat(trace)).tolist()
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    if entire_graph:
        subg = copy.deepcopy(g)
    else:
        subg = g.subgraph(subv)

    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)

    subg.seed = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.seed[seed] = 1
    else:
        subg.seed[0] = 1
    return subg


class NodeClassificationDataset(object):
    def __init__(
        self,
        data,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert positional_embedding_size > 1

        self.data = data
        self.graphs = [self.data]
        self.length = sum([g.num_nodes for g in self.graphs])
        self.total = self.length

    def __len__(self):
        return self.length

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].num_nodes:
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].num_nodes
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        g = self.graphs[graph_idx]

        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = g.random_walk(start=[node_idx], length=step)[-1]

        max_nodes_per_seed = max(
            self.rw_hops,
            int((self.graphs[graph_idx].out_degree(node_idx) * math.e / (math.e - 1) / self.restart_prob) + 0.5),
        )
        traces = g.random_walk_with_restart([node_idx, other_node_idx], max_nodes_per_seed, self.restart_prob)

        graph_q = _rwr_trace_to_cogdl_graph(
            g=g,
            seed=node_idx,
            trace=torch.Tensor(traces[0]),
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        graph_k = _rwr_trace_to_cogdl_graph(
            g=g,
            seed=other_node_idx,
            trace=torch.Tensor(traces[1]),
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        return graph_q, graph_k


class NodeClassificationDatasetLabeled(NodeClassificationDataset):
    def __init__(
        self,
        dataset,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
    ):
        super(NodeClassificationDatasetLabeled, self).__init__(
            dataset,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
        )
        assert len(self.graphs) == 1
        self.num_classes = self.data.y.shape[1]

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        g = self.graphs[graph_idx]
        traces = g.random_walk_with_restart([node_idx], self.rw_hops, self.restart_prob)

        graph_q = _rwr_trace_to_cogdl_graph(
            g=g,
            seed=node_idx,
            trace=torch.Tensor(traces[0]),
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_q.y = self.data.y[idx].y
        return graph_q
        # return graph_q, self.data.y[idx].argmax().item()
