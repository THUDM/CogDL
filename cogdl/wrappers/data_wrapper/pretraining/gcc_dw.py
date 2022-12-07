import copy
import math
import operator
from typing import Tuple

from scipy.sparse import linalg
from sklearn.model_selection import StratifiedKFold

import numpy as np
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Sampler, BatchSampler

from .. import DataWrapper
from cogdl.data import batch_graphs, Graph


class GCCDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # random walk
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--rw-hops", type=int, default=256)
        parser.add_argument("--subgraph-size", type=int, default=128)
        parser.add_argument("--restart-prob", type=float, default=0.8)
        parser.add_argument("--positional-embedding-size", type=int, default=32)
        parser.add_argument(
            "--task", type=str, default="node_classification", choices=["node_classification, graph_classification"]
        )
        parser.add_argument("--num-workers", type=int, default=12)
        parser.add_argument("--num-copies", type=int, default=6)
        parser.add_argument("--num-samples", type=int, default=2000)
        parser.add_argument("--aug", type=str, default="rwr")
        parser.add_argument("--parallel", type=bool, default=True)

    def __init__(
        self,
        dataset,
        batch_size,
        finetune=False,
        num_workers=4,
        rw_hops=256,
        subgraph_size=128,
        restart_prob=0.8,
        positional_embedding_size=32,
        task="node_classification",
        freeze=False,
        pretrain=False,
        num_samples=0,
        num_copies=1,
        aug="rwr",
        num_neighbors=5,
        parallel=True
    ):
        super(GCCDataWrapper, self).__init__(dataset)
        
        if pretrain:
            data = dataset.data.graphs
        else:
            data = dataset
    
        if task == "node_classification":
            if finetune:
                finetune_dataset = NodeClassificationDatasetLabeled(
                    data, rw_hops, subgraph_size, restart_prob, positional_embedding_size
                )
            elif freeze:
                self.train_dataset = NodeClassificationDataset(
                    data, rw_hops, subgraph_size, restart_prob, positional_embedding_size
                )
            else:
                self.train_dataset = LoadBalanceGraphDataset(
                    data, rw_hops, restart_prob, positional_embedding_size,
                    num_workers, num_samples, num_copies, aug, num_neighbors, parallel
                )

        if finetune:
            graph = dataset.data
            labels = graph.y
            if len(labels.shape) != 1:
                labels = graph.y.argmax(dim=1).tolist()
            if hasattr(graph, "train_mask") and hasattr(graph, "test_mask"):
                train_idx = torch.where(graph.train_mask == 1)[0]
                test_idx = torch.where(graph.test_mask == 1)[0]
            else:
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
                idx_list = []
                for idx in skf.split(np.zeros(len(labels)), labels):
                    idx_list.append(idx)
                train_idx, test_idx = idx_list[0]
            self.train_dataset = torch.utils.data.Subset(finetune_dataset, train_idx)
            self.test_dataset = torch.utils.data.Subset(finetune_dataset, test_idx)
        
        elif task == "graph_classification":
            if finetune:
                pass
            else:
                pass
        
        self.batch_size = dataset.data.num_nodes if freeze else batch_size
        self.num_workers = num_workers
        self.finetune = finetune

        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        
        # self.num_nodes = data.num_nodes
        self.num_nodes = len(self.train_dataset)
        self.freeze = freeze 
        self.pretrain = pretrain
        self.num_samples = num_samples
        
    def train_wrapper(self):
        if self.pretrain:
            batcher_ = batcher()
        elif self.freeze:
            batcher_ = labeled_batcher_double_graphs()
        elif self.finetune:
            batcher_ = labeled_batcher_single_graph()
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=batcher_,
            shuffle=True if self.finetune else False,
            num_workers=self.num_workers,
            worker_init_fn=None if not self.pretrain else worker_init_fn,
        )
        return train_loader
    
    def test_wrapper(self):
        if self.pretrain:
            pass
        elif self.freeze:
            return self.ge_wrapper()
        elif self.finetune:
            test_loader = torch.utils.data.DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size,
                collate_fn=labeled_batcher_single_graph(),
                shuffle=True,
                num_workers=self.num_workers,
            )
            return test_loader
    
    def ge_wrapper(self):
        return self.train_wrapper()    


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    graphs = dataset.graphs
    selected_graphids = dataset.jobs[worker_id]
    dataset.step_graphs = []
    for graphid in selected_graphids:
        # g = copy.deepcopy(graphs[graphid])
        g = graphs[graphid]
        dataset.step_graphs.append(g)
    dataset.length = sum([g.num_nodes for g in dataset.step_graphs])
    dataset.degrees = [g.degrees() for g in dataset.step_graphs]
    np.random.seed(worker_info.seed % (2 ** 32))


def labeled_batcher_single_graph():  # for finetune
    def batcher_dev(batch):
        graph_q_, label_ = zip(*batch)
        graph_q = batch_graphs(graph_q_)
        graph_q.batch_size = len(graph_q_)
        return graph_q, torch.LongTensor(label_)
    return batcher_dev


def labeled_batcher_double_graphs():  # for freeze
    def batcher_dev(batch):
        graph_q_, graph_k_, label_ = zip(*batch)
        graph_q, graph_k = batch_graphs(graph_q_), batch_graphs(graph_k_)
        graph_q.batch_size = len(graph_q_)
        return graph_q, graph_k, torch.LongTensor(label_)
    return batcher_dev


def batcher():  # for pretrain
    def batcher_dev(batch):
        graph_q_, graph_k_ = zip(*batch)
        graph_q, graph_k = batch_graphs(graph_q_), batch_graphs(graph_k_)
        graph_q.batch_size = len(graph_q_)
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


def _add_undirected_graph_positional_embedding(g: Graph, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.num_nodes
    with g.local_graph():
        g.sym_norm()
        adj = g.to_scipy_csr()
    laplacian = adj

    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.pos_undirected = x.float()
    return g


def _rwr_trace_to_cogdl_graph(
    g: Graph, seed: int, trace: torch.Tensor, positional_embedding_size: int, entire_graph: bool = False
):
    subv = torch.unique(trace).tolist()
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

    subg.seed = torch.zeros(subg.num_nodes, dtype=torch.long)
    if entire_graph:
        subg.seed[seed] = 1
    else:
        subg.seed[0] = 1
    return subg


class LoadBalanceGraphDataset(torch.utils.data.IterableDataset):  # for pretrain
    def __init__(
        self,
        data,
        rw_hops=256,
        restart_prob=0.8,
        positional_embedding_size=32,
        num_workers=1,
        num_samples=10000,
        num_copies=1,
        aug="rwr",
        num_neighbors=5,
        parallel=True,
        graph_transform=None,
        step_dist=[1.0, 0.0, 0.0],
    ):
        super(LoadBalanceGraphDataset).__init__()
        self.graphs = data
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        self.parallel = parallel
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        graph_sizes = [graph.num_nodes for graph in self.graphs]

        # print("load graph done")

        # a simple greedy algorithm for load balance
        # sorted graphs w.r.t its size in decreasing order
        # for each graph, assign it to the worker with least workload
        assert num_workers % num_copies == 0
        jobs = [list() for i in range(num_workers // num_copies)]
        workloads = [0] * (num_workers // num_copies)
        graph_sizes = sorted(
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        )
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.jobs = jobs * num_copies
        self.total = self.num_samples * num_workers
        self.graph_transform = graph_transform
        assert aug in ("rwr", "ns")
        self.aug = aug
        self.num_workers = num_workers

    def __len__(self):
        return self.num_samples * self.num_workers

    def __iter__(self):
        degrees = torch.cat([g.degrees().double() ** 0.75 for g in self.step_graphs])
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob.numpy()
        )
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.step_graphs)):
            if node_idx < self.step_graphs[i].num_nodes:
                graph_idx = i
                break
            else:
                node_idx -= self.step_graphs[i].num_nodes
        
        g = self.step_graphs[graph_idx]
        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = g.random_walk([node_idx], step)[-1]

        if self.aug == "rwr":
            max_nodes_per_seed = max(
                self.rw_hops,
                int((self.degrees[graph_idx][node_idx] * math.e / (math.e - 1) / self.restart_prob) + 0.5),
            )
            traces = g.random_walk_with_restart([node_idx, other_node_idx], max_nodes_per_seed, self.restart_prob, self.parallel)

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
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)    
        return graph_q, graph_k


class NodeClassificationDataset(object):  # for freeze
    def __init__(
        self,
        data,  # Graph
        rw_hops: int = 64,
        subgraph_size: int = 64,
        restart_prob: float = 0.8,
        positional_embedding_size: int = 32,
        step_dist: list = [1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert positional_embedding_size > 1

        self.data = data.data
        self.graphs = self.data if isinstance(self.data, list) else [self.data]
        self.length = sum([g.num_nodes for g in self.graphs])
        self.total = self.length

    def __len__(self):
        return self.length

    def _convert_idx(self, idx) -> Tuple[int, int]:
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
            other_node_idx = g.random_walk([node_idx], step)[-1]

        max_nodes_per_seed = max(
            self.rw_hops,
            int((self.graphs[graph_idx].degrees()[node_idx] * math.e / (math.e - 1) / self.restart_prob) + 0.5),
        )
        # NOTICE: Please check the version of numpy and numba in README
        traces = g.random_walk_with_restart([node_idx, other_node_idx], max_nodes_per_seed, self.restart_prob)

        # traces = [[0,1,2,3], [1,2,3,4]]

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
        y = self.data.y[idx].argmax().item() if len(self.data.y.shape) != 1 else self.data.y[idx]
        return graph_q, graph_k, y


class NodeClassificationDatasetLabeled(NodeClassificationDataset):  # for finetune
    def __init__(
        self,
        data,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
    ):
        super(NodeClassificationDatasetLabeled, self).__init__(
            data, rw_hops, subgraph_size, restart_prob, positional_embedding_size, step_dist,
        )
        assert len(self.graphs) == 1
        self.num_classes = self.data.num_classes

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].num_nodes:
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].num_nodes

        g = self.graphs[graph_idx]
        traces = g.random_walk_with_restart([node_idx], self.rw_hops, self.restart_prob)

        graph_q = _rwr_trace_to_cogdl_graph(
            g=g, seed=node_idx, trace=torch.Tensor(traces[0]), positional_embedding_size=self.positional_embedding_size,
        )
        y = self.data.y[idx].argmax().item() if len(self.data.y.shape) != 1 else self.data.y[idx]
        return graph_q, y
