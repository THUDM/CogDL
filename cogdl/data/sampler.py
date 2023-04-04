
from tracemalloc import start
from turtle import pos
from typing import List
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data
from cogdl.utils import remove_self_loops, row_normalization
from cogdl.data import Graph, DataLoader
from cogdl.utils import RandomWalker


class NeighborSampler(DataLoader):
    def __init__(self, dataset, sizes: List[int], mask=None, **kwargs):
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
        else:
            batch_size = 8
        if isinstance(dataset.data, Graph):     
            self.dataset = NeighborSamplerDataset(dataset, sizes, batch_size, mask)
        else:
            self.dataset = dataset
        kwargs["batch_size"] = 1
        kwargs["shuffle"] = False
        kwargs["collate_fn"] = NeighborSampler.collate_fn
        super(NeighborSampler, self).__init__(dataset=self.dataset, **kwargs)

    @staticmethod
    def collate_fn(data):
        return data[0]

    def shuffle(self):
        self.dataset.shuffle()


class UnsupNeighborSampler(DataLoader):
    def __init__(self, dataset, sizes: List[int], mask=None, **kwargs):
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
        else:
            batch_size = 8

        if isinstance(dataset.data, Graph):
            self.dataset = UnsupNeighborSamplerDataset(dataset, sizes, batch_size, mask)
        else:
            self.dataset = dataset
        kwargs["batch_size"] = 1
        kwargs["shuffle"] = False
        kwargs["collate_fn"] = UnsupNeighborSampler.collate_fn
        super(UnsupNeighborSampler, self).__init__(dataset=self.dataset, **kwargs)

    @staticmethod
    def collate_fn(data):
        return data[0]

    def shuffle(self):
        self.dataset.shuffle()

class NeighborSamplerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sizes: List[int], batch_size: int, mask=None):
        super(NeighborSamplerDataset, self).__init__()
        self.data = dataset.data
        self.x = self.data.x
        self.y = self.data.y
        self.sizes = sizes
        self.batch_size = batch_size
        self.node_idx = torch.arange(0, self.data.x.shape[0], dtype=torch.long)
        if mask is not None:
            self.node_idx = self.node_idx[mask]
        self.num_nodes = self.node_idx.shape[0]

    def shuffle(self):
        idx = torch.randperm(self.num_nodes)
        self.node_idx = self.node_idx[idx]

    def __len__(self):
        return (self.num_nodes - 1) // self.batch_size + 1
    
    def __getitem__(self, idx):
        """
            Sample a subgraph with neighborhood sampling
        Args:
            idx: torch.Tensor / np.array
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
        batch = self.node_idx[idx * self.batch_size : (idx + 1) * self.batch_size]
        node_id = batch
        adj_list = []
        for size in self.sizes:
            src_id, graph = self.data.sample_adj(node_id, size, replace=False)
            size = (len(src_id), len(node_id))
            adj_list.append((src_id, graph, size))  # src_id, graph, (src_size, target_size)
            node_id = src_id

        if self.sizes == [-1]:
            src_id, graph, _ = adj_list[0]
            size = (len(src_id), len(batch))
            return src_id, graph, size
        else:
            return batch, node_id, adj_list[::-1]


class UnsupNeighborSamplerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sizes: List[int], batch_size: int, mask=None):
        super(UnsupNeighborSamplerDataset, self).__init__()
        self.data = dataset.data
        self.x = self.data.x
        self.edge_index=self.data.edge_index
        self.sizes = sizes
        self.batch_size = batch_size
        self.node_idx = torch.arange(0, self.data.x.shape[0], dtype=torch.long)
        self.total_num_nodes=self.num_nodes = self.node_idx.shape[0]
        if mask is not None:
            self.node_idx = self.node_idx[mask]
        self.num_nodes = self.node_idx.shape[0]
        self.random_walker = RandomWalker()

    def shuffle(self):
        idx = torch.randperm(self.num_nodes)
        self.node_idx = self.node_idx[idx]

    def __len__(self):
        return (self.num_nodes - 1) // self.batch_size + 1

    def __getitem__(self, idx):
        """
            Sample a subgraph with neighborhood sampling
        Args:
            idx: torch.Tensor / np.array
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
        batch = self.node_idx[idx * self.batch_size : (idx + 1) * self.batch_size]
        self.random_walker.build_up(self.edge_index, self.total_num_nodes)
        walk_res = self.random_walker.walk(
            batch, walk_length=2, parallel=False
        )[:,1]
       
        neg_batch = torch.randint(0, self.total_num_nodes, (batch.numel(), ),
                                  dtype=torch.int64)        
        pos_batch=torch.tensor(walk_res)
        if self.sizes != [-1]:
            batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        node_id = batch
        adj_list = []
        for size in self.sizes:
            src_id, graph = self.data.sample_adj(node_id, size, replace=False)
            size = (len(src_id), len(node_id))
            adj_list.append((src_id, graph, size))  # src_id, graph, (src_size, target_size)
            node_id = src_id

        if self.sizes == [-1]:
            src_id, graph, _ = adj_list[0]
            size = (len(src_id), len(batch))
            return src_id, graph, size
        else:
            return batch, node_id, adj_list[::-1]


class ClusteredDataset(torch.utils.data.Dataset):
    partition_tool = None

    def __init__(self, dataset, n_cluster: int, batch_size: int):
        super(ClusteredDataset).__init__()
        try:
            import metis

            ClusteredDataset.partition_tool = metis
        except Exception as e:
            print(e)
            exit(1)

        self.data = dataset.data
        self.dataset_name = dataset.__class__.__name__
        self.batch_size = batch_size
        self.n_cluster = n_cluster
        self.clusters = self.preprocess(n_cluster)
        self.batch_idx = np.array(range(n_cluster))

    def shuffle(self):
        random.shuffle(self.batch_idx)

    def __len__(self):
        return (self.n_cluster - 1) // self.batch_size + 1

    def __getitem__(self, idx):
        batch = self.batch_idx[idx * self.batch_size : (idx + 1) * self.batch_size]
        nodes = np.concatenate([self.clusters[i] for i in batch])
        subgraph = self.data.subgraph(nodes)
        subgraph.batch = torch.from_numpy(nodes)
        return subgraph

    def preprocess(self, n_cluster):
        save_name = f"{self.dataset_name}-{n_cluster}.cluster"
        if os.path.exists(save_name):
            return torch.load(save_name)
        print("Preprocessing...")
        row, col = self.data.edge_index
        (row, col), _ = remove_self_loops((row, col))
        if str(row.device) != "cpu":
            row = row.cpu().numpy()
            col = col.cpu().numpy()
        num_nodes = max(row.max(), col.max()) + 1
        adj = sp.csr_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
        indptr = adj.indptr
        indptr = np.split(adj.indices, indptr[1:])[:-1]
        _, parts = ClusteredDataset.partition_tool.part_graph(indptr, n_cluster, seed=1)
        division = [[] for _ in range(n_cluster)]
        for i, v in enumerate(parts):
            division[v].append(i)
        for k in range(len(division)):
            division[k] = np.array(division[k], dtype=np.int)
        torch.save(division, save_name)
        print("Graph clustering done")
        return division


class ClusteredLoader(DataLoader):
    def __init__(self, dataset, n_cluster: int, method="metis", **kwargs):
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
        else:
            batch_size = 20

        if isinstance(dataset, ClusteredDataset) or isinstance(dataset, RandomPartitionDataset):
            self.dataset = dataset
        elif isinstance(dataset.data, Graph):
            if method == "metis":
                self.dataset = ClusteredDataset(dataset, n_cluster, batch_size)
            else:
                self.dataset = RandomPartitionDataset(dataset, n_cluster)
        kwargs["batch_size"] = 1
        kwargs["shuffle"] = False
        super(ClusteredLoader, self).__init__(dataset=self.dataset, collate_fn=ClusteredLoader.collate_fn, **kwargs)

    @staticmethod
    def collate_fn(item):
        return item[0]

    def shuffle(self):
        self.dataset.shuffle()


class RandomPartitionDataset(torch.utils.data.Dataset):
    """
    For ClusteredLoader
    """

    def __init__(self, dataset, n_cluster):
        self.data = dataset.data
        self.n_cluster = n_cluster
        self.num_nodes = dataset.data.num_nodes
        self.parts = torch.randint(0, self.n_cluster, size=(self.num_nodes,))

    def __getitem__(self, idx):
        node_cluster = torch.where(self.parts == idx)[0]
        subgraph = self.data.subgraph(node_cluster)
        subgraph.batch = node_cluster
        return subgraph

    def __len__(self):
        return self.n_cluster

    def shuffle(self):
        self.parts = torch.randint(0, self.n_cluster, size=(self.num_nodes,))

