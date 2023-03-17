from tracemalloc import start
from turtle import pos
from typing import List
import os
import random
import numpy as np
import scipy.sparse as sp
from cogdl.utils import remove_self_loops, row_normalization, RandomWalker
from .data import Graph
import jittor


class NeighborSamplerDataset(jittor.dataset.Dataset):
    def __init__(self, dataset, sizes: List[int], batch_size=8, mask=None, **kwargs):
        super(NeighborSamplerDataset, self).__init__()
        self.graph_batch_size = batch_size
        shuffle = kwargs["shuffle"]
        data_shuffle = kwargs["shuffle"]
        self.data = dataset.data
        self.x = self.data.x
        self.y = self.data.y
        self.sizes = sizes
        self.node_idx = jittor.arange(0, self.data.x.shape[0]).long()
        if mask is not None:
            self.node_idx = self.node_idx[mask]
        self.num_nodes = self.node_idx.shape[0]
        total_len = (self.num_nodes - 1) // self.graph_batch_size + 1
        if data_shuffle:
            idx = jittor.randperm(self.num_nodes)
            self.node_idx = self.node_idx[idx]
        self.set_attrs(total_len=total_len, batch_size=1, shuffle=shuffle)

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
        batch = self.node_idx[idx * self.graph_batch_size : (idx + 1) * self.graph_batch_size]
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

    @staticmethod
    def collate_batch(data):
        return data[0]


class UnsupNeighborSamplerDataset(jittor.dataset.Dataset):
    def __init__(self, dataset, sizes: List[int], batch_size=8, mask=None, **kwargs):
        super(UnsupNeighborSamplerDataset, self).__init__()
        self.graph_batch_size = batch_size
        shuffle = kwargs["shuffle"]
        data_shuffle = kwargs["shuffle"]
        self.data = dataset.data
        self.x = self.data.x
        self.edge_index = self.data.edge_index
        self.sizes = sizes
        self.batch_size = batch_size
        self.node_idx = jittor.arange(0, self.data.x.shape[0]).long()
        self.total_num_nodes = self.num_nodes = self.node_idx.shape[0]
        if mask is not None:
            self.node_idx = self.node_idx[mask]
        self.num_nodes = self.node_idx.shape[0]
        self.random_walker = RandomWalker()
        total_len = (self.num_nodes - 1) // self.graph_batch_size + 1
        if data_shuffle:
            idx = jittor.randperm(self.num_nodes)
            self.node_idx = self.node_idx[idx]
        self.set_attrs(total_len=total_len, batch_size=1, shuffle=shuffle)

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
        batch = self.node_idx[idx * self.graph_batch_size : (idx + 1) * self.graph_batch_size]
        self.random_walker.build_up(self.edge_index, self.total_num_nodes)
        walk_res = self.random_walker.walk_one(batch, length=1, p=0.0)

        neg_batch = jittor.randint(0, self.total_num_nodes, (batch.numel(),)).long()
        pos_batch = jittor.array(walk_res)
        batch = jittor.concat([batch, pos_batch, neg_batch], dim=0)
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

    @staticmethod
    def collate_batch(data):
        return data[0]
