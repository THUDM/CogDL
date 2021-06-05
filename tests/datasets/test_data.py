import numpy as np
import torch

from cogdl.datasets import build_dataset_from_name
from cogdl.utils import get_degrees


class Test_Data(object):
    def setup_class(self):
        self.dataset = build_dataset_from_name("cora")
        self.data = self.dataset[0]
        self.num_nodes = self.data.num_nodes
        self.num_edges = self.data.num_edges
        self.num_features = self.data.num_features
        print("Call Setup")

    def test_subgraph_sampling(self):
        sampled_nodes = np.random.randint(0, self.num_nodes, (100,))
        sampled_nodes = np.unique(sampled_nodes)
        subgraph = self.data.subgraph(sampled_nodes)
        assert subgraph.x.shape[0] == len(set(sampled_nodes))
        assert subgraph.x.shape[1] == self.data.x.shape[1]

    def test_edge_subgraph_sampling(self):
        sampled_edges = np.random.randint(0, self.num_edges, (200,))
        subgraph = self.data.edge_subgraph(sampled_edges, require_idx=False)
        row, col = subgraph.edge_index
        assert row.shape[0] == col.shape[0]
        assert row.shape[0] == len(sampled_edges)

    def test_adj_sampling(self):
        sampled_nodes = np.arange(0, 10)
        edge_index = torch.stack(self.data.edge_index).t().cpu().numpy()
        edge_index = [tuple(x) for x in edge_index]
        print(np.array(edge_index).shape)
        for size in [5, -1]:
            node_idx, sampled_edge_index = self.data.sample_adj(sampled_nodes, size)
            node_idx = node_idx.cpu().numpy()
            assert (set(node_idx) & set(sampled_nodes)) == set(sampled_nodes)

    def test_to_csr(self):
        self.data._adj._to_csr()
        symmetric = self.data.is_symmetric()
        assert symmetric is True
        degrees = self.data.degrees()
        row, col = self.data.edge_index
        _degrees = get_degrees(row, col)
        assert (degrees == _degrees).all()
