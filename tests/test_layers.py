import torch
from cogdl.data import Graph
from cogdl.layers import BaseLayer, GINELayer


def test_base_layer():
    layer = BaseLayer()
    x = torch.eye(4)
    edge_index = (torch.tensor([0, 0, 0, 1, 1, 2]), torch.tensor([1, 2, 3, 2, 3, 3]))
    graph = Graph(x=x, edge_index=edge_index)
    x = layer(graph, x)
    assert tuple(x.shape) == (4, 4)


def test_gine_layer():
    layer = GINELayer()
    x = torch.eye(4)
    edge_index = (torch.tensor([0, 0, 0, 1, 1, 2]), torch.tensor([1, 2, 3, 2, 3, 3]))
    graph = Graph(x=x, edge_index=edge_index, edge_attr=torch.randn(6, 4))
    x = layer(graph, x)
    assert tuple(x.shape) == (4, 4)


if __name__ == "__main__":
    test_base_layer()
    test_gine_layer()
