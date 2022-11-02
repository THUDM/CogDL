import torch
from cogdl.data import Graph
from cogdl.layers.user_defined_layer import UserDefinedLayer

if __name__ == '__main__':
    udf_layer = UserDefinedLayer(in_dim=64, out_dim=32)
    num_nodes = 100
    num_edges = 300
    feat_dim = 64
    # load or generate your dataset
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    x = torch.randn(num_nodes, feat_dim)
    y = torch.randint(0, 2, (num_nodes,))
    graph = Graph(x=x, edge_index=edge_index, y=y)
    x_1 = udf_layer(graph, x)
    print(x_1)


