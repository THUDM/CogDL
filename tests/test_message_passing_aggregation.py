import torch
from cogdl.data import Graph


def test_data_inner_message_passing_aggregate(node_feats, node_num, edge_num):
    x = torch.rand(node_num, node_feats)
    edge_index = (torch.randint(0, node_num, (edge_num, )), torch.randint(0, node_num, (edge_num, )))
    graph = Graph(x=x, edge_index=edge_index)
    # m = graph.message_passing('u_add_v', x)
    m = graph.message_passing('u_mul_e', x)
    x = graph.aggregate('sum', x, m)
    print(x)


if __name__ == '__main__':
    node_feats = 512
    edge_num = 1000
    node_num = 500
    test_data_inner_message_passing_aggregate(node_feats, node_num, edge_num)
