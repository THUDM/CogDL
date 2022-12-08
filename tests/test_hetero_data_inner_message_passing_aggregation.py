import torch
from cogdl.data import HeteroGraph


def test_hetero_data_inner_message_passing_aggregate(node_feats, node_num, edge_num):
    x = torch.rand(node_num, node_feats)
    edge_index = (torch.randint(0, node_num, (edge_num,)), torch.randint(0, node_num, (edge_num,)))
    edge_type = {
        'l': torch.randint(0, edge_num, (int(0.5 * edge_num), )),
        'r': torch.randint(0, edge_num, (int(0.5 * edge_num), )),
    }
    node_type = {
        'x':  torch.randint(0, node_num, (int(0.5 * node_num), )),
        'y': torch.randint(0, node_num, (int(0.4 * node_num),)),
    }
    hetero_graph = HeteroGraph(x=x, edge_index=edge_index, edge_type=edge_type, node_type=node_type)
    # 基于边的异构
    # m = hetero_graph.message_passing('u_mul_e', x, edge_type='l')
    # x = hetero_graph.aggregate('sum', x, m, edge_type='l')
    # 基于点的异构
    m = hetero_graph.message_passing('u_mul_e', x, src_node_type='x', dst_node_type='y')
    x = hetero_graph.aggregate('sum', x, m, src_node_type='x', dst_node_type='y')
    print(x)



if __name__ == '__main__':
    node_feats = 512
    edge_num = 1000
    node_num = 500
    test_hetero_data_inner_message_passing_aggregate(node_feats, node_num, edge_num)