import re
import torch
from inspect import isfunction
from cogdl.data.data import Graph
from cogdl.utils.message_aggregate_utils import MessageBuiltinFunction, AggregateBuiltinFunction


class HeteroGraph(Graph):
    def __init__(self, x=None, y=None, **kwargs):
        super(HeteroGraph, self).__init__(x, y, **kwargs)
        if 'node_type' in kwargs.keys():
            self.node_type = kwargs['node_type']
            assert isinstance(self.node_type, dict)
            for k, v in self.node_type.items():
                if not isinstance(v, torch.Tensor) or v.dtype != torch.int64:
                    raise Exception('Each value of node type must be tensor type and in int data type')
        else:
            self.node_type = None

        # 异构图上对边的定义
        if 'edge_type' in kwargs.keys():
            self.edge_type = kwargs['edge_type']
            assert isinstance(self.edge_type, dict)
            for k, v in self.edge_type.items():
                if not isinstance(v, torch.Tensor) or v.dtype != torch.int64:
                    raise Exception('Each value of node type must be tensor type and in int data type')
        else:
            self.edge_type = None


    def build_hetero_mask(self, src_node_id, dst_node_id, src_node_type, dst_node_type):
        r"""
        src_node_id, dst_node_id: 需要进行消息传递的边上的起点和终点
        src_node type, dst_node_type: 消息传递的节点的
        """
        src_node_index = [i for i in range(len(src_node_id)) if src_node_id[i] in self.node_type[src_node_type]]
        dst_node_index = [i for i in range(len(dst_node_id)) if dst_node_id[i] in self.node_type[dst_node_type]]
        select_node_index = torch.Tensor(list(set(src_node_index).intersection(set(dst_node_index)))).to(torch.int64)
        if len(select_node_index) == 0:
            raise Warning('No nodes are selected according to the selection condition')
        src_node_id = src_node_id.index_select(0, select_node_index)
        dst_node_id = dst_node_id.index_select(0, select_node_index)
        return src_node_id, dst_node_id


    def message_passing(self, msg_func, x, edge_weight=None, **kwargs):
        src_node_id = self.edge_index[0]
        dst_node_id = self.edge_index[1]

        if 'edge_type' in kwargs.keys():
            # 按照预定义的边进行聚合
            edge_type = kwargs['edge_type']
            if self.edge_type is None or edge_type not in self.edge_type.keys():
                raise Exception('This heterograph does not has predefined edge types')
            edge_type_index = self.edge_type[edge_type]
            src_node_id = self.edge_index[0].index_select(0, edge_type_index)
            dst_node_id = self.edge_index[1].index_select(0, edge_type_index)
        else:
            # 按照预定义的点进行聚合
            if 'src_node_type' in kwargs.keys() and 'dst_node_type' in kwargs.keys():
                src_node_type = kwargs['src_node_type']
                dst_node_type = kwargs['dst_node_type']
            else:
                raise Exception('Lack of Arguments: "src_node_type" or "dst_node_type"')
            if self.node_type is None or src_node_type not in self.node_type.keys() or dst_node_type not in self.node_type.keys():
                raise Exception('This heterograph does not has predefined node types')
            src_node_id, dst_node_id = self.build_hetero_mask(src_node_id, dst_node_id, src_node_type, dst_node_type)

        if not isfunction(msg_func) and not isinstance(msg_func, str):
            raise RuntimeError('Only Support Message Functions and String Prompt')
        if isinstance(msg_func, str):
            if not hasattr(MessageBuiltinFunction, msg_func):
                raise NotImplementedError
            msg_func = getattr(MessageBuiltinFunction, msg_func)
        if edge_weight is None:
            edge_weight = torch.ones(len(src_node_id), x.shape[1])
        m = msg_func(x, src_node_id, dst_node_id, edge_weight)
        return m

    def aggregate(self, agg_func, x, m, **kwargs):
        src_node_id = self.edge_index[0]
        dst_node_id = self.edge_index[1]
        if 'edge_type' in kwargs.keys():
            # 按照预定义的边进行聚合
            edge_type = kwargs['edge_type']
            if self.edge_type is None or edge_type not in self.edge_type.keys():
                raise Exception('This heterograph does not has predefined edge types')
            edge_type_index = self.edge_type[edge_type]
            src_node_id = self.edge_index[0].index_select(0, edge_type_index)
            dst_node_id = self.edge_index[1].index_select(0, edge_type_index)
        else:
            # 按照预定义的点进行聚合
            if 'src_node_type' in kwargs.keys() and 'dst_node_type' in kwargs.keys():
                src_node_type = kwargs['src_node_type']
                dst_node_type = kwargs['dst_node_type']
            else:
                raise Exception('Lack of Arguments: "src_node_type" or "dst_node_type"')
            if self.node_type is None or src_node_type not in self.node_type.keys() or dst_node_type not in self.node_type.keys():
                raise Exception('This heterograph does not has predefined node types')
            src_node_id, dst_node_id = self.build_hetero_mask(src_node_id, dst_node_id, src_node_type, dst_node_type)
        out = torch.zeros(x.shape[0], m.shape[1], dtype=x.dtype).to(x.device)
        index = dst_node_id.unsqueeze(1).expand(-1, m.shape[1])
        src = m
        if not isfunction(agg_func) and not isinstance(agg_func, str):
            raise RuntimeError('Only Support Message Functions and String Prompt')
        if isinstance(agg_func, str):
            if not hasattr(AggregateBuiltinFunction, agg_func):
                raise NotImplementedError
            agg_func = getattr(AggregateBuiltinFunction, agg_func)
        h = agg_func(src, index, out=out)
        return h

# def test_hetero_data_inner_message_passing_aggregate(node_feats, node_num, edge_num):
#     x = torch.rand(node_num, node_feats)
#     edge_index = (torch.randint(0, node_num, (edge_num,)), torch.randint(0, node_num, (edge_num,)))
#     edge_type = {
#         'l': torch.randint(0, edge_num, (int(0.5 * edge_num), )),
#         'r': torch.randint(0, edge_num, (int(0.5 * edge_num), )),
#     }
#     node_type = {
#         'x':  torch.randint(0, node_num, (int(0.5 * node_num), )),
#         'y': torch.randint(0, node_num, (int(0.4 * node_num),)),
#     }
#     hetero_graph = HeteroGraph(x=x, edge_index=edge_index, edge_type=edge_type, node_type=node_type)
#     # 基于边的异构
#     # m = hetero_graph.message_passing('u_mul_e', x, edge_type='l')
#     # x = hetero_graph.aggregate('sum', x, m, edge_type='l')
#     # 基于点的异构
#     m = hetero_graph.message_passing('u_mul_e', x, src_node_type='x', dst_node_type='y')
#     x = hetero_graph.aggregate('sum', x, m, src_node_type='x', dst_node_type='y')
#     print(x)
#
#
#
# if __name__ == '__main__':
#     node_feats = 512
#     edge_num = 1000
#     node_num = 500
#     test_hetero_data_inner_message_passing_aggregate(node_feats, node_num, edge_num)