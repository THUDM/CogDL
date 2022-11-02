import math
import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_min, scatter_add, scatter_mean


class MessageBuiltinFunction:
    @staticmethod
    def keep_source(x, src_node_id, dst_node_id, edge_weight):
        return x[src_node_id]

    @staticmethod
    def keep_target(x, src_node_id, dst_node_id, edge_weight):
        return x[dst_node_id]

    @staticmethod
    def source_add_target(x, src_node_id, dst_node_id, edge_weight):
        return x[src_node_id] + x[dst_node_id]

    @staticmethod
    def source_sub_target(x, src_node_id, dst_node_id, edge_weight):
        return x[src_node_id] - x[dst_node_id]

    @staticmethod
    def source_mul_target(x, src_node_id, dst_node_id, edge_weight):
        return torch.mul(x[src_node_id], x[dst_node_id])

    @staticmethod
    def source_dot_target(x, src_node_id, dst_node_id, edge_weight):
        return torch.mm(x[src_node_id], x[dst_node_id])

    @staticmethod
    def source_div_target(x, src_node_id, dst_node_id, edge_weight):
        return torch.div(x[src_node_id], x[dst_node_id])


class AggregateBuiltinFunction:
    @staticmethod
    def add(src, index, out):
        out = scatter_add(src, index, out=out, dim=0)
        return out

    @staticmethod
    def mean(src, index, out):
        out = scatter_mean(src, index, out=out, dim=0)
        return out


class UserDefinedLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs) -> None:
        super().__init__(**kwargs)
        self.message_builtin_function = MessageBuiltinFunction()
        self.aggregate_builtin_function = AggregateBuiltinFunction()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_feat_lin = nn.Linear(in_dim, out_dim)
        if 'message_prompt' in kwargs.keys():
            self.message_prompt = kwargs['message_prompt']
        else:
            self.message_prompt = 'keep_source'

        if 'aggregate_prompt' in kwargs.keys():
            self.aggregate_prompt = kwargs['aggregate_prompt']
        else:
            self.aggregate_prompt = 'add'
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.uniform_(self.node_feat_lin.weight, -stdv, stdv)

    def forward(self, graph, x, edge_weight=None):
        x = self.node_feat_lin(x)
        src_node_id = graph.edge_index[0]
        dst_node_id = graph.edge_index[1]
        m = self.message(x, src_node_id, dst_node_id, edge_weight)
        h = self.aggregate(x, m, dst_node_id)
        return h

    def message(self, x, src_node_id, dst_node_id, edge_weight=None):
        msg_func = getattr(self.message_builtin_function, self.message_prompt)
        m = msg_func(x, src_node_id, dst_node_id, edge_weight)
        return m

    def aggregate(self, x, m, dst_node_id):
        agg_func = getattr(self.aggregate_builtin_function, self.aggregate_prompt)
        out = torch.zeros(x.shape[0], m.shape[1], dtype=x.dtype).to(x.device)
        index = dst_node_id.unsqueeze(1).expand(-1, m.shape[1])
        src = m
        h = agg_func(src, index, out=out)
        return h

