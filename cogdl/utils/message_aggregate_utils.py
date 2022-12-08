import torch
from torch_scatter import scatter_max, scatter_min, scatter_add, scatter_mean


class MessageBuiltinFunction:
    # keep original source features
    @staticmethod
    def copy_u(x, src_node_id, dst_node_id, edge_weight):
        return x[src_node_id]

    @staticmethod
    def copy_e(x, src_node_id, dst_node_id, edge_weight):
        return edge_weight

    @staticmethod
    def copy_v(x, src_node_id, dst_node_id, edge_weight):
        return x[dst_node_id]

    # source & target
    @staticmethod
    def u_add_v(x, src_node_id, dst_node_id, edge_weight):
        return x[src_node_id] + x[dst_node_id]

    @staticmethod
    def u_sub_v(x, src_node_id, dst_node_id, edge_weight):
        return x[src_node_id] - x[dst_node_id]

    @staticmethod
    def u_mul_v(x, src_node_id, dst_node_id, edge_weight):
        return torch.mul(x[src_node_id], x[dst_node_id])

    @staticmethod
    def u_div_v(x, src_node_id, dst_node_id, edge_weight):
        return torch.div(x[src_node_id], x[dst_node_id])

    # source & edge weight
    @staticmethod
    def u_add_e(x, src_node_id, dst_node_id, edge_weight):
        return x[src_node_id] + edge_weight

    @staticmethod
    def u_sub_e(x, src_node_id, dst_node_id, edge_weight):
        return x[src_node_id] - edge_weight

    @staticmethod
    def u_mul_e(x, src_node_id, dst_node_id, edge_weight):
        return torch.mul(x[src_node_id], edge_weight)

    @staticmethod
    def u_div_e(x, src_node_id, dst_node_id, edge_weight):
        return torch.div(x[src_node_id], edge_weight)

    # target & source
    @staticmethod
    def v_add_u(x, src_node_id, dst_node_id, edge_weight):
        return x[dst_node_id] + x[src_node_id]

    @staticmethod
    def v_sub_u(x, src_node_id, dst_node_id, edge_weight):
        return x[dst_node_id] - x[src_node_id]

    @staticmethod
    def v_mul_u(x, src_node_id, dst_node_id, edge_weight):
        return torch.mul(x[dst_node_id], x[src_node_id])

    @staticmethod
    def v_div_u(x, src_node_id, dst_node_id, edge_weight):
        return torch.div(x[dst_node_id], x[src_node_id])

    # target & edge weight
    @staticmethod
    def v_add_e(x, src_node_id, dst_node_id, edge_weight):
        return x[dst_node_id] + edge_weight

    @staticmethod
    def v_sub_e(x, src_node_id, dst_node_id, edge_weight):
        return x[dst_node_id] - edge_weight

    @staticmethod
    def v_mul_e(x, src_node_id, dst_node_id, edge_weight):
        return torch.mul(x[dst_node_id], edge_weight)

    @staticmethod
    def v_div_e(x, src_node_id, dst_node_id, edge_weight):
        return torch.div(x[dst_node_id], edge_weight)

    # edge weight & source
    @staticmethod
    def e_add_u(x, src_node_id, dst_node_id, edge_weight):
        return edge_weight + x[src_node_id]

    @staticmethod
    def e_sub_u(x, src_node_id, dst_node_id, edge_weight):
        return edge_weight - x[src_node_id]

    @staticmethod
    def e_mul_u(x, src_node_id, dst_node_id, edge_weight):
        return torch.mul(edge_weight, x[src_node_id])

    @staticmethod
    def e_div_u(x, src_node_id, dst_node_id, edge_weight):
        return torch.div(edge_weight, x[src_node_id])

    # edge weight & target
    @staticmethod
    def e_add_v(x, src_node_id, dst_node_id, edge_weight):
        return edge_weight + x[dst_node_id]

    @staticmethod
    def e_sub_v(x, src_node_id, dst_node_id, edge_weight):
        return edge_weight - x[dst_node_id]

    @staticmethod
    def e_mul_v(x, src_node_id, dst_node_id, edge_weight):
        return torch.mul(edge_weight, x[dst_node_id])

    @staticmethod
    def e_div_v(x, src_node_id, dst_node_id, edge_weight):
        return torch.div(edge_weight, x[dst_node_id])

    # dot manipulation
    @staticmethod
    def u_dot_v(x, src_node_id, dst_node_id, edge_weight):
        return torch.mm(x[src_node_id], x[dst_node_id])

    @staticmethod
    def u_dot_e(x, src_node_id, dst_node_id, edge_weight):
        return torch.mm(x[src_node_id], edge_weight)

    @staticmethod
    def v_dot_e(x, src_node_id, dst_node_id, edge_weight):
        return torch.mm(x[dst_node_id], edge_weight)

    @staticmethod
    def v_fot_u(x, src_node_id, dst_node_id, edge_weight):
        return torch.mm(x[dst_node_id], x[src_node_id])

    @staticmethod
    def e_dot_u(x, src_node_id, dst_node_id, edge_weight):
        return torch.mm(edge_weight, x[src_node_id])

    @staticmethod
    def e_dot_v(x, src_node_id, dst_node_id, edge_weight):
        return torch.mm(edge_weight, x[dst_node_id])


class AggregateBuiltinFunction:
    @staticmethod
    def sum(src, index, out):
        out = scatter_add(src, index, out=out, dim=0)
        return out

    @staticmethod
    def mean(src, index, out):
        out = scatter_mean(src, index, out=out, dim=0)
        return out

    @staticmethod
    def max(src, index, out):
        out = scatter_max(src, index, out=out, dim=0)
        return out

    @staticmethod
    def min(src, index, out):
        out = scatter_min(src, index, out=out, dim=0)
        return out