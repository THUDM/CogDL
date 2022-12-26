from cogdl.operators import (
    s_add_e,
    s_mul_e,
    s_sub_e,
    s_add_e_sum,
    s_sub_e_sum,
    s_mul_e_sum,
    s_add_e_mean,
    s_sub_e_mean,
    s_mul_e_mean,
    s_sub_t,
    s_add_t,
    s_mul_t,
    s_dot_t,
    s_div_t,
)
from cogdl.data import Graph
import torch


def build_toy_data():
    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 200))
    g = Graph(x=x, edge_index=edge_index)
    nedge = g.num_edges
    edge_attr = torch.randn(nedge, 10)
    g.edge_attr = edge_attr
    return g


def test_s_ops_t():
    g = build_toy_data()
    row, col = g.edge_index
    src = g.x[col]
    dst = g.x[row]

    exp = src + dst
    out = s_add_t(g, g.x)
    assert (out == exp).all()

    exp = src - dst
    out = s_sub_t(g, g.x)
    assert (out == exp).all()

    exp = src * dst
    out = s_mul_t(g, g.x)
    assert (out == exp).all()

    exp = src / dst
    out = s_div_t(g, g.x)
    assert (out == exp).all()

    exp = (src * dst).sum(dim=-1, keepdim=True)
    out = s_dot_t(g, g.x)
    assert (out == exp).all()


def msg_sum(msg, dst, num_nodes):
    out = torch.zeros((num_nodes, msg.shape[1]), dtype=msg.dtype)
    out = out.scatter_add_(dim=0, src=msg, index=dst.view(-1, 1).expand(msg.shape[0], msg.shape[1]))
    return out


def msg_mean(msg, dst, num_nodes):
    deg = torch.zeros((num_nodes,), dtype=torch.float)
    deg = deg.scatter_add_(dim=0, src=torch.ones_like(dst).float(), index=dst)
    deg_inv = deg.pow(-1)
    deg_inv[torch.isinf(deg_inv)] = 0
    out = msg_sum(msg, dst, num_nodes)
    return out * deg_inv.view(-1, 1)


def test_s_ops_e_sum():
    g = build_toy_data()
    edge_attr = g.edge_attr
    row, col = g.edge_index
    src = g.x[col]

    nnode = g.num_nodes
    msg = src + edge_attr
    exp = msg_sum(msg, row, nnode)
    out = s_add_e_sum(g, g.x, g.edge_attr)
    assert (exp == out).all()

    msg = src - edge_attr
    exp = msg_sum(msg, row, nnode)
    out = s_sub_e_sum(g, g.x, g.edge_attr)
    assert (exp == out).all()

    msg = src * edge_attr
    exp = msg_sum(msg, row, nnode)
    out = s_mul_e_sum(g, g.x, g.edge_attr)
    assert (exp == out).all()


def test_s_ops_e_mean():
    g = build_toy_data()
    edge_attr = g.edge_attr
    row, col = g.edge_index
    src = g.x[col]

    nnode = g.num_nodes
    msg = src + edge_attr
    exp = msg_mean(msg, row, nnode)
    out = s_add_e_mean(g, g.x, edge_attr)
    assert (exp == out).all()

    msg = src - edge_attr
    exp = msg_mean(msg, row, nnode)
    out = s_sub_e_mean(g, g.x, edge_attr)
    assert (exp == out).all()

    msg = src * edge_attr
    exp = msg_mean(msg, row, nnode)
    out = s_mul_e_mean(g, g.x, edge_attr)
    assert (exp == out).all()


def test_s_ops_e():
    g = build_toy_data()
    edge_attr = g.edge_attr
    row, col = g.edge_index
    src = g.x[col]

    exp = src + edge_attr
    out = s_add_e(g, g.x, edge_attr)
    assert (exp == out).all()

    exp = src - edge_attr
    out = s_sub_e(g, g.x, edge_attr)
    assert (exp == out).all()

    exp = src * edge_attr
    out = s_mul_e(g, g.x, edge_attr)
    assert (exp == out).all()
