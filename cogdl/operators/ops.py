import torch


def scatter_add(data, dst, num_nodes, dim=0):
    num_edges, num_feats = data.shape
    out = torch.zeros((num_nodes, num_feats), dtype=data.dtype, device=data.device)
    if len(dst.shape) == 1:
        dst = dst.view(-1, 1)
    index = dst.expand(num_edges, num_feats)
    out = out.scatter_add_(dim=dim, index=index, src=data)
    return out


# [src] - (op) - [edge_attr] - (aggr - op)


def op_src_edge(op, src, e_feat):
    if op == "add":
        return src + e_feat
    elif op == "mul":
        return src * e_feat
    elif op == "sub":
        return src - e_feat
    else:
        raise NotImplementedError


def op_aggr(op, msg, dst, num_nodes):
    if op == "sum":
        out = scatter_add(msg, dst, num_nodes)
    elif op == "mean":
        out = scatter_add(msg, dst, num_nodes)
        deg = torch.zeros((num_nodes,), dtype=torch.float, device=msg.device)
        deg = deg.scatter_add_(dim=0, index=dst, src=torch.ones_like(dst).float())
        deg_inv = deg.pow(-1)
        deg_inv[torch.isinf(deg_inv)] = 0
        out = out * deg_inv.view(-1, 1)
    else:
        raise NotImplementedError
    return out


def src_op_e_aggr_coo(op1, op2, n_feat, e_feat, row, col, data=None):
    nnode = n_feat.shape[0]
    if len(e_feat.shape) == 1:
        e_feat = e_feat.view(-1, 1)
    src = n_feat[col]
    msg = op_src_edge(op1, src, e_feat)
    if data is not None:
        msg = msg * data.view(-1, 1)
    out = op_aggr(op2, msg, row, nnode)
    return out


def s_add_e_sum(g, n_feat, e_feat, weight=False):
    row, col = g.edge_index
    if weight:
        data = g.edge_weight
    else:
        data = None
    return src_op_e_aggr_coo("add", "sum", n_feat, e_feat, row, col, data=data)


def s_mul_e_sum(g, n_feat, e_feat, weight=False):
    row, col = g.edge_index
    if weight:
        data = g.edge_weight
    else:
        data = None
    return src_op_e_aggr_coo("mul", "sum", n_feat, e_feat, row, col, data=data)


def s_sub_e_sum(g, n_feat, e_feat, weight=False):
    row, col = g.edge_index
    if weight:
        data = g.edge_weight
    else:
        data = None
    return src_op_e_aggr_coo("sub", "sum", n_feat, e_feat, row, col, data=data)


def s_add_e_mean(g, n_feat, e_feat, weight=False):
    row, col = g.edge_index
    if weight:
        data = g.edge_weight
    else:
        data = None
    return src_op_e_aggr_coo("add", "mean", n_feat, e_feat, row, col, data=data)


def s_mul_e_mean(g, n_feat, e_feat, weight=False):
    row, col = g.edge_index
    if weight:
        data = g.edge_weight
    else:
        data = None
    return src_op_e_aggr_coo("mul", "mean", n_feat, e_feat, row, col, data=data)


def s_sub_e_mean(g, n_feat, e_feat, weight=False):
    row, col = g.edge_index
    if weight:
        data = g.edge_weight
    else:
        data = None
    return src_op_e_aggr_coo("sub", "mean", n_feat, e_feat, row, col, data=data)


# [src] - (op) - [edge]


def s_add_e(g, n_feat, e_feat):
    _, col = g.edge_index
    return n_feat[col] + e_feat


def s_sub_e(g, n_feat, e_feat):
    _, col = g.edge_index
    return n_feat[col] - e_feat


def s_mul_e(g, n_feat, e_feat):
    _, col = g.edge_index
    return n_feat[col] * e_feat


# [src] - (op) - [dst]


def src_op_target_coo(op, g, src, tgt):
    if tgt is None:
        row, col = g.edge_index
        src, tgt = src[col], src[row]
    if op == "add":
        return src + tgt
    elif op == "mul":
        return src * tgt
    elif op == "sub":
        return src - tgt
    elif op == "dot":
        return (src * tgt).sum(1, keepdim=True)
    elif op == "div":
        out = src / tgt
        out[torch.isinf(out)] = 0
        return out
    else:
        raise NotImplementedError


def s_add_t(g, src, dst=None):
    return src_op_target_coo("add", g, src, dst)


def s_mul_t(g, src, dst=None):
    return src_op_target_coo("mul", g, src, dst)


def s_sub_t(g, src, dst=None):
    return src_op_target_coo("sub", g, src, dst)


def s_dot_t(g, src, dst=None):
    return src_op_target_coo("dot", g, src, dst)


def s_div_t(g, src, dst=None):
    return src_op_target_coo("div", g, src, dst)


def message_passing(send_func, msg_func, rec_fun):
    pass
