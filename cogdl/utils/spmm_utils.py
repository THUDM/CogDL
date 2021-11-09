import torch


CONFIGS = {
    "fast_spmm": None,
    "csrmhspmm": None,
    "csr_edge_softmax": None,
    "fused_gat_func": None,
    "spmm_flag": False,
    "mh_spmm_flag": False,
    "fused_gat_flag": False,
}


def check_fused_gat():
    return CONFIGS["fused_gat_func"] is not None


def initialize_spmm():
    if CONFIGS["spmm_flag"]:
        return
    CONFIGS["spmm_flag"] = True
    if torch.cuda.is_available():
        from cogdl.operators.spmm import csrspmm

        CONFIGS["fast_spmm"] = csrspmm
        # if csrspmm is None:
        #     print("Failed to load fast version of SpMM, use torch.scatter_add instead.")


def spmm_scatter(row, col, values, b):
    r"""
    Args:
        (row, col): Tensor, shape=(2, E)
        values : Tensor, shape=(E,)
        b : Tensor, shape=(N, d)
    """
    output = b.index_select(0, col) * values.unsqueeze(-1)
    output = torch.zeros_like(b).scatter_add_(0, row.unsqueeze(-1).expand_as(output), output)
    return output


class SpMM(torch.nn.Module):
    def __init__(self, actnn=False):
        super().__init__()
        initialize_spmm()
        self.actnn = actnn
        self.fast_spmm = CONFIGS["fast_spmm"]

    def forward(self, graph, x):
        if self.fast_spmm is not None and str(x.device) != "cpu":
            if graph.out_norm is not None:
                x = graph.out_norm * x

            row_ptr, col_indices = graph.row_indptr, graph.col_indices
            csr_data = graph.raw_edge_weight
            x = self.fast_spmm(row_ptr.int(), col_indices.int(), x, csr_data, graph.is_symmetric(), actnn=self.actnn)

            if graph.in_norm is not None:
                x = graph.in_norm * x
        else:
            row, col = graph.edge_index
            x = spmm_scatter(row, col, graph.edge_weight, x)
        return x


def spmm(graph, x, actnn=False):
    return SpMM(actnn=actnn)(graph, x)


def initialize_edge_softmax():
    if CONFIGS["mh_spmm_flag"]:
        return
    CONFIGS["mh_spmm_flag"] = True
    if torch.cuda.is_available():
        from cogdl.operators.edge_softmax import csr_edge_softmax
        from cogdl.operators.mhspmm import csrmhspmm

        CONFIGS["csrmhspmm"] = csrmhspmm
        CONFIGS["csr_edge_softmax"] = csr_edge_softmax


def edge_softmax_val(graph, edge_val):
    """
    Args:
        graph: cogdl.Graph
        edge_val: torch.Tensor, shape=(E, 1)
    Returns:
        Softmax values of edge values for nodes
    """
    edge_val_max = edge_val.max().item()
    while edge_val_max > 10:
        edge_val -= edge_val / 2
        edge_val_max = edge_val.max().item()

    with graph.local_graph():
        edge_val = torch.exp(edge_val)
        graph.edge_weight = edge_val
        x = torch.ones(graph.num_nodes, 1).to(edge_val.device)
        node_sum = spmm(graph, x).squeeze()
        row = graph.edge_index[0]
        softmax_values = edge_val / node_sum[row]
        return softmax_values


class MulEdgeSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        initialize_edge_softmax()
        self.csr_edge_softmax = CONFIGS["csr_edge_softmax"]

    def forward(self, graph, edge_val):
        if self.csr_edge_softmax is not None and edge_val.device.type != "cpu":
            if len(edge_val.shape) == 1:
                edge_val = edge_val.view(-1, 1)
                val = self.csr_edge_softmax(graph.row_indptr.int(), edge_val)
                val = val.view(-1)
            else:
                val = self.csr_edge_softmax(graph.row_indptr.int(), edge_val)
            return val
        else:
            val = []
            for i in range(edge_val.shape[1]):
                val.append(edge_softmax_val(graph, edge_val[:, i]))
            return torch.stack(val).t()


def mul_edge_softmax(graph, edge_val):
    return MulEdgeSoftmax()(graph, edge_val)


class MultiHeadSpMM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        initialize_spmm()
        initialize_edge_softmax()
        self.spmm = CONFIGS["fast_spmm"]
        self.csrmhspmm = CONFIGS["csrmhspmm"]

    def forward(self, graph, attention, h):
        nhead = h.shape[1]
        if self.csrmhspmm is not None and h.device.type != "cpu":
            if nhead > 1:
                h_prime = self.csrmhspmm(graph.row_indptr.int(), graph.col_indices.int(), h, attention)
                out = h_prime.view(h_prime.shape[0], -1)
            else:
                edge_weight = attention.view(-1)
                with graph.local_graph():
                    graph.edge_weight = edge_weight
                    out = spmm(graph, h.squeeze(1))
        else:
            with graph.local_graph():
                h_prime = []
                h = h.permute(1, 0, 2).contiguous()
                for i in range(nhead):
                    edge_weight = attention[:, i]
                    graph.edge_weight = edge_weight
                    hidden = h[i]
                    assert not torch.isnan(hidden).any()
                    h_prime.append(spmm(graph, hidden))
            out = torch.cat(h_prime, dim=1)
        return out


def mh_spmm(graph, attention, h):
    return MultiHeadSpMM()(graph, attention, h)


def initialize_fused_gat():
    if CONFIGS["fused_gat_flag"]:
        return
    CONFIGS["fused_gat_flag"] = True
    if torch.cuda.is_available():
        from cogdl.operators.fused_gat import fused_gat_func

        CONFIGS["fused_gat_func"] = fused_gat_func


class FusedGATOp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        initialize_fused_gat()
        self.fused_gat_func = CONFIGS["fused_gat_func"]

    def forward(self, attn_row, attn_col, graph, negative_slope, in_feat):
        return self.fused_gat_func(
            attn_row,
            attn_col,
            graph.row_indptr.int(),
            graph.col_indices.int(),
            graph.row_indptr.int(),
            graph.col_indices.int(),
            negative_slope,
            in_feat,
        )


def fused_gat_op(attn_row, attn_col, graph, negative_slope, in_feat):
    return FusedGATOp()(attn_row, attn_col, graph, negative_slope, in_feat)
