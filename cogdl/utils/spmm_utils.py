import torch
from .utils import build_args_from_dict


def spmm_scatter(row, col, values, b):
    r"""
    Args:
        indices : Tensor, shape=(2, E)
        values : Tensor, shape=(E,)
        b : Tensor, shape=(N, )
    """
    output = b.index_select(0, col) * values.unsqueeze(-1)
    output = torch.zeros_like(b).scatter_add_(0, row.unsqueeze(-1).expand_as(output), output)
    return output


CONFIGS = {"fast_spmm": None, "csrmhspmm": None, "csr_edge_softmax": None, "spmm_flag": False, "mh_spmm_flag": False}


def init_operator_configs(args=None):
    if args is not None and args.cpu:
        CONFIGS["fast_spmm"] = None
        CONFIGS["csrmhspmm"] = None
        CONFIGS["csr_edge_softmax"] = None
        return

    if CONFIGS["spmm_flag"] or CONFIGS["mh_spmm_flag"]:
        return
    if args is None:
        args = build_args_from_dict({"fast_spmm": True, "cpu": not torch.cuda.is_available()})
    initialize_spmm(args)
    initialize_edge_softmax(args)


def initialize_spmm(args):
    CONFIGS["spmm_flag"] = True
    if hasattr(args, "fast_spmm") and args.fast_spmm is True and not args.cpu:
        try:
            from cogdl.operators.spmm import csrspmm

            CONFIGS["fast_spmm"] = csrspmm
            # print("Using fast-spmm to speed up training")
        except Exception:
            print("Failed to load fast version of SpMM, use torch.scatter_add instead.")


def initialize_edge_softmax(args):
    CONFIGS["mh_spmm_flag"] = True
    if torch.cuda.is_available() and not args.cpu:
        from cogdl.operators.edge_softmax import csr_edge_softmax
        from cogdl.operators.mhspmm import csrmhspmm

        CONFIGS["csrmhspmm"] = csrmhspmm
        CONFIGS["csr_edge_softmax"] = csr_edge_softmax


def check_fast_spmm():
    return CONFIGS["fast_spmm"] is not None


def check_mh_spmm():
    return CONFIGS["csrmhspmm"] is not None


def check_edge_softmax():
    return CONFIGS["csr_edge_softmax"] is not None


def spmm(graph, x, actnn=False):
    fast_spmm = CONFIGS["fast_spmm"]
    if fast_spmm is not None and str(x.device) != "cpu":
        if graph.out_norm is not None:
            x = graph.out_norm * x

        row_ptr, col_indices = graph.row_indptr, graph.col_indices
        csr_data = graph.raw_edge_weight
        x = fast_spmm(row_ptr.int(), col_indices.int(), x, csr_data, graph.is_symmetric(), actnn=actnn)

        if graph.in_norm is not None:
            x = graph.in_norm * x
    else:
        row, col = graph.edge_index
        x = spmm_scatter(row, col, graph.edge_weight, x)
    return x


def edge_softmax(graph, edge_val):
    """
    Args:
        indices: Tensor, shape=(2, E)
        values: Tensor, shape=(N,)
        shape: tuple(int, int)

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


def mul_edge_softmax(graph, edge_val):
    """
    Returns:
        Softmax values of multi-dimension edge values. shape: [E, H]
    """
    csr_edge_softmax = CONFIGS["csr_edge_softmax"]
    if csr_edge_softmax is not None and edge_val.device.type != "cpu":
        val = csr_edge_softmax(graph.row_indptr.int(), edge_val)
        return val
    else:
        val = []
        for i in range(edge_val.shape[1]):
            val.append(edge_softmax(graph, edge_val[:, i]))
        return torch.stack(val).t()


def mh_spmm(graph, attention, h):
    """
        Multi-head spmm
    Args:
        graph: Graph
        attention: torch.Tensor([E, H])
        h: torch.Tensor([N, d])

    Returns:
        torch.Tensor([N, H, d])
    """
    csrmhspmm = CONFIGS["csrmhspmm"]
    return csrmhspmm(graph.row_indptr.int(), graph.col_indices.int(), h, attention)
