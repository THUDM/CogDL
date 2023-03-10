from cogdl.data import Graph
import scipy.sparse as sp
from cogdl import function as BF

import numpy as np
import scipy

#TODO  BF.sparse.mm()

def getGRBGraph(graph: Graph):
    features = graph.x
    if hasattr(graph, "grb_adj") and graph.grb_adj is not None:
        adj = graph.grb_adj
    else:
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        if type(edge_attr) == BF.dtype_dict('tensor'):
            edge_attr = BF.cpu(edge_attr).numpy()
        if type(edge_index) == BF.dtype_dict('tensor'):
            edge_index = BF.cpu(edge_index).numpy()
        if type(edge_index) == tuple:
            edge_index = [BF.cpu(edge_index[0]).numpy(), BF.cpu(edge_index[1]).numpy()]
        adj = sp.csr_matrix((edge_attr, edge_index), shape=[graph.num_nodes, graph.num_nodes])

    return adj, features


def getGraph(adj, features: BF.dtype_dict('tensor'), labels: BF.dtype_dict('tensor') = None, device="cpu"):
    if type(adj) != BF.dtype_dict('tensor'):
        edge_index, edge_attr = adj2edge(adj, device)
        data = BF.to(Graph(x=features, y=labels, edge_index=edge_index, edge_attr=edge_attr), device)
    else:
        if adj.is_sparse:
            adj_np = sp.csr_matrix(BF.cpu(adj.to_dense().detach()).numpy())
        else:
            adj_np = sp.csr_matrix(BF.cpu(adj.detach()).numpy())
        # print(type(adj_np))
        edge_index, edge_attr = adj2edge(adj_np, device)
        data = BF.to(Graph(x=features, y=labels, edge_index=edge_index, edge_attr=edge_attr, grb_adj=adj), device)
    return data


def updateGraph(graph, adj, features: BF.dtype_dict('tensor')):
    if type(adj) != BF.dtype_dict('tensor'):
        edge_index, edge_attr = adj2edge(adj, BF.device(graph))
        graph.x = features
        graph.edge_index = edge_index
        graph.edge_attr = edge_attr
    else:
        if adj.is_sparse:
            adj_np = sp.csr_matrix(BF.cpu(adj.to_dense().detach()).numpy())
        else:
            adj_np = sp.csr_matrix(BF.cpu(adj.detach()).numpy())
        edge_index, edge_attr = adj2edge(adj_np, BF.device(graph))
        graph.x = features
        graph.edge_index = edge_index
        graph.edge_attr = edge_attr
        graph.grb_adj = adj


def adj2edge(adj: sp.csr.csr_matrix, device="cpu"):
    row, col = adj.nonzero()
    data = adj.data
    row = BF.tensor(row, dtype=BF.dtype_dict("long"))
    col = BF.tensor(col, dtype=BF.dtype_dict("long"))
    edge_index = BF.stack([row, col], dim=0)
    edge_attr = BF.tensor(data, dtype=BF.dtype_dict("float"))
    return BF.to(edge_index,device), BF.to(edge_attr,device)


def adj_to_tensor(adj):
    r"""

    Description
    -----------
    Convert adjacency matrix in scipy sparse format to torch sparse tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    Returns
    -------
    adj_tensor : BF.Tensor
        Adjacency matrix in form of ``N * N`` sparse tensor.

    """
    if type(adj) == BF.dtype_dict('tensor'):
        return adj
    if type(adj) != scipy.sparse.coo.coo_matrix:
        adj = adj.tocoo()
    sparse_row = BF.LongTensor(adj.row).unsqueeze(1)
    sparse_col = BF.LongTensor(adj.col).unsqueeze(1)
    sparse_concat = BF.cat((sparse_row, sparse_col), 1)
    sparse_data = BF.FloatTensor(adj.data)
    adj_tensor = BF.sparse_FloatTensor(sparse_concat.t(), sparse_data, adj.shape)

    return adj_tensor


def adj_preprocess(adj, adj_norm_func=None, mask=None, device="cpu"):
    r"""

    Description
    -----------
    Preprocess the adjacency matrix.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix or a tuple
        Adjacency matrix in form of ``N * N`` sparse matrix.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``None``.
    mask : torch.Tensor, optional
        Mask of nodes in form of ``N * 1`` torch bool tensor. Default: ``None``.
    model_type : str, optional
        Type of model's backend, choose from ["torch", "cogdl", "dgl"]. Default: ``"torch"``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    Returns
    -------
    adj : torch.Tensor or a tuple
        Adjacency matrix in form of ``N * N`` sparse tensor or a tuple.

    """

    if adj_norm_func is not None:
        adj = adj_norm_func(adj)

    if type(adj) is tuple or type(adj) is list:
        if mask is not None:
            adj = [
                BF.to(adj_to_tensor(adj_[mask][:, mask]), device)
                if type(adj_) != BF.dtype_dict('tensor')
                else BF.to(adj_[mask][:, mask], device)
                for adj_ in adj
            ]
        else:
            adj = [BF.to(adj_to_tensor(adj_),device) if type(adj_) != BF.dtype_dict('tensor') else BF.to(adj_.to,device) for adj_ in adj]
    else:
        if type(adj) != BF.dtype_dict('tensor'):
            if mask is not None:
                adj = BF.to(adj_to_tensor(adj[mask][:, mask]), device)
            else:
                adj = BF.to(adj_to_tensor(adj), device)
        else:
            if mask is not None:
                adj = BF.to(adj[mask][:, mask], device)
            else:
                adj = BF.to(adj, device)

    return adj


def feat_preprocess(features, feat_norm=None, device="cpu"):
    r"""

    Description
    -----------
    Preprocess the features.

    Parameters
    ----------
    features : torch.Tensor or numpy.array
        Features in form of torch tensor or numpy array.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    Returns
    -------
    features : torch.Tensor
        Features in form of torch tensor on chosen device.

    """

    def feat_normalize(feat, norm=None):
        if norm == "arctan":
            feat = 2 * np.arctan(feat) / np.pi
        elif norm == "tanh":
            feat = np.tanh(feat)
        else:
            feat = feat

        return feat

    if type(features) != BF.dtype_dict('tensor'):
        features = BF.FloatTensor(features)
    elif features.dtype() != BF.dtype_dict("float"):
        features = features.float()
    if feat_norm is not None:
        features = feat_normalize(features, norm=feat_norm)

    features = BF.to(features, device)

    return features


def label_preprocess(labels, device="cpu"):
    r"""

    Description
    -----------
    Convert labels to torch tensor.

    Parameters
    ----------
    labels : torch.Tensor
        Labels in form of torch tensor.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    Returns
    -------
    labels : torch.Tensor
        Features in form of torch tensor on chosen device.

    """

    if type(labels) != BF.dtype_dict('tensor'):
        labels = BF.LongTensor(labels)
    elif labels.dtype != BF.dtype_dict('long'):
        labels = labels.long()

    labels = BF.to(labels, device)

    return labels


def eval_acc(pred, labels, mask=None):
    r"""

    Description
    -----------
    Accuracy metric for node classification.

    Parameters
    ----------
    pred : torch.Tensor
        Output logits of model in form of ``N * 1``.
    labels : torch.LongTensor
        Labels in form of ``N * 1``.
    mask : torch.Tensor, optional
        Mask of nodes to evaluate in form of ``N * 1`` torch bool tensor. Default: ``None``.

    Returns
    -------
    acc : float
        Node classification accuracy.

    """

    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0
    acc = (BF.argmax(pred, dim=1) == labels).float().sum() / len(pred)

    return acc


def evaluate(model, graph, feat_norm=None, adj_norm_func=None, eval_metric=eval_acc, mask=None, device="cpu"):
    """

    Parameters
    ----------
    model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
    graph: cogdl.data.Graph
        Graph for the model to evaluate.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``None``.
    eval_metric : func of cogdl.utils.grb_utils, optional
        Evaluation metric, like accuracy or F1 score. Default: ``cogdl.utils.grb_utils.eval_acc``.
    mask : torch.tensor, optional
            Mask of target nodes.  Default: ``None``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    Returns
    -------
    score : float
        Score on masked nodes.

    """
    BF.to(model, device)
    model.eval()
    adj, features = getGRBGraph(graph)
    labels = graph.y
    adj = adj_preprocess(adj, adj_norm_func=adj_norm_func, device=device)
    features = feat_preprocess(features, feat_norm=feat_norm, device=device)
    labels = label_preprocess(labels=labels, device=device)
    updateGraph(graph, adj, features)
    logits = model(graph)
    if logits.shape[0] > labels.shape[0]:
        logits = logits[: labels.shape[0]]
    score = eval_metric(logits, labels, mask)

    return score


def GCNAdjNorm(adj, order=-0.5):
    r"""

    Description
    -----------
    Normalization of adjacency matrix proposed in `GCN <https://arxiv.org/abs/1609.02907>`__.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix or torch.FloatTensor
        Adjacency matrix in form of ``N * N`` sparse matrix (or in form of ``N * N`` dense tensor).
    order : float, optional
        Order of degree matrix. Default: ``-0.5``.


    Returns
    -------
    adj : scipy.sparse.csr.csr_matrix
        Normalized adjacency matrix in form of ``N * N`` sparse matrix.

    """
    if sp.issparse(adj):
        adj = sp.eye(adj.shape[0]) + adj
        adj.data[np.where((adj.data > 0) * (adj.data == 1))[0]] = 1
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, order).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        adj = d_mat_inv @ adj @ d_mat_inv
    else:
        rowsum =BF.sparse_mm(adj, BF.ones((adj.shape[0], 1), device=BF.device(adj))) + 1
        d_inv =BF.pow(rowsum, order).flatten()
        d_inv[BF.isinf(d_inv)] = 0.0

        self_loop_idx =BF.stack(
            (BF.arange(adj.shape[0], device=BF.device(adj)),BF.arange(adj.shape[0], device=BF.device(adj)))
        )
        self_loop_val =BF.ones_like(self_loop_idx[0], dtype=adj.dtype)
        indices =BF.cat((self_loop_idx, adj.indices()), dim=1)
        values =BF.cat((self_loop_val, adj.values()))
        values = d_inv[indices[0]] * values * d_inv[indices[1]]
        adj =BF.sparse_FloatTensor(indices, values, adj.shape).coalesce()

    return adj


def SAGEAdjNorm(adj, order=-1):
    r"""

    Description
    -----------
    Normalization of adjacency matrix proposed in `GraphSAGE <https://arxiv.org/abs/1706.02216>`__.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    order : float, optional
        Order of degree matrix. Default: ``-0.5``.


    Returns
    -------
    adj : scipy.sparse.csr.csr_matrix
        Normalized adjacency matrix in form of ``N * N`` sparse matrix.

    """
    if sp.issparse(adj):
        adj = sp.eye(adj.shape[0]) + adj
        for i in range(len(adj.data)):
            if adj.data[i] > 0 and adj.data[i] != 1:
                adj.data[i] = 1
            if adj.data[i] < 0:
                adj.data[i] = 0
        adj.eliminate_zeros()
        adj = sp.coo_matrix(adj)
        if order == 0:
            return adj.tocoo()
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, order).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        adj = d_mat_inv @ adj
    else:
        adj =BF.to(BF.eye(adj.shape[0]), adj) + adj
        rowsum = adj.sum(1)
        d_inv =BF.pow(rowsum, order).flatten()
        d_inv[BF.isinf(d_inv)] = 0.0
        d_mat_inv =BF.diag(d_inv)
        adj = d_mat_inv @ adj

    return adj


def SPARSEAdjNorm(adj, order=-0.5):
    r"""

    Description
    -----------
    Normalization of adjacency matrix proposed in `GCN <https://arxiv.org/abs/1609.02907>`__.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix orBF.FloatTensor
        Adjacency matrix in form of ``N * N`` sparse matrix (or in form of ``N * N`` dense tensor).
    order : float, optional
        Order of degree matrix. Default: ``-0.5``.


    Returns
    -------
    adj : scipy.sparse.csr.csr_matrix
        Normalized adjacency matrix in form of ``N * N`` sparse matrix.

    """
    if sp.issparse(adj):
        adj = sp.eye(adj.shape[0]) + adj
        adj.data[np.where((adj.data > 0) * (adj.data == 1))[0]] = 1
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, order).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        adj = d_mat_inv @ adj @ d_mat_inv
    else:
        rowsum = BF.sparse.mm(adj, BF.ones((adj.shape[0], 1), device=BF.device(adj))) + 1
        d_inv =BF.pow(rowsum, order).flatten()
        d_inv[BF.isinf(d_inv)] = 0.0

        self_loop_idx =BF.stack(
            (BF.arange(adj.shape[0], device=BF.device(adj)),BF.arange(adj.shape[0], device=BF.device(adj)))
        )
        self_loop_val =BF.ones_like(self_loop_idx[0], dtype=adj.dtype)
        indices =BF.cat((self_loop_idx, adj.indices()), dim=1)
        values =BF.cat((self_loop_val, adj.values()))
        values = d_inv[indices[0]] * values * d_inv[indices[1]]
        adj =BF.sparse_FloatTensor(indices, values, adj.shape).coalesce()

    return adj


def RobustGCNAdjNorm(adj):
    r"""

    Description
    -----------
    Normalization of adjacency matrix proposed in `RobustGCN <http://pengcui.thumedialab.com/papers/RGCN.pdf>`__.

    Parameters
    ----------
    adj : tuple of scipy.sparse.csr.csr_matrix
        Tuple of adjacency matrix in form of ``N * N`` sparse matrix.

    Returns
    -------
    adj0 : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    adj1 : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.

    """
    adj0 = GCNAdjNorm(adj, order=-0.5)
    adj1 = GCNAdjNorm(adj, order=-1)

    return adj0, adj1


def feature_normalize(features):
    x_sum =BF.sum(features, dim=1)
    x_rev = x_sum.pow(-1).flatten()
    x_rev[BF.isnan(x_rev)] = 0.0
    x_rev[BF.isinf(x_rev)] = 0.0
    features = features * x_rev.unsqueeze(-1).expand_as(features)

    return features
