import numpy as np
import random

from scipy.linalg import block_diag
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_remaining_self_loops

from .. import BaseModel, register_model
from cogdl.data import DataLoader


class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, adj, anext, s_l):
        # entropy.mean(-1).mean(-1): 1/n in node and batch
        # entropy = (torch.distributions.Categorical(
            # probs=s_l).entropy()).sum(-1).mean(-1)
        entropy = (torch.distributions.Categorical(
            probs=s_l).entropy()).mean()
        assert not torch.isnan(entropy)
        return entropy


class LinkPredLoss(nn.Module):
    def forward(self, adj, anext, s_l):
        link_pred_loss = (
            adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        return link_pred_loss.mean()


class GraphSAGE(nn.Module):
    r"""GraphSAGE from `"Inductive Representation Learning on Large Graphs" <https://arxiv.org/pdf/1706.02216.pdf>`__.

    ..math::
        h^{i+1}_{\mathcal{N}(v)}=AGGREGATE_{k}(h_{u}^{k})
        h^{k+1}_{v} = \sigma(\mathbf{W}^{k}·CONCAT(h_{v}^{k}, h_{\mathcal{N}(v)}))

    Args:
        in_feats (int) : Size of each input sample.
        hidden_dim (int) : Size of hidden layer dimension.
        out_feats (int) : Size of each output sample.
        num_layers (int) : Number of GraphSAGE Layers.
        dropout (float, optional) : Size of dropout, default: ``0.5``.
        normalize (bool, optional) : Normalze features after each layer if True, default: ``True``.
    """
    def __init__(self, in_feats, hidden_dim, out_feats, num_layers, dropout=0.5, normalize=False, concat=False, use_bn=False):
        super(GraphSAGE, self).__init__()
        self.convlist = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        if num_layers == 1:
            self.convlist.append(SAGEConv(in_feats, out_feats, normalize, concat))
        else:
            self.convlist.append(SAGEConv(in_feats, hidden_dim, normalize, concat))
            if use_bn:
                self.bn_list.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.convlist.append(SAGEConv(hidden_dim, hidden_dim, normalize, concat))
                if use_bn:
                    self.bn_list.append(nn.BatchNorm1d(hidden_dim))
            self.convlist.append(SAGEConv(hidden_dim, out_feats, normalize, concat))

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i in range(self.num_layers-1):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.convlist[i](h, edge_index, edge_weight)
            if self.use_bn:
                h = self.bn_list[i](h)
        return self.convlist[self.num_layers-1](h, edge_index, edge_weight)


class BatchedGraphSAGE(nn.Module):
    r"""GraphSAGE with mini-batch

    Args:
        in_feats (int) : Size of each input sample.
        out_feats (int) : Size of each output sample.
        use_bn (bool) : Apply batch normalization if True, default: ``True``.
        self_loop (bool) : Add self loop if True, default: ``True``.
    """
    def __init__(self, in_feats, out_feats, use_bn=True, self_loop=True):
        super(BatchedGraphSAGE, self).__init__()
        self.self_loop = self_loop
        self.use_bn = use_bn
        self.weight = nn.Linear(in_feats, out_feats, bias=True)

        nn.init.xavier_uniform_(self.weight.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj):
        device = x.device
        if self.self_loop:
            adj = adj + torch.eye(x.shape[1]).to(device)
        adj = adj / adj.sum(dim=1, keepdim=True)
        h = torch.matmul(adj, x)
        h = self.weight(h)
        h = F.normalize(h, dim=2, p=2)
        h = F.relu(h)
        # TODO: shape = [a, 0, b]
        # if self.use_bn and h.shape[1] > 0:
        #     self.bn = nn.BatchNorm1d(h.shape[1]).to(device)
        #     h = self.bn(h)
        return h


class BatchedDiffPoolLayer(nn.Module):
    r"""DIFFPOOL from paper `"Hierarchical Graph Representation Learning
    with Differentiable Pooling" <https://arxiv.org/pdf/1806.08804.pdf>`__.

    .. math::
        X^{(l+1)} = S^{l)}^T Z^{(l)}
        A^{(l+1)} = S^{(l)}^T A^{(l)} S^{(l)}
        Z^{(l)} = GNN_{l, embed}(A^{(l)}, X^{(l)})
        S^{(l)} = softmax(GNN_{l,pool}(A^{(l)}, X^{(l)}))

    Parameters
    ----------
    in_feats : int
        Size of each input sample.
    out_feats : int
        Size of each output sample.
    assign_dim : int
        Size of next adjacency matrix.
    batch_size : int
        Size of each mini-batch.
    dropout : float, optional
        Size of dropout, default: ``0.5``.
    link_pred_loss : bool, optional
        Use link prediction loss if True, default: ``True``.
    """
    def __init__(self, in_feats, out_feats, assign_dim, batch_size, dropout=0.5, link_pred_loss=True, entropy_loss=True):
        super(BatchedDiffPoolLayer, self).__init__()
        self.assign_dim = assign_dim
        self.dropout = dropout
        self.use_link_pred = link_pred_loss
        self.batch_size = batch_size
        self.embd_gnn = SAGEConv(in_feats, out_feats, normalize=False)
        self.pool_gnn = SAGEConv(in_feats, assign_dim, normalize=False)

        self.loss_dict = dict()


    def forward(self, x, edge_index, batch, edge_weight=None):
        embed = self.embd_gnn(x, edge_index)
        pooled = F.softmax(self.pool_gnn(x, edge_index), dim=-1)
        device = x.device
        masked_tensor = []
        value_set, value_counts = torch.unique(batch, return_counts=True)
        batch_size = len(value_set)
        for i in value_counts:
            masked = torch.ones((i, int(pooled.size()[1]/batch_size)))
            masked_tensor.append(masked)
        masked = torch.FloatTensor(block_diag(*masked_tensor)).to(device)

        result = torch.nn.functional.softmax(masked * pooled, dim=-1)
        result = result * masked
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
        # result = masked_softmax(pooled, masked, memory_efficient=False)

        h = torch.matmul(result.t(), embed)
        if not edge_weight:
            edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        adj = torch.sparse_coo_tensor(edge_index, edge_weight)
        adj_new = torch.sparse.mm(adj, result)
        adj_new = torch.mm(result.t(), adj_new)

        if self.use_link_pred:
            adj_loss = torch.norm((adj.to_dense() - torch.mm(result, result.t()))) / np.power((len(batch)), 2)
            self.loss_dict["adj_loss"] = adj_loss
        entropy_loss = (torch.distributions.Categorical(probs=pooled).entropy()).mean()
        assert not torch.isnan(entropy_loss)
        self.loss_dict["entropy_loss"] = entropy_loss
        return adj_new, h

    def get_loss(self):
        loss_n = 0
        for _, value in self.loss_dict.items():
            loss_n += value
        return loss_n


class BatchedDiffPool(nn.Module):
    r"""DIFFPOOL layer with batch forward

    Parameters
    ----------
    in_feats : int
        Size of each input sample.
    next_size : int
        Size of next adjacency matrix.
    emb_size : int
        Dimension of next node feature matrix.
    use_bn : bool, optional
        Apply batch normalization if True, default: ``True``.
    self_loop : bool, optional
        Add self loop if True, default: ``True``.
    use_link_loss : bool, optional
        Use link prediction loss if True, default: ``True``.
    use_entropy : bool, optioinal
        Use entropy prediction loss if True, default: ``True``.
    """
    def __init__(self, in_feats, next_size, emb_size, use_bn=True, self_loop=True, use_link_loss=False, use_entropy=True):
        super(BatchedDiffPool, self).__init__()
        self.use_link_loss = use_link_loss
        self.use_bn = use_bn
        self.feat_trans = BatchedGraphSAGE(in_feats, emb_size)
        self.assign_trans = BatchedGraphSAGE(in_feats, next_size)

        self.link_loss = LinkPredLoss()
        self.entropy = EntropyLoss()

        self.loss_module = nn.ModuleList()
        if use_link_loss:
            self.loss_module.append(LinkPredLoss())
        if use_entropy:
            self.loss_module.append(EntropyLoss())
        self.loss = {}

    def forward(self, x, adj):
        h = self.feat_trans(x, adj)
        next_l = F.softmax(self.assign_trans(x, adj), dim=-1)

        h = torch.matmul(next_l.transpose(-1, -2), h)
        next = torch.matmul(next_l.transpose(-1, -2), torch.matmul(adj, next_l))

        for layer in self.loss_module:
            self.loss[str(type(layer).__name__)] = layer(adj, next, next_l)

        return h, next

    def get_loss(self):
        value = 0
        for _, v in self.loss.items():
            value += v
        return value


def toBatchedGraph(batch_adj, batch_feat, node_per_pool_graph):
    adj_list = [batch_adj[i:i+node_per_pool_graph, i:i+node_per_pool_graph]
                for i in range(0, batch_adj.size()[0], node_per_pool_graph)]
    feat_list = [batch_feat[i:i+node_per_pool_graph, :] for i in range(0, batch_adj.size()[0], node_per_pool_graph)]
    adj_list = list(map(lambda x: torch.unsqueeze(x, 0), adj_list))
    feat_list = list(map(lambda x: torch.unsqueeze(x, 0), feat_list))
    adj = torch.cat(adj_list, dim=0)
    feat = torch.cat(feat_list, dim=0)
    return adj, feat


@register_model("diffpool")
class DiffPool(BaseModel):
    r"""DIFFPOOL from paper `Hierarchical Graph Representation Learning
    with Differentiable Pooling <https://arxiv.org/pdf/1806.08804.pdf>`__.

    Parameters
    ----------
    in_feats : int
        Size of each input sample.
    hidden_dim : int
        Size of hidden layer dimension of GNN.
    embed_dim : int
        Size of embeded node feature, output size of GNN.
    num_classes : int
        Number of target classes.
    num_layers : int
        Number of GNN layers.
    num_pool_layers : int
        Number of pooling.
    assign_dim : int
        Embedding size after the first pooling.
    pooling_ratio : float
        Size of each poolling ratio.
    batch_size : int
        Size of each mini-batch.
    dropout : float, optional
        Size of dropout, default: `0.5`.
    no_link_pred : bool, optional
        If True, use link prediction loss, default: `True`.
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--num-pooling-layers", type=int, default=1)
        parser.add_argument("--no-link-pred", dest="no_link_pred", action="store_true")
        parser.add_argument("--pooling-ratio", type=float, default=0.15)
        parser.add_argument("--embedding-dim", type=int, default=64)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=0.001)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.embedding_dim,
            args.num_classes,
            args.num_layers,
            args.num_pooling_layers,
            int(args.max_graph_size * args.pooling_ratio) * args.batch_size,
            args.pooling_ratio,
            args.batch_size,
            args.dropout,
            args.no_link_pred
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        random.shuffle(dataset)
        train_size = int(len(dataset) * args.train_ratio)
        test_size = int(len(dataset) * args.test_ratio)
        bs = args.batch_size
        train_loader = DataLoader(dataset[:train_size], batch_size=bs, drop_last=True)
        test_loader = DataLoader(dataset[-test_size:], batch_size=bs, drop_last=True)
        if args.train_ratio + args.test_ratio < 1:
            valid_loader = DataLoader(dataset[train_size:-test_size], batch_size=bs, drop_last=True)
        else:
            valid_loader = test_loader
        return train_loader, valid_loader, test_loader

    def __init__(self, in_feats, hidden_dim, embed_dim, num_classes, num_layers, num_pool_layers,  assign_dim,
                 pooling_ratio, batch_size, dropout=0.5, no_link_pred=True, concat=False, use_bn=False):
        super(DiffPool, self).__init__()
        self.assign_dim = assign_dim
        self.assign_dim_list = [assign_dim]
        self.use_bn = use_bn
        self.dropout = dropout
        self.use_link_loss = not no_link_pred
        # assert num_layers > 3, "layers > 3"
        self.diffpool_layers = nn.ModuleList()
        self.before_pooling = GraphSAGE(in_feats, hidden_dim, embed_dim,
                                        num_layers=num_layers, dropout=dropout, use_bn=self.use_bn)
        self.init_diffpool = BatchedDiffPoolLayer(embed_dim, hidden_dim, assign_dim, batch_size, dropout, self.use_link_loss)

        pooled_emb_dim = embed_dim
        self.after_pool = nn.ModuleList()
        after_per_pool = nn.ModuleList()
        for _ in range(num_layers-1):
            after_per_pool.append(BatchedGraphSAGE(hidden_dim, hidden_dim))
        after_per_pool.append(BatchedGraphSAGE(hidden_dim, pooled_emb_dim))
        self.after_pool.append(after_per_pool)

        for _ in range(num_pool_layers-1):
            self.assign_dim = int(self.assign_dim//batch_size * pooling_ratio) * batch_size
            self.diffpool_layers.append(BatchedDiffPool(
                pooled_emb_dim, self.assign_dim, hidden_dim, use_bn=self.use_bn, use_link_loss=self.use_link_loss
            ))

            for _ in range(num_layers - 1):
                after_per_pool.append(BatchedGraphSAGE(hidden_dim, hidden_dim))
            after_per_pool.append(BatchedGraphSAGE(hidden_dim, pooled_emb_dim))
            self.after_pool.append(after_per_pool)

            self.assign_dim_list.append(self.assign_dim)

        if concat:
            out_dim = pooled_emb_dim * (num_pool_layers+1)
        else:
            out_dim = pooled_emb_dim
        self.fc = nn.Linear(out_dim, num_classes)

    def reset_parameters(self):
        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight.data, gain=nn.init.calculate_gain('relu'))
                if i.bias is not None:
                    nn.init.constant_(i.bias.data, 0.)

    def after_pooling_forward(self, gnn_layers, adj, x, concat=False):
        readouts = []
        h = x
        for layer in gnn_layers:
            h = layer(h, adj)
            readouts.append(h)
        readout = torch.cat(readouts, dim=1)
        return h

    def forward(self, batch):
        readouts_all = []

        init_emb = self.before_pooling(batch.x, batch.edge_index)
        adj, h = self.init_diffpool(init_emb, batch.edge_index, batch.batch)
        value_set, value_counts = torch.unique(batch.batch, return_counts=True)
        batch_size = len(value_set)
        adj, h = toBatchedGraph(adj, h, adj.size(0)//batch_size)
        h = self.after_pooling_forward(self.after_pool[0], adj, h)
        readout = torch.sum(h, dim=1)
        readouts_all.append(readout)

        for i, diff_layer in enumerate(self.diffpool_layers):
            h, adj = diff_layer(h, adj)
            h = self.after_pooling_forward(self.after_pool[i+1], adj, h)
            readout = torch.sum(h, dim=1)
            readouts_all.append(readout)
        pred = self.fc(readout)
        if batch.y is not None:
            return pred, self.loss(pred, batch.y)
        return pred, None

    def loss(self, prediction, label):
        criterion = nn.CrossEntropyLoss()
        loss_n = criterion(prediction, label)
        loss_n += self.init_diffpool.get_loss()
        for layer in self.diffpool_layers:
            loss_n += layer.get_loss()
        return loss_n

