import numpy as np
from scipy.linalg import block_diag
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

from .. import BaseModel, register_model
from cogdl.data import DataLoader

def masked_softmax(matrix, mask, dim=-1, memory_efficient=True,
                   mask_fill_value=-1e32):
    '''
    masked_softmax for dgl batch graph
    code snippet contributed by AllenNLP (https://github.com/allenai/allennlp)
    '''
    if mask is None:
        result = torch.nn.functional.softmax(matrix, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < matrix.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = torch.nn.functional.softmax(matrix * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_matrix = matrix.masked_fill((1 - mask).byte(),
                                               mask_fill_value)
            result = torch.nn.functional.softmax(masked_matrix, dim=dim)
    return result

class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, adj, anext, s_l):
        entropy = (torch.distributions.Categorical(
            probs=s_l).entropy()).sum(-1).mean(-1)
        assert not torch.isnan(entropy)
        return entropy


class LinkPredLoss(nn.Module):
    def forward(self, adj, anext, s_l):
        link_pred_loss = (
            adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        return link_pred_loss.mean()

class GraphSAGE(nn.Module):
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
        # TODO: shape = [a, 0, b] ?
        # if self.use_bn and h.shape[1] > 0:
        #     self.bn = nn.BatchNorm1d(h.shape[1]).to(device)
        #     h = self.bn(h)
        return h


class BatchedDiffPoolLayer(nn.Module):
    def __init__(self, in_feats, out_feats, assign_dim, batch_size, dropout=0.5, link_pred_loss=True):
        super(BatchedDiffPoolLayer, self).__init__()
        self.assign_dim = assign_dim
        self.dropout = dropout
        self.use_link_pred = link_pred_loss
        self.batch_size = batch_size
        self.embd_gnn = SAGEConv(in_feats, out_feats, normalize=True, concat=True)
        self.pool_gnn = SAGEConv(in_feats, assign_dim, normalize=True, concat=True)
        self.loss_dict = dict()
        self.loss_type = EntropyLoss()

    def forward(self, x, edge_index, batch, edge_weight=None):
        embed = self.embd_gnn(x, edge_index)
        pooled = self.pool_gnn(x, edge_index)
        device = x.device
        masked_tensor = []
        value_set, value_counts = torch.unique(batch, return_counts=True)
        batch_size = len(value_set)
        # batch_size = self.batch_size
        for i in value_counts:
            masked = torch.ones((i, int(pooled.size()[1]/batch_size)))
            masked_tensor.append(masked)
        masked = torch.FloatTensor(block_diag(*masked_tensor)).to(device)

        # result = torch.nn.functional.softmax(masked * pooled, dim=-1)
        # result = result * masked
        # result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
        result = masked_softmax(pooled, masked, memory_efficient=False)


        h = torch.matmul(result.t(), embed)
        if not edge_weight:
            edge_weight = torch.ones(edge_index.shape[1]).cuda()
        adj = torch.sparse_coo_tensor(edge_index, edge_weight)
        adj_new = torch.sparse.mm(adj, result)
        adj_new = torch.mm(result.t(), adj_new)

        if self.use_link_pred:
            adj_loss = torch.norm((adj.to_dense() - torch.mm(result, result.t()))) / np.power((len(batch)), 2)
            self.loss_dict["adj_loss"] = adj_loss
        self.loss_dict["entropy_loss"] = self.loss_type(adj, adj_new, masked)
        return adj_new, h


class BatchedDiffPool(nn.Module):
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
        next_l = self.assign_trans(x, adj)

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
    # for i in range(num_graphs):
    #     start = i * node_per_pool_graph
    #     end = (i + 1) * node_per_pool_graph
    #     adj_list.append(batch_adj[start:end, start:end])
    #     feat_list.append(batch_feat[start:end, :])
    adj_list = list(map(lambda x: torch.unsqueeze(x, 0), adj_list))
    feat_list = list(map(lambda x: torch.unsqueeze(x, 0), feat_list))
    adj = torch.cat(adj_list, dim=0)
    feat = torch.cat(feat_list, dim=0)
    return adj, feat


@register_model("diffpool")
class DiffPool(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--num-pooling-layers", type=int, default=2)
        parser.add_argument("--no-link-pred", dest="no_link_pred", action="store_false")
        parser.add_argument("--pooling-ratio", type=float, default=0.15)
        parser.add_argument("--embedding-dim", type=int, default=64)
        parser.add_argument("--hidden-dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)


    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_dim,
            args.embedding_dim,
            args.num_classes,
            args.num_layers,
            args.num_pooling_layers,
            args.dropout,
            args.no_link_pred,
            int(args.max_graph_size * args.pooling_ratio) * args.batch_size,
            args.pooling_ratio,
            args.batch_size
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        train_index = int(len(dataset) * args.train_ratio)
        test_index = int(len(dataset) * args.test_ratio)
        train_data = dataset[:train_index]
        valid_data = dataset[train_index:-test_index]
        test_data = dataset[-test_index:]
        return DataLoader(train_data, args.batch_size, drop_last=True), DataLoader(valid_data, args.batch_size, drop_last=True),\
                DataLoader(test_data, 1)

    def __init__(self, in_feats, hidden_dim, embed_dim, num_classes, num_layers, num_pool_layers, dropout, no_link_pred,
                 assign_dim, pooling_ratio, batch_size, concat=False, use_bn=False):
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

    def forward(self, x, edge_index, batch=None, label=None):
        readouts_all = []

        init_emb = self.before_pooling(x, edge_index)
        adj, h = self.init_diffpool(init_emb, edge_index, batch)
        value_set, value_counts = torch.unique(batch, return_counts=True)
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
        if label is not None:
            return pred, self.loss(pred, label)
        return pred, None

    def loss(self, prediction, label):
        criterion = nn.CrossEntropyLoss()
        # TODO: loss
        loss_n = criterion(F.softmax(prediction), label)
        for layer in self.diffpool_layers:
            loss_n += layer.get_loss()
        return loss_n

