import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .. import BaseModel, register_model
from cogdl.data import DataLoader


def scatter_sum(src, index, dim, dim_size):
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def spare2dense_batch(x, batch=None, fill_value=0):
    batch_size = batch[-1] + 1
    batch_num_nodes = scatter_sum(batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size)
    max_num_nodes = batch_num_nodes.max().item()
    batch_cum_nodes = torch.cat([batch.new_zeros(1), batch_num_nodes.cumsum(dim=0)])

    idx = torch.arange(x.size(0), dtype=torch.long, device=x.device)
    idx = idx - batch_cum_nodes[batch] + batch * max_num_nodes

    new_size = [batch_size * max_num_nodes, x.size(1)]
    out = x.new_full(new_size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes, x.size(1)])
    return out


@register_model("sortpool")
class SortPool(BaseModel):
    r"""Implimentation of sortpooling in paper `"An End-to-End Deep Learning
    Architecture for Graph Classification" <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>__.`

    Parameters
    ----------
    in_feats : int
        Size of each input sample.
    out_feats : int
        Size of each output sample.
    hidden_dim : int
        Dimension of hidden layer embedding.
    num_classes : int
        Number of target classes.
    num_layers : int
        Number of graph neural network layers before pooling.
    k : int, optional
        Number of selected features to sort, default: ``30``.
    out_channel : int
        Number of the first convolution's output channels.
    kernel_size : int
        Size of the first convolution's kernel.
    dropout : float, optional
        Size of dropout, default: ``0.5``.
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--out-channels", type=int, default=32)
        parser.add_argument("--k", type=int, default=30)
        parser.add_argument("--kernel-size", type=int, default=5)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.out_channels,
            args.kernel_size,
            args.k,
            args.dropout
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        random.shuffle(dataset)
        train_size = int(len(dataset) * args.train_ratio)
        test_size = int(len(dataset) * args.test_ratio)
        bs = args.batch_size
        train_loader = DataLoader(dataset[:train_size], batch_size=bs)
        test_loader = DataLoader(dataset[-test_size:], batch_size=bs)
        if args.train_ratio + args.test_ratio < 1:
            valid_loader = DataLoader(dataset[train_size:-test_size], batch_size=bs)
        else:
            valid_loader = test_loader
        return train_loader, valid_loader, test_loader

    def __init__(self, in_feats, hidden_dim, num_classes, num_layers, out_channel, kernel_size, k=30, dropout=0.5):
        super(SortPool, self).__init__()
        self.k = k
        self.dropout = dropout
        self.num_layers = num_layers
        self.gnn_convs = nn.ModuleList()
        self.gnn_convs.append(SAGEConv(in_feats, hidden_dim))
        for _ in range(self.num_layers-1):
            self.gnn_convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.conv1d = nn.Conv1d(hidden_dim, out_channel, kernel_size)
        self.fc1 = nn.Linear(out_channel * (self.k - kernel_size + 1), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        h = batch.x
        for i in range(self.num_layers):
            h = self.gnn_convs[i](h, batch.edge_index)
            h = F.relu(h)

        h, _ = h.sort(dim=-1)
        fill_value = h.min().item() - 1
        batch_h = spare2dense_batch(h, batch.batch, fill_value)
        batch_size, num_nodes, xdim = batch_h.size()

        _, order = batch_h[:, :, -1].sort(dim=-1, descending=True)
        order = order + torch.arange(batch_size, dtype=torch.long, device=order.device).view(-1, 1) * num_nodes

        batch_h = batch_h.view(batch_size * num_nodes, xdim)
        batch_h = batch_h[order].view(batch_size, num_nodes, xdim)

        if num_nodes >= self.k:
            batch_h = batch_h[:, :self.k].contiguous()
        else:
            fill_batch = batch_h.new_full((batch_size, self.k - num_nodes, xdim), fill_value)
            batch_h = torch.cat([batch_h, fill_batch], dim=1)
        batch_h[batch_h == fill_value] = 0
        h = batch_h

        # h = h.view(batch_size, self.k, -1).permute(0, 2, 1) # bn * hidden * k
        h = h.permute(0, 2, 1) # bn * hidden * k
        h = F.relu(self.conv1d(h)).view(batch_size, -1)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)
        if batch.y is not None:
            pred = F.log_softmax(h, dim=-1)
            loss = F.nll_loss(pred, batch.y)
            return h, loss
        return h, None




