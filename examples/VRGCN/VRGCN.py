import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from cogdl.utils import spmm
from cogdl.data import Graph


class History(torch.nn.Module):
    r"""A historical embedding storage module."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb = torch.empty(num_embeddings, embedding_dim, device="cpu", pin_memory=True)
        self._device = torch.device("cpu")
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.fill_(0)

    def _apply(self, fn):
        self._device = fn(torch.zeros(1)).device
        return self

    @torch.no_grad()
    def pull(self, n_id=None):
        out = self.emb
        if n_id is not None:
            assert n_id.device == self.emb.device
            out = out.index_select(0, n_id)
        return out.to(device=self._device)

    @torch.no_grad()
    def push(self, x, n_id=None):
        assert n_id.device == self.emb.device
        self.emb[n_id] = x.to(self.emb.device)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class VRGCN(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        device=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.residual = residual
        self.num_layers = num_layers

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for i in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.norms = nn.ModuleList()
        for i in range(num_layers):
            norm = nn.LayerNorm(hidden_channels)
            self.norms.append(norm)

        self.histories = torch.nn.ModuleList(
            [
                History(num_nodes, hidden_channels) if i != 0 else History(num_nodes, in_channels)
                for i in range(num_layers)
            ]
        )

        self._device = device

    def reset_parameters(self):
        for history in self.histories:
            history.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, sample_ids_adjs, full_ids_adjs) -> Tensor:
        sample_ids, sample_adjs = sample_ids_adjs
        full_ids, full_adjs = full_ids_adjs

        """VR-GCN"""
        x = x[sample_ids[0]].to(self._device)
        x_list = []
        for i in range(self.num_layers):
            sample_adj, cur_id, target_id = sample_adjs[i], sample_ids[i], sample_ids[i + 1]
            full_id, full_adj = full_ids[i], full_adjs[i]
            full_adj = full_adj.to(x.device)
            sample_adj = sample_adj.to(x.device)

            x = x - self.histories[i].pull(cur_id).detach()
            h = self.histories[i].pull(full_id)

            x = spmm(sample_adj, x)[: target_id.shape[0]] + spmm(full_adj, h)[: target_id.shape[0]].detach()
            x = self.lins[i](x)

            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = x.relu_()
                x_list.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        """history embedding update"""
        for i in range(1, self.num_layers):
            self.histories[i].push(x_list[i - 1].detach(), sample_ids[i])
        return x.log_softmax(dim=-1)

    def initialize_history(self, x, test_loader):
        _, xs = self.inference_batch(x, test_loader)
        for i in range(self.num_layers):
            self.histories[i].push(xs[i].detach(), torch.arange(0, self.histories[i].num_embeddings))

    @torch.no_grad()
    def inference(self, x, adj):
        x = x.to(self._device)
        origin_device = adj.device
        adj = adj.to(self._device)
        xs = [x]
        for i in range(self.num_layers):
            x = spmm(adj, x)
            x = self.lins[i](x)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = x.relu_()
            xs.append(x)
        adj = adj.to(origin_device)
        return x, xs

    @torch.no_grad()
    def inference_batch(self, x, test_loader):
        device = self._device
        xs = [x]
        for i in range(self.num_layers):
            tmp_x = []
            for target_id, full_id, full_adj in test_loader:
                full_adj = full_adj.to(device)
                agg_x = spmm(full_adj, x[full_id].to(device))[: target_id.shape[0]]
                agg_x = self.lins[i](agg_x)

                if i != self.num_layers - 1:
                    agg_x = self.norms[i](agg_x)
                    agg_x = agg_x.relu_()
                tmp_x.append(agg_x.cpu())
            x = torch.cat(tmp_x, dim=0)
            xs.append(x)
        return x, xs
