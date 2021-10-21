import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spspmm

from .. import BaseModel
from cogdl.layers import GCNLayer
from cogdl.utils import remove_self_loops, coalesce


class GTConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.num_nodes = num_nodes
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        filter = F.softmax(self.weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index, edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value * filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value * filter[i][j]))
            row, col = total_edge_index.detach()
            row, col, value = coalesce(row, col, total_edge_value)
            index = torch.stack([row, col])
            results.append((index, value))
        return results


class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
            self.conv2 = GTConv(in_channels, out_channels, num_nodes)
        else:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)

    def forward(self, A, H_=None):
        if self.first:
            result_A = self.conv1(A)
            result_B = self.conv2(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        H = []
        device = result_A[0][0].device
        for i in range(len(result_A)):
            # a_edge, a_value = result_A[i][0].cpu(), result_A[i][1].cpu()
            # b_edge, b_value = result_B[i][0].cpu(), result_B[i][1].cpu()

            a_edge, a_value = result_A[i][0], result_A[i][1]
            b_edge, b_value = result_B[i][0], result_B[i][1]

            edges, values = spspmm(a_edge, a_value, b_edge, b_value, self.num_nodes, self.num_nodes, self.num_nodes)
            H.append((edges.to(device), values.to(device)))
        return H, W


class GTN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--num-nodes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--num-edge", type=int, default=2)
        parser.add_argument("--num-channels", type=int, default=2)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_edge,
            args.num_channels,
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_nodes,
            args.num_layers,
        )

    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_nodes, num_layers):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=False))
        self.layers = nn.ModuleList(layers)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.gcn = GCNLayer(in_features=self.w_in, out_features=w_out)
        self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            edge, value = H[i]
            edge, value = remove_self_loops(edge, value)
            edge = torch.stack(edge)
            deg_row, deg_col = self.norm(edge.detach(), self.num_nodes, value.detach())
            value = deg_col * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        with torch.no_grad():
            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
            edge_weight = edge_weight.view(-1)
            assert edge_weight.size(0) == edge_index.size(1)
            row, col = edge_index
            deg = torch.zeros((num_nodes,)).to(edge_index.device)
            deg = deg.scatter_add_(dim=0, src=edge_weight, index=row).squeeze()
            deg_inv_sqrt = deg.pow(-1)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def forward(self, graph):
        A = graph.adj
        X = graph.x
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        with graph.local_graph():
            for i in range(self.num_channels):
                if i == 0:
                    edge_index, edge_weight = H[i][0], H[i][1]
                    graph.edge_index = edge_index.detach()
                    graph.edge_weight = edge_weight
                    X_ = self.gcn(graph, X)
                    X_ = F.relu(X_)
                else:
                    edge_index, edge_weight = H[i][0], H[i][1]
                    graph.edge_index = edge_index.detach()
                    graph.edge_weight = edge_weight
                    X_ = torch.cat((X_, F.relu(self.gcn(graph, X))), dim=1)
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        out = self.linear2(X_)
        return out
