import random

import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
    dropout_adj,
    is_undirected,
    accuracy,
    negative_sampling,
    batched_negative_sampling,
    to_undirected,
)
import torch_geometric.nn.inits as tgi

from cogdl.trainers.supergat_trainer import SuperGATTrainer
from .. import BaseModel, register_model

from typing import List


# borrowed from https://github.com/dongkwan-kim/SuperGAT
def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SuperGATLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
        is_super_gat=True,
        attention_type="basic",
        super_gat_criterion=None,
        neg_sample_ratio=0.0,
        edge_sample_ratio=1.0,
        pretraining_noise_ratio=0.0,
        use_pretraining=False,
        to_undirected_at_neg=False,
        scaling_factor=None,
        cache_label=False,
        cache_attention=False,
        **kwargs,
    ):
        super(SuperGATLayer, self).__init__(aggr="add", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.is_super_gat = is_super_gat
        self.attention_type = attention_type
        self.super_gat_criterion = super_gat_criterion
        self.neg_sample_ratio = neg_sample_ratio
        self.edge_sample_ratio = edge_sample_ratio
        self.pretraining_noise_ratio = pretraining_noise_ratio
        self.pretraining = None if not use_pretraining else True
        self.to_undirected_at_neg = to_undirected_at_neg
        self.cache_label = cache_label
        self.cache_attention = cache_attention

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))

        if self.is_super_gat:

            if self.attention_type == "gat_originated":  # GO
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))

            elif self.attention_type == "dot_product":  # DP
                pass

            elif self.attention_type == "scaled_dot_product":  # SD
                self.scaling_factor = scaling_factor or np.sqrt(self.out_channels)

            elif self.attention_type.endswith("mask_only"):  # MX
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))

            else:
                raise ValueError

        else:
            if self.attention_type.endswith("gat_originated") or self.attention_type == "basic":
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))

            elif self.attention_type.endswith("dot_product"):
                pass

            else:
                raise ValueError

        self.cache = {
            "num_updated": 0,
            "att": None,  # Use only when self.cache_attention == True for task_type == "Attention_Dist"
            "att_with_negatives": None,  # Use as X for supervision.
            "att_label": None,  # Use as Y for supervision.
        }

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        tgi.glorot(self.weight)
        tgi.zeros(self.bias)
        for name, param in self.named_parameters():
            if name.startswith("att_scaling"):
                tgi.ones(param)
            elif name.startswith("att_bias"):
                tgi.zeros(param)
            elif name.startswith("att_mh"):
                tgi.glorot(param)

    def forward(self, x, edge_index, size=None, batch=None, neg_edge_index=None, attention_edge_index=None):
        """
        :param x: [N, F]
        :param edge_index: [2, E]
        :param size:
        :param batch: None or [B]
        :param neg_edge_index: When using explicitly given negative edges.
        :param attention_edge_index: [2, E'], Use for link prediction
        :return:
        """
        if isinstance(edge_index, tuple):
            edge_index = torch.stack(edge_index)
        if self.pretraining and self.pretraining_noise_ratio > 0.0:
            edge_index, _ = dropout_adj(
                edge_index,
                p=self.pretraining_noise_ratio,
                force_undirected=is_undirected(edge_index),
                num_nodes=x.size(0),
                training=self.training,
            )

        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # [N, F0] * [F0, heads * F] = [N, heads * F]
        x = torch.matmul(x, self.weight)
        x = x.view(-1, self.heads, self.out_channels)

        propagated = self.propagate(edge_index, size=size, x=x)

        if (self.is_super_gat and self.training) or (attention_edge_index is not None) or (neg_edge_index is not None):

            device = next(self.parameters()).device
            num_pos_samples = int(self.edge_sample_ratio * edge_index.size(1))
            num_neg_samples = int(self.neg_sample_ratio * self.edge_sample_ratio * edge_index.size(1))

            if attention_edge_index is not None:
                neg_edge_index = None

            elif neg_edge_index is not None:
                pass

            elif batch is None:
                if self.to_undirected_at_neg:
                    edge_index_for_ns = to_undirected(edge_index, num_nodes=x.size(0))
                else:
                    edge_index_for_ns = edge_index
                neg_edge_index = negative_sampling(
                    edge_index=edge_index_for_ns,
                    num_nodes=x.size(0),
                    num_neg_samples=num_neg_samples,
                )
            else:
                neg_edge_index = batched_negative_sampling(
                    edge_index=edge_index,
                    batch=batch,
                    num_neg_samples=num_neg_samples,
                )

            if self.edge_sample_ratio < 1.0:
                pos_indices = random.sample(range(edge_index.size(1)), num_pos_samples)
                pos_indices = torch.tensor(pos_indices).long().to(device)
                pos_edge_index = edge_index[:, pos_indices]
            else:
                pos_edge_index = edge_index

            att_with_negatives = self._get_attention_with_negatives(
                x=x,
                edge_index=pos_edge_index,
                neg_edge_index=neg_edge_index,
                total_edge_index=attention_edge_index,
            )  # [E + neg_E, heads]

            # Labels
            if self.training and (self.cache["att_label"] is None or not self.cache_label):
                att_label = torch.zeros(att_with_negatives.size(0)).float().to(device)
                att_label[: pos_edge_index.size(1)] = 1.0
            elif self.training and self.cache["att_label"] is not None:
                att_label = self.cache["att_label"]
            else:
                att_label = None
            self._update_cache("att_label", att_label)
            self._update_cache("att_with_negatives", att_with_negatives)

        return propagated

    def message(self, edge_index_i, x_i, x_j, size_i):
        """
        :param edge_index_i: [E]
        :param x_i: [E, heads * F]
        :param x_j: [E, heads * F]
        :param size_i: N
        :return: [E, heads, F]
        """
        x_j = x_j.view(-1, self.heads, self.out_channels)  # [E, heads, F]
        if x_i is not None:
            x_i = x_i.view(-1, self.heads, self.out_channels)  # [E, heads, F]

        # Compute attention coefficients. [E, heads]
        alpha = self._get_attention(edge_index_i, x_i, x_j, size_i)
        if self.cache_attention:
            self._update_cache("att", alpha)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # [E, heads, F] * [E, heads, 1] = [E, heads, F]
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        """
        :param aggr_out: [N, heads, F]
        :return: [N, heads * F]
        """
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def _get_attention(
        self, edge_index_i, x_i, x_j, size_i, normalize=True, with_negatives=False, **kwargs
    ) -> torch.Tensor:
        """
        :param edge_index_i: [E]
        :param x_i: [E, heads, F]
        :param x_j: [E, heads, F]
        :param size_i: N
        :return: [E, heads]
        """

        # Compute attention coefficients.
        if self.attention_type == "basic" or self.attention_type.endswith("gat_originated"):
            # [E, heads, 2F] * [1, heads, 2F] -> [E, heads]
            alpha = torch.einsum("ehf,xhf->eh", torch.cat([x_i, x_j], dim=-1), self.att_mh_1)

        elif self.attention_type == "scaled_dot_product":
            alpha = torch.einsum("ehf,ehf->eh", x_i, x_j) / self.scaling_factor

        elif self.attention_type == "dot_product":
            # [E, heads, F] * [E, heads, F] -> [E, heads]
            alpha = torch.einsum("ehf,ehf->eh", x_i, x_j)

        elif "mask" in self.attention_type:

            # [E, heads, F] * [E, heads, F] -> [E, heads]
            logits = torch.einsum("ehf,ehf->eh", x_i, x_j)

            if self.attention_type.endswith("scaling"):
                logits = logits / self.att_scaling

            if with_negatives:
                return logits

            # [E, heads, 2F] * [1, heads, 2F] -> [E, heads]
            alpha = torch.einsum("ehf,xhf->eh", torch.cat([x_i, x_j], dim=-1), self.att_mh_1)
            alpha = torch.einsum("eh,eh->eh", alpha, torch.sigmoid(logits))

        else:
            raise ValueError

        if normalize:
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        return alpha

    def _get_attention_with_negatives(self, x, edge_index, neg_edge_index, total_edge_index=None):
        """
        :param x: [N, heads * F]
        :param edge_index: [2, E]
        :param neg_edge_index: [2, neg_E]
        :param total_edge_index: [2, E + neg_E], if total_edge_index is given, use it.
        :return: [E + neg_E, heads]
        """

        if neg_edge_index is not None and neg_edge_index.size(1) <= 0:
            neg_edge_index = torch.zeros((2, 0, self.heads))

        if total_edge_index is None:
            total_edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]

        total_edge_index_j, total_edge_index_i = total_edge_index  # [E + neg_E]
        x_i = torch.index_select(x, 0, total_edge_index_i)  # [E + neg_E, heads * F]
        x_j = torch.index_select(x, 0, total_edge_index_j)  # [E + neg_E, heads * F]
        size_i = x.size(0)  # N

        x_j = x_j.view(-1, self.heads, self.out_channels)  # [E + neg_E, heads, F]
        if x_i is not None:
            x_i = x_i.view(-1, self.heads, self.out_channels)  # [E + neg_E, heads, F]

        alpha = self._get_attention(total_edge_index_i, x_i, x_j, size_i, normalize=False, with_negatives=True)
        return alpha

    def __repr__(self):
        return "{}({}, {}, heads={}, concat={}, att_type={}, nsr={}, pnr={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads,
            self.concat,
            self.attention_type,
            self.neg_sample_ratio,
            self.pretraining_noise_ratio,
        )

    def _update_cache(self, key, val):
        self.cache[key] = val
        self.cache["num_updated"] += 1

    def get_attention_dist(self, edge_index: torch.Tensor, num_nodes: int):
        """
        :param edge_index: tensor the shape of which is [2, E]
        :param num_nodes: number of nodes
        :return: Tensor list L the length of which is N.
            L[i] = a_ji for e_{ji} in {E}
                - a_ji = normalized attention coefficient of e_{ji} (shape: [heads, #neighbors])
        """

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E]

        att = self.cache["att"]  # [E, heads]

        att_dist_list = []
        for node_idx in range(num_nodes):
            att_neighbors = att[edge_index[1] == node_idx, :].t()  # [heads, #neighbors]
            att_dist_list.append(att_neighbors)

        return att_dist_list


@register_model("supergat")
class SuperGAT(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--num-features', type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--patience", type=int, default=100)
        parser.add_argument('--hidden-size', type=int, default=16)
        parser.add_argument("--heads", default=8, type=int)
        parser.add_argument("--out-heads", default=None, type=int)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument("--attention-type", type=str, default="basic")
        parser.add_argument("--super-gat-criterion", type=str, default=None)
        parser.add_argument("--neg-sample-ratio", type=float, default=0.5)
        parser.add_argument("--edge-sampling-ratio", type=float, default=0.8)
        parser.add_argument("--scaling-factor", type=float, default=None)
        parser.add_argument("--to-undirected-at-neg", action="store_true")
        parser.add_argument("--to-undirected", action="store_true")
        parser.add_argument("--pretraining-noise-ratio", type=float, default=0.0)
        parser.add_argument("--val-interval", type=int, default=1)
        parser.add_argument("--att-lambda", default=0., type=float)
        parser.add_argument("--total-pretraining-epoch", default=0, type=int)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.conv1 = SuperGATLayer(
            args.num_features,
            args.hidden_size,
            heads=args.heads,
            dropout=args.dropout,
            concat=True,
            is_super_gat=True,
            attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio,
            edge_sample_ratio=args.edge_sampling_ratio,
            pretraining_noise_ratio=args.pretraining_noise_ratio,
            use_pretraining=False,
            to_undirected_at_neg=args.to_undirected_at_neg,
            scaling_factor=args.scaling_factor,
        )

        self.conv2 = SuperGATLayer(
            args.hidden_size * args.heads,
            args.num_classes,
            heads=(args.out_heads or args.heads),
            dropout=args.dropout,
            concat=False,
            is_super_gat=True,
            attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio,
            edge_sample_ratio=args.edge_sampling_ratio,
            pretraining_noise_ratio=args.pretraining_noise_ratio,
            use_pretraining=False,
            to_undirected_at_neg=args.to_undirected_at_neg,
            scaling_factor=args.scaling_factor,
        )

    def forward_for_all_layers(self, x, edge_index, batch=None, **kwargs):
        x1 = F.dropout(x, p=self.args.dropout, training=self.training)
        x1 = self.conv1(x1, edge_index, batch=batch, **kwargs)
        x2 = F.elu(x1)
        x2 = F.dropout(x2, p=self.args.dropout, training=self.training)
        x2 = self.conv2(x2, edge_index, batch=batch, **kwargs)
        return x1, x2

    def forward(self, x, edge_index, batch=None, **kwargs) -> torch.Tensor:

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv1(x, edge_index, batch=batch, **kwargs)
        x = F.elu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index, batch=batch, **kwargs)

        return x

    def set_layer_attrs(self, name, value):
        setattr(self.conv1, name, value)
        setattr(self.conv2, name, value)

    def get_attention_dist_by_layer(self, edge_index, num_nodes) -> List[List[torch.Tensor]]:
        """
        :param edge_index: tensor the shape of which is [2, E]
        :param num_nodes: number of nodes
        :return List[List[torch.Tensor]]: [L, N, [#neighbors, heads]]
        """
        return [
            self.conv1.get_attention_dist(edge_index, num_nodes),
            self.conv2.get_attention_dist(edge_index, num_nodes),
        ]

    def modules(self) -> List[SuperGATLayer]:
        return [self.conv1, self.conv2]

    @staticmethod
    def get_trainer(args):
        return SuperGATTrainer


@register_model("supergat-large")
class LargeSuperGAT(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--num-features', type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--patience", type=int, default=100)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument('--hidden-size', type=int, default=8)
        parser.add_argument("--heads", default=8, type=int)
        parser.add_argument("--out-heads", default=None, type=int)
        parser.add_argument('--dropout', type=float, default=0.6)
        parser.add_argument("--attention-type", type=str, default="basic")
        parser.add_argument("--super-gat-criterion", type=str, default=None)
        parser.add_argument("--neg-sample-ratio", type=float, default=0.5)
        parser.add_argument("--edge-sampling-ratio", type=float, default=0.8)
        parser.add_argument("--scaling-factor", type=float, default=None)
        parser.add_argument("--to-undirected-at-neg", action="store_true")
        parser.add_argument("--to-undirected", action="store_true")
        parser.add_argument("--use-bn", action="store_true")
        parser.add_argument("--pretraining-noise-ratio", type=float, default=0.0)
        parser.add_argument("--val-interval", type=int, default=1)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_layers = self.args.num_layers

        conv_common_kwargs = dict(
            dropout=args.dropout,
            is_super_gat=True,
            attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio,
            edge_sample_ratio=args.edge_sampling_ratio,
            pretraining_noise_ratio=args.pretraining_noise_ratio,
            use_pretraining=args.use_pretraining,
            to_undirected_at_neg=args.to_undirected_at_neg,
            scaling_factor=args.scaling_factor,
        )
        self.conv_list = []
        self.bn_list = []
        for conv_id in range(1, self.num_layers + 1):
            if conv_id == 1:  # first layer
                in_channels, out_channels = args.num_features, args.hidden_size
                heads, concat = args.heads, True
            elif conv_id == self.num_layers:  # last layer
                in_channels, out_channels = args.hidden_size * args.heads, args.num_classes
                heads, concat = args.out_heads or args.heads, False
            else:
                in_channels, out_channels = args.hidden_size * args.heads, args.hidden_size
                heads, concat = args.heads, True
            # conv
            conv = SuperGATLayer(in_channels, out_channels, heads=heads, concat=concat, **conv_common_kwargs)
            conv_name = "conv{}".format(conv_id)
            self.conv_list.append(conv)
            setattr(self, conv_name, conv)
            self.add_module(conv_name, conv)
            # bn
            if args.use_bn and conv_id != self.num_layers:  # not last layer
                bn = nn.BatchNorm1d(out_channels * heads)
                bn_name = "bn{}".format(conv_id)
                self.bn_list.append(bn)
                setattr(self, bn_name, bn)
                self.add_module(bn_name, bn)

        print(next(self.modules()))

    def forward(self, x, edge_index, batch=None, **kwargs) -> torch.Tensor:
        for conv_idx, conv in enumerate(self.conv_list):
            x = F.dropout(x, p=self.args.dropout, training=self.training)
            x = conv(x, edge_index, **kwargs)
            if conv_idx != self.num_layers - 1:
                if self.args.use_bn:
                    x = self.bn_list[conv_idx](x)
                x = F.elu(x)
        return x

    def set_layer_attrs(self, name, value):
        for conv in self.conv_list:
            setattr(conv, name, value)

    def get_attention_dist_by_layer(self, edge_index, num_nodes) -> List[List[torch.Tensor]]:
        """
        :param edge_index: tensor the shape of which is [2, E]
        :param num_nodes: number of nodes
        :return List[List[torch.Tensor]]: [L, N, [#neighbors, heads]]
        """
        attention_dist_by_layer = []
        for conv in self.conv_list:
            attention_dist_by_layer.append(conv.get_attention_dist(edge_index, num_nodes))
        return attention_dist_by_layer

    def modules(self) -> List[SuperGATLayer]:
        return self.conv_list

    @staticmethod
    def get_trainer(args):
        return SuperGATTrainer
