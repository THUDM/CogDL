import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .. import BaseModel, register_model
from cogdl.trainers.sampled_trainer import SAINTTrainer
from cogdl.utils import spmm


F_ACT = {"relu": nn.ReLU(), "I": lambda x: x}

"""
Borrowed from https://github.com/GraphSAINT/GraphSAINT
"""


class HighOrderAggregator(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, act="relu", order=1, aggr="mean", bias="norm-nn", **kwargs):
        """
        Layer implemented here combines the GraphSAGE-mean [1] layer with MixHop [2] layer.
        We define the concept of `order`: an order-k layer aggregates neighbor information
        from 0-hop all the way to k-hop. The operation is approximately:
            X W_0 [+] A X W_1 [+] ... [+] A^k X W_k
        where [+] is some aggregation operation such as addition or concatenation.

        Special cases:
            Order = 0  -->  standard MLP layer
            Order = 1  -->  standard GraphSAGE layer

        [1]: https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
        [2]: https://arxiv.org/abs/1905.00067

        Inputs:
            dim_in      int, feature dimension for input nodes
            dim_out     int, feature dimension for output nodes
            dropout     float, dropout on weight matrices W_0 to W_k
            act         str, activation function. See F_ACT at the top of this file
            order       int, see definition above
            aggr        str, if 'mean' then [+] operation adds features of various hops
                            if 'concat' then [+] concatenates features of various hops
            bias        str, if 'bias' then apply a bias vector to features of each hop
                            if 'norm' then perform batch-normalization on output features

        Outputs:
            None
        """
        super(HighOrderAggregator, self).__init__()
        assert bias in ["bias", "norm", "norm-nn"]
        self.order, self.aggr = order, aggr
        self.act, self.bias = F_ACT[act], bias
        self.dropout = dropout
        self.f_lin, self.f_bias = [], []
        self.offset, self.scale = [], []
        self.num_param = 0
        for o in range(self.order + 1):
            self.f_lin.append(nn.Linear(dim_in, dim_out, bias=False))
            nn.init.xavier_uniform_(self.f_lin[-1].weight)
            self.f_bias.append(nn.Parameter(torch.zeros(dim_out)))
            self.num_param += dim_in * dim_out
            self.num_param += dim_out
            self.offset.append(nn.Parameter(torch.zeros(dim_out)))
            self.scale.append(nn.Parameter(torch.ones(dim_out)))
            if self.bias == "norm" or self.bias == "norm-nn":
                self.num_param += 2 * dim_out
        self.f_lin = nn.ModuleList(self.f_lin)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.params = nn.ParameterList(self.f_bias + self.offset + self.scale)
        self.f_bias = self.params[: self.order + 1]
        if self.bias == "norm":
            self.offset = self.params[self.order + 1 : 2 * self.order + 2]
            self.scale = self.params[2 * self.order + 2 :]
        elif self.bias == "norm-nn":
            final_dim_out = dim_out * ((aggr == "concat") * (order + 1) + (aggr == "mean"))
            self.f_norm = nn.BatchNorm1d(final_dim_out, eps=1e-9, track_running_stats=True)
        self.num_param = int(self.num_param)

    def _f_feat_trans(self, _feat, _id):
        feat = self.act(self.f_lin[_id](_feat) + self.f_bias[_id])
        if self.bias == "norm":
            mean = feat.mean(dim=1).view(feat.shape[0], 1)
            var = feat.var(dim=1, unbiased=False).view(feat.shape[0], 1) + 1e-9
            feat_out = (feat - mean) * self.scale[_id] * torch.rsqrt(var) + self.offset[_id]
        else:
            feat_out = feat
        return feat_out

    def forward(self, input):
        """
        Inputs:.
            adj_norm        normalized adj matrix of the subgraph
            feat_in         2D matrix of input node features

        Outputs:
            adj_norm        same as input (to facilitate nn.Sequential)
            feat_out        2D matrix of output node features
        """

        graph, x = input
        feat_in = self.f_dropout(x)
        feat_hop = [feat_in]
        # generate A^i X
        for o in range(self.order):
            feat_hop.append(spmm(graph, x))
        feat_partial = [self._f_feat_trans(ft, idf) for idf, ft in enumerate(feat_hop)]
        if self.aggr == "mean":
            feat_out = feat_partial[0]
            for o in range(len(feat_partial) - 1):
                feat_out += feat_partial[o + 1]
        elif self.aggr == "concat":
            feat_out = torch.cat(feat_partial, 1)
        else:
            raise NotImplementedError
        if self.bias == "norm-nn":
            feat_out = self.f_norm(feat_out)
        return graph, feat_out  # return adj_norm to support Sequential


def parse_arch(architecture, aggr, act, bias, hidden_size, num_features):
    num_layers = len(architecture.split("-"))
    # set default values, then update by arch_gcn
    bias_layer = [bias] * num_layers
    act_layer = [act] * num_layers
    aggr_layer = [aggr] * num_layers
    dims_layer = [hidden_size] * num_layers
    order_layer = [int(order) for order in architecture.split("-")]
    return [num_features] + dims_layer, order_layer, act_layer, bias_layer, aggr_layer


@register_model("graphsaint")
class GraphSAINT(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--architecture", type=str, default="1-1-0")
        parser.add_argument("--aggr", type=str, default="concat")
        parser.add_argument("--act", type=str, default="relu")
        parser.add_argument("--bias", type=str, default="norm")
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--weight-decay", type=int, default=0)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.architecture,
            args.aggr,
            args.act,
            args.bias,
            args.weight_decay,
            args.dropout,
            args.hidden_size,
        )

    def __init__(self, num_features, num_classes, architecture, aggr, act, bias, weight_decay, dropout, hidden_size):
        """
        Build the multi-layer GNN architecture.

        Inputs:
            num_classes         int, number of classes a node can belong to
            arch_gcn            dict, config for each GNN layer
            train_params        dict, training hyperparameters (e.g., learning rate)
            feat_full           np array of shape N x f, where N is the total num of
                                nodes and f is the dimension for input node feature
            label_full          np array, for single-class classification, the shape
                                is N x 1 and for multi-class classification, the
                                shape is N x c (where c = num_classes)
            cpu_eval            bool, if True, will put the model on CPU.

        Outputs:
            None
        """
        super(GraphSAINT, self).__init__()
        self.aggregator_cls = HighOrderAggregator
        self.mulhead = 1
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.sigmoid_loss = True
        self.num_classes = num_classes
        self.num_layers = len(architecture.split("-"))
        _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer = parse_arch(
            architecture, aggr, act, bias, hidden_size, num_features
        )
        # get layer index for each conv layer, useful for jk net last layer aggregation
        self.set_idx_conv()
        self.set_dims(_dims)

        self.loss = 0
        self.opt_op = None

        # build the model below
        self.num_params = 0
        self.aggregators, num_param = self.get_aggregators()
        self.num_params += num_param
        self.conv_layers = nn.Sequential(*self.aggregators)
        self.classifier = HighOrderAggregator(
            self.dims_feat[-1], self.num_classes, act="I", order=0, dropout=self.dropout, bias="bias"
        )
        self.num_params += self.classifier.num_param

    def set_dims(self, dims):
        """
        Set the feature dimension / weight dimension for each GNN or MLP layer.
        We will use the dimensions set here to initialize PyTorch layers.

        Inputs:
            dims        list, length of node feature for each hidden layer

        Outputs:
            None
        """
        self.dims_feat = [dims[0]] + [
            ((self.aggr_layer[layer] == "concat") * self.order_layer[layer] + 1) * dims[layer + 1]
            for layer in range(len(dims) - 1)
        ]
        self.dims_weight = [(self.dims_feat[layer], dims[layer + 1]) for layer in range(len(dims) - 1)]

    def set_idx_conv(self):
        """
        Set the index of GNN layers for the full neural net. For example, if
        the full NN is having 1-0-1-0 arch (1-hop graph conv, followed by 0-hop
        MLP, ...). Then the layer indices will be 0, 2.
        """
        idx_conv = np.where(np.array(self.order_layer) >= 1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer) - 1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer) == 1)[0])

    def forward(self, graph):
        x = graph.x
        _, emb_subg = self.conv_layers(((graph, x)))
        emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
        pred_subg = self.classifier((None, emb_subg_norm))[1]
        return pred_subg

    def _loss(self, preds, labels, norm_loss):
        """
        The predictor performs sigmoid (for multi-class) or softmax (for single-class)
        """
        if self.sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)
            return torch.nn.BCEWithLogitsLoss(weight=norm_loss, reduction="sum")(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction="none")(preds, labels)
            return (norm_loss * _ls).sum()

    def get_aggregators(self):
        """
        Return a list of aggregator instances. to be used in self.build()
        """
        num_param = 0
        aggregators = []
        for layer in range(self.num_layers):
            aggr = self.aggregator_cls(
                *self.dims_weight[layer],
                dropout=self.dropout,
                act=self.act_layer[layer],
                order=self.order_layer[layer],
                aggr=self.aggr_layer[layer],
                bias=self.bias_layer[layer],
                mulhead=self.mulhead,
            )
            num_param += aggr.num_param
            aggregators.append(aggr)
        return aggregators, num_param

    def predict(self, data):
        return self.forward(data)
        # return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1

    @staticmethod
    def get_trainer(task, args):
        return SAINTTrainer
