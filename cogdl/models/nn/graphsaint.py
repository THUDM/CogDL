import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .. import BaseModel
from cogdl.layers import SAINTLayer


def parse_arch(architecture, aggr, act, bias, hidden_size, num_features):
    num_layers = len(architecture.split("-"))
    # set default values, then update by arch_gcn
    bias_layer = [bias] * num_layers
    act_layer = [act] * num_layers
    aggr_layer = [aggr] * num_layers
    dims_layer = [hidden_size] * num_layers
    order_layer = [int(order) for order in architecture.split("-")]
    return [num_features] + dims_layer, order_layer, act_layer, bias_layer, aggr_layer


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
            args.dropout,
            args.hidden_size,
        )

    def __init__(self, num_features, num_classes, architecture, aggr, act, bias, dropout, hidden_size):
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
        self.aggregator_cls = SAINTLayer
        self.mulhead = 1
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
        self.conv_layers = nn.ModuleList(self.aggregators)
        self.classifier = SAINTLayer(
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
        for layer in self.conv_layers:
            x = layer(graph, x)
        emb_subg_norm = F.normalize(x, p=2, dim=1)
        pred_subg = self.classifier(None, emb_subg_norm)
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
