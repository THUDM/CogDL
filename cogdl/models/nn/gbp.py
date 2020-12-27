import torch.nn as nn
import torch
import math
import torch.nn.functional as F

from .. import BaseModel, register_model


class Dense(nn.Module):
    r"""
    GBP layer: https://arxiv.org/pdf/2010.15421.pdf
    """

    def __init__(self, in_features, out_features, bias="none"):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        if bias == "bn":
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)

        if self.in_features == self.out_features:
            output = output + input

        return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


@register_model("gbp")
class GnnBP(BaseModel):
    r"""
        The GBP model from the `"Scalable Graph Neural Networks via Bidirectional
    Propagation"
        <https://arxiv.org/pdf/2010.15421.pdf>`_ paper

        Args:
            num_features (int) : Number of input features.
            num_layers (int) : the number of hidden layers
            hidden_size (int) : The dimension of node representation.
            num_classes (int) : Number of classes.
            dropout (float) : Dropout rate for model training.
            bias (str) : bias
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument('--alpha', type=float, default=0.1, help='decay factor')
        parser.add_argument('--rmax', type=float, default=1e-5, help='threshold.')
        parser.add_argument('--rrz', type=float, default=0.0, help='r.')
        parser.add_argument("--bias", default='none')
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_layers, args.hidden_size, args.num_classes, args.dropout, args.bias)

    def __init__(self, num_features, num_layers, hidden_size, num_classes, dropout, bias):
        super(GnnBP, self).__init__()

        self.fcs = nn.ModuleList()
        self.fcs.append(Dense(num_features, hidden_size, bias))
        for _ in range(num_layers - 2):
            self.fcs.append(Dense(hidden_size, hidden_size, bias))
        self.fcs.append(Dense(hidden_size, num_classes))
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for fc in self.fcs[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x

    def node_classification_loss(self, data):
        pred = self.forward(data.x)
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(
            pred[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data.x)
