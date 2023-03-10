import jittor
from jittor import nn
from cogdl.models import BaseModel
from cogdl.utils import spmm
from cogdl.datasets.planetoid_data import CoraDataset,PubMedDataset

class Grand(BaseModel):
    """
    Implementation of GRAND in paper `"Graph Random Neural Networks for Semi-Supervised Learning on Graphs"`
    <https://arxiv.org/abs/2005.11079>

    Parameters
    ----------
    nfeat : int
        Size of each input features.
    nhid : int
        Size of hidden features.
    nclass : int
        Number of output classes.
    input_droprate : float
        Dropout rate of input features.
    hidden_droprate : float
        Dropout rate of hidden features.
    use_bn : bool
        Using batch normalization.
    dropnode_rate : float
        Rate of dropping elements of input features
    tem : float
        Temperature to sharpen predictions.
    lam : float
         Proportion of consistency loss of unlabelled data
    order : int
        Order of adjacency matrix
    sample : int
        Number of augmentations for consistency loss
    alpha : float
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--hidden-dropout", type=float, default=0.5)
        parser.add_argument("--input-dropout", type=float, default=0.5)
        parser.add_argument("--bn", type=bool, default=True)
        parser.add_argument("--dropnode-rate", type=float, default=0.5)
        parser.add_argument("--order", type=int, default=5)
        parser.add_argument("--alpha", type=float, default=0.2)

        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.input_dropout,
            args.hidden_dropout,
            args.bn,
            args.dropnode_rate,
            args.order,
            args.alpha,
        )

    def __init__(
        self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn, dropnode_rate, order, alpha,
    ):
        super(Grand, self).__init__()
        self.layer1 = nn.Linear(nfeat, nhid)
        self.layer2 = nn.Linear(nhid, nclass)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.order = order
        self.dropnode_rate = dropnode_rate
        self.alpha = alpha

    def drop_node(self, x):
        n = x.shape[0]
        drop_rates = jittor.ones(n) * self.dropnode_rate
        if self.is_train:
            masks = jittor.bernoulli(1.0 - drop_rates).unsqueeze(1)
            x = masks * x

        else:
            x = x * (1.0 - self.dropnode_rate)
        return x

    def rand_prop(self, graph, x):
        x = self.drop_node(x)

        y = x
        for i in range(self.order):
            x = spmm(graph, x)
            y.add_(x)
        return y.divide(self.order + 1.0)

    def normalize_x(self, x):
        row_sum = x.sum(1)
        row_inv = row_sum.pow(-1)
        row_inv[row_inv == float("inf")] = 0
        x = x * row_inv[:, None]
        return x

    def execute(self, graph):
        graph.sym_norm()
        x = graph.x
        x = self.normalize_x(x)
        x = self.rand_prop(graph, x)
        if self.use_bn:
            x = self.bn1(x)
        x = nn.dropout(x, self.input_droprate, is_train=self.is_train)
        x = nn.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = nn.dropout(x, self.hidden_droprate, is_train=self.is_train)
        x = self.layer2(x)
        return x

    def predict(self, data):
        return self.execute(data)

