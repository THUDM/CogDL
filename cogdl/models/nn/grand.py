import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from cogdl.layers import MLP
from cogdl.utils import spmm


@register_model("grand")
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
        parser.add_argument("--bn", type=bool, default=False)
        parser.add_argument("--dropnode-rate", type=float, default=0.5)
        parser.add_argument('--order', type=int, default=5)
        parser.add_argument('--tem', type=float, default=0.5)
        parser.add_argument('--lam', type=float, default=0.5)
        parser.add_argument('--sample', type=int, default=2)    
        parser.add_argument('--alpha', type=float, default=0.2)        

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
            args.tem,
            args.lam,
            args.order,
            args.sample,
            args.alpha,
        )

    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        input_droprate,
        hidden_droprate,
        use_bn,
        dropnode_rate,
        tem,
        lam,
        order,
        sample,
        alpha,
    ):
        super(Grand, self).__init__()
        self.layer1 = nn.Linear(nfeat, nhid)
        self.layer2 = nn.Linear(nhid, nclass)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.tem = tem
        self.lam = lam
        self.order = order
        self.dropnode_rate = dropnode_rate
        self.sample = sample
        self.alpha = alpha

    def dropNode(self, x):
        n = x.shape[0]
        drop_rates = torch.ones(n) * self.dropnode_rate
        if self.training:
            masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(1)
            x = masks.to(x.device) * x

        else:
            x = x * (1.0 - self.dropnode_rate)
        return x

    def rand_prop(self, graph, x):
        x = self.dropNode(x)

        y = x
        for i in range(self.order):
            x = spmm(graph, x).detach_()
            y.add_(x)
        return y.div_(self.order + 1.0).detach_()

    def consis_loss(self, logps, train_mask):
        temp = self.tem
        ps = [torch.exp(p)[~train_mask] for p in logps]
        sum_p = 0.0
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p / len(ps)
        sharp_p = (torch.pow(avg_p, 1.0 / temp) / torch.sum(torch.pow(avg_p, 1.0 / temp), dim=1, keepdim=True)).detach()
        loss = 0.0
        for p in ps:
            loss += torch.mean((p - sharp_p).pow(2).sum(1))
        loss = loss / len(ps)

        return self.lam * loss

    def normalize_x(self, x):
        row_sum = x.sum(1)
        row_inv = row_sum.pow_(-1)
        row_inv.masked_fill_(row_inv == float("inf"), 0)
        x = x * row_inv[:, None]
        return x

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        x = self.normalize_x(x)
        x = self.rand_prop(graph, x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)
        return x

    def node_classification_loss(self, graph):
        output_list = []
        for i in range(self.sample):
            output_list.append(self.forward(graph))
        loss_train = 0.0
        for output in output_list:
            loss_train += self.loss_fn(output[graph.train_mask], graph.y[graph.train_mask])
        loss_train = loss_train / self.sample

        if len(graph.y.shape) > 1:
            output_list = [torch.sigmoid(x) for x in output_list]
        else:
            output_list = [F.log_softmax(x, dim=-1) for x in output_list]
        loss_consis = self.consis_loss(output_list, graph.train_mask)

        return loss_train + loss_consis

    def predict(self, data):
        return self.forward(data)
