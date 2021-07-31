import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from conv import myGATConv

import torch.nn.functional as F

from cogdl import experiment, options
from cogdl.models import BaseModel, register_model
from cogdl.utils import accuracy
from cogdl.data import Graph


@register_model("simple_hgn")
class SimpleHGN(BaseModel):
    r"""The Simple-HGN model from the `"Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks"`_ paper

    Args:
        num_features (int) : Number of input features.
        num_classes (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

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
        parser.add_argument("--num-heads", type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--slope', type=float, default=0.05)
        parser.add_argument('--edge-dim', type=int, default=64)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        heads = [args.num_heads] * args.num_layers + [1]
        return cls(
            args.edge_dim,
            args.num_edge * 2 + 1,
            [args.num_features],
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            heads,
            args.dropout,
            args.dropout,
            args.slope,
            True,
            0.05,
            True,
        )

    def __init__(
        self,
        edge_dim,
        num_etypes,
        in_dims,
        num_hidden,
        num_classes,
        num_layers,
        heads,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        alpha,
        use_cuda,
    ):
        super(SimpleHGN, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
        self.g = None
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu
        # self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        # for fc in self.fc_list:
        #    nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(
            myGATConv(
                edge_dim,
                num_etypes,
                in_dims[0],
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                alpha=alpha,
            )
        )
        # hidden layers
        for l in range(1, num_layers):  # noqa E741
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                myGATConv(
                    edge_dim,
                    num_etypes,
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    alpha=alpha,
                )
            )
        # output projection
        self.gat_layers.append(
            myGATConv(
                edge_dim,
                num_etypes,
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                alpha=alpha,
            )
        )
        self.epsilon = torch.FloatTensor([1e-12]).to(self.device)

    def build_g_feat(self, A):
        edge2type = {}
        edges = []
        weights = []
        for k, mat in enumerate(A):
            edges.append(mat[0].cpu().numpy())
            weights.append(mat[1].cpu().numpy())
            for u, v in zip(*edges[-1]):
                edge2type[(u, v)] = k
        edges = np.concatenate(edges, axis=1)
        weights = np.concatenate(weights)
        edges = torch.tensor(edges).to(self.device)
        weights = torch.tensor(weights).to(self.device)
        
        g = Graph(edge_index=edges, edge_weight=weights)
        g = g.to(self.device)
        e_feat = []
        for u, v in zip(*g.edge_index):
            u = u.cpu().item()
            v = v.cpu().item()
            e_feat.append(edge2type[(u, v)])
        e_feat = torch.tensor(e_feat, dtype=torch.long).to(self.device)
        g.edge_type = e_feat
        self.g = g

    def forward(self, A, X, target_x, target):
        # h = []
        # for fc, feature in zip(self.fc_list, [X]):
        #    h.append(fc(feature))
        h = X  # torch.cat(h, 0)
        if self.g is None:
            self.build_g_feat(A)
        res_attn = None
        for l in range(self.num_layers):  # noqa E741
            h, res_attn = self.gat_layers[l](self.g, h, res_attn=res_attn)
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, res_attn=None)
        # logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        y = logits[target_x]
        loss = self.cross_entropy_loss(y, target)
        return loss, y, None

    def loss(self, data):
        loss, y, _ = self.forward(data.adj, data.x, data.train_node, data.train_target)
        return loss

    def evaluate(self, data, nodes, targets):
        loss, y, _ = self.forward(data.adj, data.x, nodes, targets)
        f1 = accuracy(y, targets)
        return loss.item(), f1


if __name__ == "__main__":
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    experiment(task="heterogeneous_node_classification", dataset="gtn-acm", model="simple_hgn", args=args)
