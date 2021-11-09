import numpy as np
import torch
import torch.nn as nn

from conv import myGATConv

import torch.nn.functional as F

from cogdl import experiment, options
from cogdl.datasets.gtn_data import ACM_GTNDataset, DBLP_GTNDataset, IMDB_GTNDataset
from cogdl.models import BaseModel
from cogdl.data import Graph


class SimpleHGN(BaseModel):
    r"""The Simple-HGN model from the `"Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks"`_ paper"""

    def __init__(
        self,
        in_dims,
        num_classes,
        edge_dim=64,
        num_etypes=5,
        num_hidden=64,
        num_layers=2,
        heads=[8, 8, 1],
        feat_drop=0.5,
        attn_drop=0.5,
        negative_slope=0.05,
        residual=True,
        alpha=0.05,
    ):
        super(SimpleHGN, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.g = None
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu
        # input projection (no residual)
        self.gat_layers.append(
            myGATConv(
                edge_dim,
                num_etypes,
                in_dims,
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
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))

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

    def forward(self, data):
        A = data.adj
        X = data.x
        h = X
        if self.g is None:
            self.build_g_feat(A)
        res_attn = None
        for l in range(self.num_layers):  # noqa E741
            h, res_attn = self.gat_layers[l](self.g, h, res_attn=res_attn)
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, res_attn=None)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))

        return logits


if __name__ == "__main__":
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args.mw = "heterogeneous_gnn_mw"
    args.dw = "heterogeneous_gnn_dw"
    args = options.parse_args_and_arch(parser, args)
    if args.dataset[0] == "gtn-acm":
        dataset = ACM_GTNDataset()
    elif args.dataset[0] == "gtn-dblp":
        dataset = DBLP_GTNDataset()
    elif args.dataset[0] == "gtn-imdb":
        dataset = IMDB_GTNDataset()
    else:
        raise NotImplementedError
    hgn = SimpleHGN(in_dims=dataset.num_features, num_classes=dataset.num_classes)
    experiment(dataset=dataset, model=hgn, dw="heterogeneous_gnn_dw", mw="heterogeneous_gnn_mw", args=args)
