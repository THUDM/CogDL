from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel
from cogdl.layers import GCNLayer
from cogdl.utils import get_activation
from cogdl.data import Graph


class GraceEncoder(nn.Module):
    def __init__(
        self, in_feats: int, out_feats: int, num_layers: int, activation: str = "relu",
    ):
        super(GraceEncoder, self).__init__()
        shapes = [in_feats] + [2 * out_feats] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList([GCNLayer(shapes[i], shapes[i + 1]) for i in range(num_layers)])
        self.activation = get_activation(activation)

    def forward(self, graph: Graph, x: torch.Tensor):
        h = x
        for layer in self.layers:
            h = layer(graph, h)
            h = self.activation(h)
        return h


class GRACE(BaseModel):
    @staticmethod
    def add_args(parser):
        # fmt : off
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--proj-hidden-size", type=int, default=128)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--drop-feature-rates", type=float, nargs="+", default=[0.3, 0.4])
        parser.add_argument("--drop-edge-rates", type=float, nargs="+", default=[0.2, 0.4])
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--batch-size", type=int, default=-1)
        parser.add_argument("--tau", type=float, default=0.4)
        # fmt : on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_feats=args.num_features,
            hidden_size=args.hidden_size,
            proj_hidden_size=args.proj_hidden_size,
            num_layers=args.num_layers,
            drop_feature_rates=args.drop_feature_rates,
            drop_edge_rates=args.drop_edge_rates,
            tau=args.tau,
            activation=args.activation,
            batch_size=args.batch_size,
        )

    def __init__(
        self,
        in_feats: int,
        hidden_size: int,
        proj_hidden_size: int,
        num_layers: int,
        drop_feature_rates: List[float],
        drop_edge_rates: List[float],
        tau: float = 0.5,
        activation: str = "relu",
        batch_size: int = -1,
    ):
        super(GRACE, self).__init__()

        self.tau = tau
        self.drop_feature_rates = drop_feature_rates
        self.drop_edge_rates = drop_edge_rates
        self.batch_size = batch_size

        self.project_head = nn.Sequential(
            nn.Linear(hidden_size, proj_hidden_size), nn.ELU(), nn.Linear(proj_hidden_size, hidden_size)
        )
        self.encoder = GraceEncoder(in_feats, hidden_size, num_layers, activation)

    def forward(
        self, graph: Graph, x: torch.Tensor = None,
    ):
        if x is None:
            x = graph.x
        graph.sym_norm()
        return self.encoder(graph, x)

    def embed(self, data):
        pred = self.forward(data, data.x)
        return pred
