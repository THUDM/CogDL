from typing import Optional, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import register_model, BaseModel
from cogdl.models.nn.gcn import GraphConvolution
from cogdl.utils import get_activation, filter_adj, add_remaining_self_loops, symmetric_normalization
from cogdl.trainers.self_supervised_trainer import SelfSupervisedTrainer


class GraceEncoder(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_layers: int,
        activation: str = "relu",
    ):
        super(GraceEncoder, self).__init__()
        shapes = [in_feats] + [2 * out_feats] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList([GraphConvolution(shapes[i], shapes[i + 1]) for i in range(num_layers)])
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
            h = self.activation(h)
        return h


@register_model("grace")
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
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        num_nodes = x.shape[0]
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight)
        edge_weight = symmetric_normalization(num_nodes, edge_index)
        return self.encoder(x, edge_index, edge_weight)

    def prop(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        drop_feature_rate: float = 0.0,
        drop_edge_rate: float = 0.0,
    ):
        x = self.drop_feature(x, drop_feature_rate)
        edge_index, edge_weight = self.drop_adj(edge_index, edge_weight, drop_edge_rate)
        return self.forward(x, edge_index, edge_weight)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        def score_func(emb1, emb2):
            scores = torch.matmul(emb1, emb2.t())
            scores = torch.exp(scores / self.tau)
            return scores

        intro_scores = score_func(z1, z1)
        inter_scores = score_func(z1, z2)

        _loss = -torch.log(intro_scores.diag() / (intro_scores.sum(1) - intro_scores.diag() + inter_scores.sum(1)))
        return torch.mean(_loss)

    def batched_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch_size: int,
    ):
        num_nodes = z1.shape[0]
        num_batches = (num_nodes - 1) // batch_size + 1

        losses = []
        indices = torch.arange(num_nodes).to(z1.device)
        for i in range(num_batches):
            train_indices = indices[i * batch_size : (i + 1) * batch_size]
            _loss = self.contrastive_loss(z1[train_indices], z2)
            losses.append(_loss)
        return sum(losses) / len(losses)

    def node_classification_loss(self, data):
        z1 = self.prop(data.x, data.edge_index, data.edge_attr, self.drop_feature_rates[0], self.drop_edge_rates[0])
        z2 = self.prop(data.x, data.edge_index, data.edge_attr, self.drop_feature_rates[1], self.drop_edge_rates[1])

        z1 = self.project_head(z1)
        z2 = self.project_head(z2)

        if self.batch_size > 0:
            return 0.5 * (self.batched_loss(z1, z2, self.batch_size) + self.batched_loss(z2, z1, self.batch_size))
        else:
            return 0.5 * (self.contrastive_loss(z1, z2) + self.contrastive_loss(z2, z1))

    def embed(self, data):
        pred = self.forward(data.x, data.edge_index, data.edge_attr)
        return pred

    def drop_adj(self, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.5):
        if drop_rate < 0.0 or drop_rate > 1.0:
            raise ValueError("Dropout probability has to be between 0 and 1, " "but got {}".format(drop_rate))
        if not self.training:
            return edge_index, edge_weight

        mask = edge_index.new_full((edge_index.size(1),), 1 - drop_rate, dtype=torch.float)
        mask = torch.bernoulli(mask).to(torch.bool)
        edge_index, edge_weight = filter_adj(edge_index[0], edge_index[1], edge_weight, mask)
        return edge_index, edge_weight

    def drop_feature(self, x: torch.Tensor, droprate: float):
        n = x.shape[1]
        drop_rates = torch.ones(n) * droprate
        if self.training:
            masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(0).expand_as(x)
            x = masks.to(x.device) * x
        return x

    @staticmethod
    def get_trainer(tasktype: Any, args):
        return SelfSupervisedTrainer
