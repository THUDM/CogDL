import math

from typing import Any
import torch
import torch.nn as nn

from .. import BaseModel, register_model
from cogdl.utils import get_activation, spmm
from cogdl.trainers.ppr_trainer import PPRGoTrainer


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / nn.math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)


class PPRGoMLP(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_layers, dropout, activation="relu"):
        super(PPRGoMLP, self).__init__()
        self.dropout = dropout
        self.nlayers = num_layers
        shapes = [hidden_size] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList()
        self.layers.append(LinearLayer(in_feats, hidden_size, bias=False))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(shapes[i], shapes[i + 1], bias=False))
        self.activation = get_activation(activation)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)
            h = layer(h)
            if i != self.nlayers - 1:
                h = self.activation(h)
        return h


@register_model("pprgo")
class PPRGo(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--nprop-inference", type=int, default=2)

        parser.add_argument("--alpha", type=float, default=0.5)
        parser.add_argument("--k", type=int, default=32)
        parser.add_argument("--norm", type=str, default="sym")
        parser.add_argument("--eps", type=float, default=1e-4)

        parser.add_argument("--eval-step", type=int, default=4)
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--test-batch-size", type=int, default=10000)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_feats=args.num_features,
            hidden_size=args.hidden_size,
            out_feats=args.num_classes,
            num_layers=args.num_layers,
            alpha=args.alpha,
            dropout=args.dropout,
            activation=args.activation,
            nprop=args.nprop_inference,
        )

    def __init__(self, in_feats, hidden_size, out_feats, num_layers, alpha, dropout, activation="relu", nprop=2):
        super(PPRGo, self).__init__()
        self.alpha = alpha
        self.nprop = nprop
        self.fc = PPRGoMLP(in_feats, hidden_size, out_feats, num_layers, dropout, activation)

    def forward(self, x, targets, ppr_scores):
        h = self.fc(x)
        h = ppr_scores.unsqueeze(1) * h
        batch_size = targets[-1] + 1
        out = torch.zeros(batch_size, h.shape[1]).to(x.device).to(x.dtype)
        out = out.scatter_add_(dim=0, index=targets[:, None].repeat(1, h.shape[1]), src=h)
        return out

    def node_classification_loss(self, x, targets, ppr_scores, y):
        pred = self.forward(x, targets, ppr_scores)
        pred = nn.functional.log_softmax(pred, dim=-1)
        loss = nn.functional.nll_loss(pred, y)
        return loss

    def predict(self, x, edge_index, batch_size, norm_func):
        device = next(self.fc.parameters()).device
        num_nodes = x.shape[0]
        pred_logits = []
        with torch.no_grad():
            for i in range(0, num_nodes, batch_size):
                batch_x = x[i : i + batch_size].to(device)
                batch_logits = self.fc(batch_x)
                pred_logits.append(batch_logits.cpu())
        pred_logits = torch.cat(pred_logits, dim=0)

        edge_weight = norm_func(num_nodes, edge_index)
        edge_weight = edge_weight * (1 - self.alpha)

        predictions = pred_logits
        for _ in range(self.nprop):
            predictions = spmm(edge_index, edge_weight, predictions) + self.alpha * pred_logits
        return predictions

    @staticmethod
    def get_trainer(taskType: Any, args: Any):
        return PPRGoTrainer
