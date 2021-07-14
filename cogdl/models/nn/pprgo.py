from typing import Any
import torch

from .. import BaseModel, register_model
from cogdl.utils import spmm
from cogdl.layers import PPRGoLayer
from cogdl.trainers.ppr_trainer import PPRGoTrainer


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
        self.fc = PPRGoLayer(in_feats, hidden_size, out_feats, num_layers, dropout, activation)

    def forward(self, x, targets, ppr_scores):
        h = self.fc(x)
        h = ppr_scores.unsqueeze(1) * h
        batch_size = targets[-1] + 1
        out = torch.zeros(batch_size, h.shape[1]).to(x.device).to(x.dtype)
        out = out.scatter_add_(dim=0, index=targets[:, None].repeat(1, h.shape[1]), src=h)
        return out

    def node_classification_loss(self, x, targets, ppr_scores, y):
        pred = self.forward(x, targets, ppr_scores)
        loss = self.loss_fn(pred, y)
        return loss

    def predict(self, graph, batch_size, norm):
        device = next(self.fc.parameters()).device
        x = graph.x
        num_nodes = x.shape[0]
        pred_logits = []
        with torch.no_grad():
            for i in range(0, num_nodes, batch_size):
                batch_x = x[i : i + batch_size].to(device)
                batch_logits = self.fc(batch_x)
                pred_logits.append(batch_logits.cpu())
        pred_logits = torch.cat(pred_logits, dim=0)

        with graph.local_graph():
            if norm == "sym":
                graph.sym_norm()
            elif norm == "row":
                graph.row_norm()
            else:
                raise NotImplementedError
            edge_weight = graph.edge_weight * (1 - self.alpha)

            graph.edge_weight = edge_weight
            predictions = pred_logits
            for _ in range(self.nprop):
                predictions = spmm(graph, predictions) + self.alpha * pred_logits
        return predictions

    @staticmethod
    def get_trainer(args: Any):
        return PPRGoTrainer
