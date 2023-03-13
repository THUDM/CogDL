import torch

from .. import BaseModel
from cogdl.utils import spmm
from cogdl.layers import PPRGoLayer


class PPRGo(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--nprop-inference", type=int, default=2)
        parser.add_argument("--alpha", type=float, default=0.5)

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
            norm=args.norm if hasattr(args, "norm") else "sym",
        )

    def __init__(
        self, in_feats, hidden_size, out_feats, num_layers, alpha, dropout, activation="relu", nprop=2, norm="sym"
    ):
        super(PPRGo, self).__init__()
        self.alpha = alpha
        self.norm = norm
        self.nprop = nprop
        self.fc = PPRGoLayer(in_feats, hidden_size, out_feats, num_layers, dropout, activation)

    def forward(self, x, targets, ppr_scores):
        h = self.fc(x)
        h = ppr_scores.unsqueeze(1) * h
        batch_size = targets[-1] + 1
        out = torch.zeros(batch_size, h.shape[1]).to(x.device).to(x.dtype)
        out = out.scatter_add_(dim=0, index=targets[:, None].repeat(1, h.shape[1]), src=h)
        return out

    def predict(self, graph, batch_size=10000):
        device = next(self.parameters()).device
        x = graph.x
        num_nodes = x.shape[0]
        pred_logits = []
        with torch.no_grad():
            for i in range(0, num_nodes, batch_size):
                batch_x = x[i : i + batch_size].to(device)
                batch_logits = self.fc(batch_x)
                pred_logits.append(batch_logits.cpu())
        pred_logits = torch.cat(pred_logits, dim=0)
        pred_logits = pred_logits.to(device)

        with graph.local_graph():
            if self.norm == "sym":
                graph.sym_norm()
            elif self.norm == "row":
                graph.row_norm()
            else:
                raise NotImplementedError
            edge_weight = graph.edge_weight * (1 - self.alpha)

            graph.edge_weight = edge_weight
            predictions = pred_logits
            for _ in range(self.nprop):
                predictions = spmm(graph, predictions) + self.alpha * pred_logits
        return predictions
