import argparse
import torch.nn as nn
import torch.nn.functional as F
from cogdl import experiment
from cogdl.models import BaseModel
from cogdl.layers import GCNLayer, SAGELayer, GATLayer
from cogdl.datasets.ogb import OGBArxivDataset


class GCN(BaseModel):
    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout,
        activation="relu",
        norm="batchnorm",
    ):
        super(GCN, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList(
            [
                GCNLayer(
                    shapes[i],
                    shapes[i + 1],
                    dropout=dropout if i != num_layers - 1 else 0,
                    norm=norm if i != num_layers - 1 else None,
                    activation=activation if i != num_layers - 1 else None,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers):
            h = self.layers[i](graph, h)
        return h


class SAGE(BaseModel):
    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_size,
        num_layers,
        aggr="mean",
        dropout=0.5,
        norm="batchnorm",
        activation="relu",
        normalize=False,
    ):
        super(SAGE, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                SAGELayer(
                    shapes[i],
                    shapes[i + 1],
                    aggr=aggr,
                    normalize=normalize if i != num_layers - 1 else False,
                    dropout=dropout if i != num_layers - 1 else False,
                    norm=norm if i != num_layers - 1 else None,
                    activation=activation if i != num_layers - 1 else None,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        for layer in self.layers:
            x = layer(graph, x)
        return x


class GAT(BaseModel):
    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout=0.5,
        input_drop=0.25,
        attn_drop=0.0,
        alpha=0.2,
        nhead=4,
        residual=True,
        last_nhead=1,
        norm="batchnorm",
    ):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.input_drop = input_drop
        self.attentions = nn.ModuleList()
        self.attentions.append(
            GATLayer(in_feats, hidden_size, nhead=nhead, attn_drop=attn_drop, alpha=alpha, residual=residual, norm=norm)
        )
        for i in range(num_layers - 2):
            self.attentions.append(
                GATLayer(
                    hidden_size * nhead,
                    hidden_size,
                    nhead=nhead,
                    attn_drop=attn_drop,
                    alpha=alpha,
                    residual=residual,
                    norm=norm,
                )
            )
        self.attentions.append(
            GATLayer(
                hidden_size * nhead,
                out_feats,
                attn_drop=attn_drop,
                alpha=alpha,
                nhead=last_nhead,
                residual=False,
            )
        )
        self.num_layers = num_layers
        self.last_nhead = last_nhead
        self.residual = residual

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        x = F.dropout(x, p=self.input_drop, training=self.training)
        for i, layer in enumerate(self.attentions):
            x = layer(graph, x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OGBN-Arxiv (CogDL GNNs)")
    parser.add_argument("--gnn", type=str, default="gcn")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--nhead", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    dataset = OGBArxivDataset()
    if args.gnn == "gcn":
        gnn = GCN(
            in_feats=dataset.num_features,
            hidden_size=args.hidden_size,
            out_feats=dataset.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.gnn == "sage":
        gnn = SAGE(
            in_feats=dataset.num_features,
            hidden_size=args.hidden_size,
            out_feats=dataset.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.gnn == "gat":
        gnn = GAT(
            in_feats=dataset.num_features,
            hidden_size=args.hidden_size,
            out_feats=dataset.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            nhead=args.nhead,
        )

    experiment(
        model=gnn,
        dataset=dataset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=list(range(args.runs)),
    )
