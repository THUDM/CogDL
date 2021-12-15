import argparse
import torch.nn as nn
from cogdl import experiment
from cogdl.models import BaseModel
from cogdl.layers import SAGELayer
from cogdl.datasets.ogb import OGBProductsDataset


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
        x = graph.x
        for layer in self.layers:
            x = layer(graph, x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OGBN-Products (CogDL GNNs)")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-cluster", type=int, default=15000)
    parser.add_argument("--eval-step", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--logger", type=str, default=None)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    dataset = OGBProductsDataset()
    gnn = SAGE(
        in_feats=dataset.num_features,
        hidden_size=args.hidden_size,
        out_feats=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    experiment(
        model=gnn,
        dataset=dataset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=list(range(args.runs)),
        dw="cluster_dw",
        batch_size=args.batch_size,
        n_cluster=args.n_cluster,
        cpu_inference=True,
        eval_step=args.eval_step,
        logger=args.logger,
        patience=args.patience,
    )
