import torch.nn as nn
import torch.nn.functional as F
from cogdl.trainers.sampled_trainer import RandomClusterTrainer
from cogdl.utils import get_activation
from cogdl.layers import ResGNNLayer, GENConv

from .. import BaseModel, register_model


@register_model("deepergcn")
class DeeperGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--num-layers", type=int, default=14)
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--aggr", type=str, default="softmax_sg")
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--p", type=float, default=1.0)
        parser.add_argument("--learn-beta", action="store_true")
        parser.add_argument("--learn-p", action="store_true")
        parser.add_argument("--learn-msg-scale", action="store_true")
        parser.add_argument("--use-msg-norm", action="store_true")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_feat=args.num_features,
            hidden_size=args.hidden_size,
            out_feat=args.num_classes,
            num_layers=args.num_layers,
            activation=args.activation,
            dropout=args.dropout,
            aggr=args.aggr,
            beta=args.beta,
            p=args.p,
            learn_beta=args.learn_beta,
            learn_p=args.learn_p,
            learn_msg_scale=args.learn_msg_scale,
            use_msg_norm=args.use_msg_norm,
            edge_attr_size=args.edge_attr_size,
        )

    def __init__(
        self,
        in_feat,
        hidden_size,
        out_feat,
        num_layers,
        activation="relu",
        dropout=0.0,
        aggr="max",
        beta=1.0,
        p=1.0,
        learn_beta=False,
        learn_p=False,
        learn_msg_scale=True,
        use_msg_norm=False,
        edge_attr_size=None,
    ):
        super(DeeperGCN, self).__init__()
        self.dropout = dropout
        self.feat_encoder = nn.Linear(in_feat, hidden_size)

        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(
                ResGNNLayer(
                    conv=GENConv(
                        in_feats=hidden_size,
                        out_feats=hidden_size,
                        aggr=aggr,
                        beta=beta,
                        p=p,
                        learn_beta=learn_beta,
                        learn_p=learn_p,
                        use_msg_norm=use_msg_norm,
                        learn_msg_scale=learn_msg_scale,
                        edge_attr_size=edge_attr_size,
                    ),
                    in_channels=hidden_size,
                    activation=activation,
                    dropout=dropout,
                    checkpoint_grad=False,
                )
            )
        self.norm = nn.BatchNorm1d(hidden_size, affine=True)
        self.activation = get_activation(activation)
        self.fc = nn.Linear(hidden_size, out_feat)

    def forward(self, graph):
        x = graph.x
        h = self.feat_encoder(x)
        for layer in self.layers:
            h = layer(graph, h)
        h = self.activation(self.norm(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc(h)
        return h

    def predict(self, graph):
        return self.forward(graph)
