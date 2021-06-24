import torch.nn as nn

from cogdl.utils import accuracy
from cogdl.layers import HANLayer

from .. import BaseModel, register_model


@register_model("han")
class HAN(BaseModel):
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
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_edge,
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_nodes,
            args.num_layers,
        )

    def __init__(self, num_edge, w_in, w_out, num_class, num_nodes, num_layers):
        super(HAN, self).__init__()
        self.num_edge = num_edge
        self.num_nodes = num_nodes
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(HANLayer(num_edge, w_in, w_out))
            else:
                layers.append(HANLayer(num_edge, w_out, w_out))

        self.layers = nn.ModuleList(layers)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.linear = nn.Linear(self.w_out, self.num_class)

    def forward(self, graph, target_x, target):
        X = graph.x
        for i in range(self.num_layers):
            X = self.layers[i](graph, X)

        y = self.linear(X[target_x])
        loss = self.cross_entropy_loss(y, target)
        return loss, y

    def loss(self, data):
        loss, y = self.forward(data, data.train_node, data.train_target)
        return loss

    def evaluate(self, data, nodes, targets):
        loss, y = self.forward(data, nodes, targets)
        f1 = accuracy(y, targets)
        return loss.item(), f1
