import torch
import torch.nn as nn
from .. import BaseModel, register_model


class PairNorm(nn.Module):
    def __init__(self, mode="PN", scale=1):
        """
        mode:
          'None' : No normalization
          'PN'   : Original version
          'PN-SI'  : Scale-Individually version
          'PN-SCS' : Scale-and-Center-Simultaneously version

        ('SCS'-mode is not in the paper but we found it works well in practice,
          especially for GCN and GAT.)
        PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ["None", "PN", "PN-SI", "PN-SCS"]
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == "None":
            return x

        col_mean = x.mean(dim=0)
        if self.mode == "PN":
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == "PN-SI":
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == "PN-SCS":
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


@register_model("sgcpn")
class SGC(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate.")
        parser.add_argument("--num-layers", type=int, default=40, help="Number of layers.")
        parser.add_argument("--norm-mode", type=str, default="PN", help="Mode for PairNorm, {None, PN, PN-SI, PN-SCS}.")
        parser.add_argument("--norm-scale", type=float, default=10, help="Row-normalization scale.")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.dropout,
            args.num_layers,
            args.norm_mode,
            args.norm_scale,
        )

    def __init__(self, nfeat, nclass, dropout, nlayer=2, norm_mode="None", norm_scale=10):
        super(SGC, self).__init__()
        self.linear = torch.nn.Linear(nfeat, nclass)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.dropout = nn.Dropout(p=dropout)
        self.nlayer = nlayer

    def forward(self, x, adj):
        x = self.norm(x)
        for _ in range(self.nlayer):
            x = adj.mm(x)
            x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def node_classification_loss(self, data):
        output = self.forward(data.x, data.adj)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        return loss

    def predict(self, data):
        return self.forward(data.x, data.adj)
