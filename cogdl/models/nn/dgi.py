import numpy as np
import torch
import torch.nn as nn

from .. import BaseModel
from cogdl.utils import get_activation, spmm


# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else get_activation(act)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, graph, seq, sparse=False):
        seq_fts = self.fc(seq)
        if len(seq_fts.shape) > 2:
            if sparse:
                out = torch.unsqueeze(spmm(graph, torch.squeeze(seq_fts, 0)), 0)
            else:
                out = torch.bmm(graph, seq_fts)
        else:
            if sparse:
                out = spmm(graph, torch.squeeze(seq_fts, 0))
            else:
                out = torch.mm(graph, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class DGIModel(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument("--activation", type=str, default="prelu")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.activation)

    def __init__(self, in_feats, hidden_size, activation):
        super(DGIModel, self).__init__()
        self.gcn = GCN(in_feats, hidden_size, activation)
        self.sparse = True

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        logits = self.gcn(graph, x, self.sparse)
        return logits

    # Detach the return variables
    def embed(self, data):
        h_1 = self.gcn(data, data.x, self.sparse)
        return h_1.detach()
