import numpy as np
import jittor
from jittor import nn,Module
from cogdl import function as BF
from cogdl.models import BaseModel
from cogdl.utils import get_activation, spmm
from cogdl.datasets.planetoid_data import CoraDataset

# Borrowed from https://github.com/PetarV-/DGI
class GCN(Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else get_activation(act)

        if bias:
            self.bias = BF.FloatTensor(out_ft)
            #self.bias.data.fill_(0.0)
            self.bias[self.bias.bool()]=0.0
        else:
            self.register_parameter("bias", None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            jittor.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias[m.bias.bool()]=0.0

    # Shape of seq: (batch, nodes, features)
    def execute(self, graph, seq, sparse=False):
        seq_fts = self.fc(seq)
        if len(seq_fts.shape) > 2:
            if sparse:
                out = jittor.unsqueeze(spmm(graph, jittor.squeeze(seq_fts, 0)), 0)
            else:
                out = jittor.bmm(graph, seq_fts)
        else:
            if sparse:
                if list(seq_fts.shape)[0]==1:
                    seq_fts = jittor.squeeze(seq_fts, 0)
                out = spmm(graph, seq_fts)
            else:
                out = jittor.matmul(graph, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class DGIModel(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--lr", type=int, default=0.001)
        parser.add_argument("--hidden_size", type=int, default=128)
        parser.add_argument("--activation", type=str, default="prelu")
        parser.add_argument("--patience", type=int, default=20)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.activation)

    def __init__(self, in_feats, hidden_size, activation):
        super(DGIModel, self).__init__()
        self.gcn = GCN(in_feats, hidden_size, activation)
        self.sparse = True

    def execute(self, graph):
        graph.sym_norm()
        x = graph.x
        logits = self.gcn(graph, x, self.sparse)
        return logits

    # Detach the return variables
    def embed(self, data):
        h_1 = self.gcn(data, data.x, self.sparse)
        return h_1


