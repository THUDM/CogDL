import numpy as np
import torch
import torch.nn as nn

from .. import BaseModel, register_model
from cogdl.utils import get_activation, spmm
from cogdl.trainers.self_supervised_trainer import SelfSupervisedPretrainer
from cogdl.models.self_supervised_model import SelfSupervisedContrastiveModel


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


# Borrowed from https://github.com/PetarV-/DGI
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        dim = len(seq.shape) - 2
        if msk is None:
            return torch.mean(seq, dim)
        else:
            return torch.sum(seq * msk, dim) / torch.sum(msk)


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2))

        return logits


@register_model("dgi")
class DGIModel(SelfSupervisedContrastiveModel):
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
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_size)

        self.loss_f = nn.BCEWithLogitsLoss()
        self.cache = None
        self.sparse = True

    def _forward(self, graph, seq1, seq2, sparse, msk):
        h_1 = self.gcn(graph, seq1, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(graph, seq2, sparse)

        ret = self.disc(c, h_1, h_2)

        return ret

    def augment(self, graph):
        idx = np.random.permutation(graph.num_nodes)
        augmented_graph = graph.clone()
        augmented_graph.x = augmented_graph.x[idx, :]
        return augmented_graph

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        shuf_fts = self.augment(graph).x

        logits = self._forward(graph, x, shuf_fts, True, None)
        return logits

    def loss(self, data):
        if self.cache is None:
            num_nodes = data.num_nodes
            lbl_1 = torch.ones(1, num_nodes)
            lbl_2 = torch.zeros(1, num_nodes)
            self.cache = {"labels": torch.cat((lbl_1, lbl_2), 1).to(data.x.device)}
        labels = self.cache["labels"].to(data.x.device)

        logits = self.forward(data)
        logits = logits.unsqueeze(0)
        loss = self.loss_f(logits, labels)
        return loss

    def self_supervised_loss(self, data):
        return self.loss(data)

    # Detach the return variables
    def embed(self, data, msk=None):
        h_1 = self.gcn(data, data.x, self.sparse)
        # c = self.read(h_1, msk)
        return h_1.detach()  # , c.detach()

    @staticmethod
    def get_trainer(args):
        return SelfSupervisedPretrainer
