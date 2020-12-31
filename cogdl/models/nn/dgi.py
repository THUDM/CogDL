import numpy as np
import torch
import torch.nn as nn

from .. import BaseModel, register_model
from cogdl.utils import add_remaining_self_loops, symmetric_normalization, get_activation
from cogdl.trainers.self_supervised_trainer import SelfSupervisedTrainer


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
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if len(seq_fts.shape) > 2:
            if sparse:
                out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
            else:
                out = torch.bmm(adj, seq_fts)
        else:
            if sparse:
                out = torch.spmm(adj, torch.squeeze(seq_fts, 0))
            else:
                out = torch.mm(adj, seq_fts)
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
class DGIModel(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument("--max-epoch", type=int, default=1000)
        parser.add_argument("--activation", type=str, default="prelu")
        parser.add_argument("--patience", type=int, default=20)
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

    def _forward(self, seq1, seq2, adj, sparse, msk):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2)

        return ret

    def forward(self, x, edge_index, edge_attr=None):
        num_nodes = x.shape[0]
        if self.cache is None:
            self.cache = dict()
        if "edge_weight" not in self.cache:
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr)
            edge_weight = symmetric_normalization(x.shape[0], edge_index, edge_weight)
            self.cache["edge_index"] = edge_index
            self.cache["edge_weight"] = edge_weight
        edge_index, edge_weight = self.cache["edge_index"].to(x.device), self.cache["edge_weight"].to(x.device)
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))

        idx = np.random.permutation(num_nodes)
        shuf_fts = x[idx, :]

        logits = self._forward(x, shuf_fts, adj, True, None)
        return logits

    def loss(self, data):
        if self.cache is None:
            num_nodes = data.x.shape[0]
            lbl_1 = torch.ones(1, num_nodes)
            lbl_2 = torch.zeros(1, num_nodes)
            self.cache = {"labels": torch.cat((lbl_1, lbl_2), 1).to(data.x.device)}
        labels = self.cache["labels"].to(data.x.device)

        logits = self.forward(data.x, data.edge_index, data.edge_attr)
        logits = logits.unsqueeze(0)
        loss = self.loss_f(logits, labels)
        return loss

    def node_classification_loss(self, data):
        return self.loss(data)

    # Detach the return variables
    def embed(self, data, msk=None):
        if "edge_weight" in self.cache:
            edge_index, edge_weight = self.cache["edge_index"], self.cache["edge_weight"]
        else:
            edge_index, edge_weight = data.edge_index, data.edge_attr
        num_nodes = data.x.shape[0]
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
        h_1 = self.gcn(data.x, adj, self.sparse)
        # c = self.read(h_1, msk)
        return h_1.detach()  # , c.detach()

    @staticmethod
    def get_trainer(taskType, args):
        return SelfSupervisedTrainer
