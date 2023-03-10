import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .. import BaseModel
from .dgi import GCN
from cogdl.utils.ppr_utils import build_topk_ppr_matrix_from_data
from cogdl.data import Graph


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def compute_ppr(adj, index, alpha=0.4, epsilon=1e-4, k=8, norm="row"):
    return build_topk_ppr_matrix_from_data(adj, alpha, epsilon, index, k, norm).tocsr()


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


# Borrowed from https://github.com/kavehhassani/mvgrl
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

    def forward(self, c1, c2, h1, h2, h3, h4):
        c_x1 = torch.unsqueeze(c1, 0)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 0)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 1)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 1)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 1)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 1)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 0)
        return logits


# Mainly borrowed from https://github.com/kavehhassani/mvgrl
class MVGRL(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument("--sample-size", type=int, default=2000)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--alpha", type=float, default=0.2)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.sample_size, args.batch_size, args.alpha, args.dataset)

    def __init__(self, in_feats, hidden_size, sample_size=2000, batch_size=4, alpha=0.2, dataset="cora"):
        super(MVGRL, self).__init__()
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.sparse = True
        self.dataset_name = dataset

        self.gcn1 = GCN(in_feats, hidden_size, "prelu")
        self.gcn2 = GCN(in_feats, hidden_size, "prelu")
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(hidden_size)
        self.loss_f = nn.BCEWithLogitsLoss()

        self.cache = None

    def _forward(self, adj, diff, seq1, seq2, msk):
        out_shape = list(seq1.shape[:-1]) + [self.hidden_size]

        seq1 = seq1.view(-1, seq1.shape[-1])
        seq2 = seq2.view(-1, seq2.shape[-1])

        h_1 = self.gcn1(adj, seq1, True)
        h_1 = h_1.view(out_shape)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(diff, seq1, True)
        h_2 = h_2.view(out_shape)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(adj, seq2, True)
        h_4 = self.gcn2(diff, seq2, True)
        h_3 = h_3.view(out_shape)
        h_4 = h_4.view(out_shape)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)

        return ret, h_1, h_2

    def augment(self, graph):
        num_nodes = graph.num_nodes
        adj = sp.coo_matrix(
            (graph.edge_weight.cpu().numpy(), (graph.edge_index[0].cpu().numpy(), graph.edge_index[1].cpu().numpy())),
            shape=(graph.num_nodes, graph.num_nodes),
        )
        diff = compute_ppr(adj.tocsr(), np.arange(num_nodes), self.alpha).tocoo()
        return adj, diff

    def preprocess(self, graph):
        print("MVGRL preprocessing...")
        graph.add_remaining_self_loops()
        graph.sym_norm()

        adj, diff = self.augment(graph)

        if self.cache is None:
            self.cache = dict()
        graphs = []
        for g in [adj, diff]:
            row = torch.from_numpy(g.row).long()
            col = torch.from_numpy(g.col).long()
            val = torch.from_numpy(g.data).float()
            edge_index = torch.stack([row, col])
            graphs.append(Graph(edge_index=edge_index, edge_weight=val))

        self.cache["diff"] = graphs[1]
        self.cache["adj"] = graphs[0]
        print("Preprocessing Done...")

    def forward(self, graph):
        if not self.training:
            return self.embed(graph)

        x = graph.x
        if self.cache is None or "diff" not in self.cache:
            self.preprocess(graph)
        diff, adj = self.cache["diff"], self.cache["adj"]

        self.sample_size = min(self.sample_size, graph.num_nodes - self.batch_size)
        idx = np.random.randint(0, graph.num_nodes - self.sample_size + 1, self.batch_size)
        logits = []
        for i in idx:
            ba = adj.subgraph(list(range(i, i + self.sample_size))).to(self.device)
            bd = diff.subgraph(list(range(i, i + self.sample_size))).to(self.device)
            bf = x[i : i + self.sample_size].to(self.device)
            idx = np.random.permutation(self.sample_size)
            shuf_fts = bf[idx, :].to(self.device)
            logit, _, _ = self._forward(ba, bd, bf, shuf_fts, None)
            logits.append(logit)

        return torch.stack(logits)

    def loss(self, data):
        if self.sample_size > data.num_nodes:
            self.sample_size = data.num_nodes
        if self.cache is None:
            self.device = next(self.gcn1.parameters()).device
            lbl_1 = torch.ones(self.batch_size, self.sample_size * 2)
            lbl_2 = torch.zeros(self.batch_size, self.sample_size * 2)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            lbl = lbl.to(self.device)
            self.cache = {"labels": lbl}
        lbl = self.cache["labels"]
        logits = self.forward(data)
        loss = self.loss_f(logits, lbl)
        return loss

    def embed(self, data, msk=None):
        adj = self.cache["adj"].to(self.device)
        diff = self.cache["diff"].to(self.device)
        h_1 = self.gcn1(adj, data.x.to(self.device), True)
        h_2 = self.gcn2(diff, data.x.to(self.device), True)
        # c = self.read(h_1, msk)
        return (h_1 + h_2).detach()  # , c.detach()
