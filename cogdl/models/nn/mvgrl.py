import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.linalg import fractional_matrix_power, inv
from sklearn.preprocessing import MinMaxScaler

from .. import BaseModel, register_model
from .dgi import GCN, AvgReadout
from cogdl.utils import add_remaining_self_loops, symmetric_normalization
from cogdl.trainers.self_supervised_trainer import SelfSupervisedTrainer


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


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
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


# Mainly borrowed from https://github.com/kavehhassani/mvgrl
@register_model("mvgrl")
class MVGRL(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument("--max-epochs", type=int, default=1000)
        parser.add_argument("--sample-size", type=int, default=2000)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--sparse", action="store_true")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.sample_size, args.batch_size, args.sparse, args.dataset)

    def __init__(self, in_feats, hidden_size, sample_size=2000, batch_size=4, sparse=False, dataset="cora"):
        super(MVGRL, self).__init__()
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.sparse = sparse
        self.dataset_name = dataset

        self.gcn1 = GCN(in_feats, hidden_size, "prelu")
        self.gcn2 = GCN(in_feats, hidden_size, "prelu")
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(hidden_size)
        self.loss_f = nn.BCEWithLogitsLoss()

        self.cache = None

    def _forward(self, seq1, seq2, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)

        return ret, h_1, h_2

    def preprocess(self, x, edge_index, edge_attr=None):
        num_nodes = x.shape[0]

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr)

        adj = edge_index.cpu().numpy()
        edge_weight = symmetric_normalization(x.shape[0], edge_index, edge_weight)
        adj = sp.coo_matrix((edge_weight.cpu().numpy(), (adj[0], adj[1])), shape=(num_nodes, num_nodes)).todense()

        g = nx.Graph()
        g.add_nodes_from(list(range(num_nodes)))
        g.add_edges_from(edge_index.cpu().numpy().transpose())
        diff = compute_ppr(g, 0.2)

        if self.dataset_name == "citeseer":
            epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
            avg_degree = np.sum(adj) / adj.shape[0]
            epsilon = epsilons[
                np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0]) for e in epsilons])
            ]

            diff[diff < epsilon] = 0.0
            scaler = MinMaxScaler()
            scaler.fit(diff)
            diff = scaler.transform(diff)

        if self.cache is None:
            self.cache = dict()
        self.cache["diff"] = diff
        self.cache["adj"] = adj
        self.device = next(self.gcn1.parameters()).device

    def forward(self, x, edge_index, edge_attr=None):
        if self.cache is None or "diff" not in self.cache:
            self.preprocess(x, edge_index, edge_attr)
        diff, adj = self.cache["diff"], self.cache["adj"]

        idx = np.random.randint(0, adj.shape[-1] - self.sample_size + 1, self.batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i : i + self.sample_size, i : i + self.sample_size])
            bd.append(diff[i : i + self.sample_size, i : i + self.sample_size])
            bf.append(x[i : i + self.sample_size])

        ba = np.array(ba).reshape(self.batch_size, self.sample_size, self.sample_size)
        bd = np.array(bd)
        bd = bd.reshape(self.batch_size, self.sample_size, self.sample_size)
        bf = torch.stack(bf).reshape(self.batch_size, self.sample_size, x.shape[1])

        if self.sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba)).to(self.device)
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd)).to(self.device)
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = bf.to(self.device)
        idx = np.random.permutation(self.sample_size)
        shuf_fts = bf[:, idx, :]

        bf = bf.to(self.device)
        ba = ba.to(self.device)
        bd = bd.to(self.device)
        shuf_fts = shuf_fts.to(self.device)

        logits, _, _ = self._forward(bf, shuf_fts, ba, bd, self.sparse, None)
        return logits

    def loss(self, data):
        if self.cache is None:
            self.device = next(self.gcn1.parameters()).device
            lbl_1 = torch.ones(self.batch_size, self.sample_size * 2)
            lbl_2 = torch.zeros(self.batch_size, self.sample_size * 2)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            lbl = lbl.to(self.device)
            self.cache = {"labels": lbl}
        lbl = self.cache["labels"]
        logits = self.forward(data.x, data.edge_index, data.edge_attr)
        loss = self.loss_f(logits, lbl)
        return loss

    def node_classification_loss(self, data):
        return self.loss(data)

    def embed(self, data, msk=None):
        adj = torch.from_numpy(self.cache["adj"]).float().to(data.x.device)
        diff = torch.from_numpy(self.cache["diff"]).float().to(data.x.device)
        h_1 = self.gcn1(data.x, adj, self.sparse)
        h_2 = self.gcn2(data.x, diff, self.sparse)
        # c = self.read(h_1, msk)
        return (h_1 + h_2).detach()  # , c.detach()

    @staticmethod
    def get_trainer(taskType, args):
        return SelfSupervisedTrainer
