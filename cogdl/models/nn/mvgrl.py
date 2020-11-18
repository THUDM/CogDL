import math

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power, inv
from sklearn.preprocessing import MinMaxScaler
from torch.nn.parameter import Parameter
from tqdm import tqdm

from .. import BaseModel, register_model
from .dgi import GCN, AvgReadout, LogReg


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

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
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

# Borrowed from https://github.com/kavehhassani/mvgrl
class Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h, 'prelu')
        self.gcn2 = GCN(n_in, n_h, 'prelu')
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        return ret, h_1, h_2

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

@register_model("mvgrl")
class MVGRL(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument("--max-epochs", type=int, default=1000)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.max_epochs)

    def __init__(self, nfeat, nhid, nclass, max_epochs):
        super(MVGRL, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(nfeat, nhid).to(self.device)
        self.nhid = nhid
        self.nclass = nclass
        self.epochs = max_epochs
        self.patience = 50

    def train(self, data, dataset_name):
        num_nodes = data.x.shape[0]
        features = preprocess_features(data.x.numpy())

        adj = sp.coo_matrix(
            (np.ones(data.edge_index.shape[1]), data.edge_index),
            (num_nodes, num_nodes),
        )
        adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

        g = nx.Graph()
        g.add_nodes_from(list(range(num_nodes)))
        g.add_edges_from(data.edge_index.numpy().transpose())
        diff = compute_ppr(g, 0.2)

        if dataset_name == 'citeseer':
            epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
            avg_degree = np.sum(adj) / adj.shape[0]
            epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
                                        for e in epsilons])]

            diff[diff < epsilon] = 0.0
            scaler = MinMaxScaler()
            scaler.fit(diff)
            diff = scaler.transform(diff)

        best = 1e9
        best_t = 0
        cnt_wait = 0
        sparse = False
        b_xent = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0)

        ft_size = features.shape[1]
        sample_size = 2000
        batch_size = 4
        lbl_1 = torch.ones(batch_size, sample_size * 2)
        lbl_2 = torch.zeros(batch_size, sample_size * 2)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        lbl = lbl.to(self.device)

        epoch_iter = tqdm(range(self.epochs))
        for epoch in epoch_iter:
            self.model.train()
            optimizer.zero_grad()

            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
            ba, bd, bf = [], [], []
            for i in idx:
                ba.append(adj[i: i + sample_size, i: i + sample_size])
                bd.append(diff[i: i + sample_size, i: i + sample_size])
                bf.append(features[i: i + sample_size])

            ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
            bd = np.array(bd)
            bd = bd.reshape(batch_size, sample_size, sample_size)
            bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

            if sparse:
                ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
                bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
            else:
                ba = torch.FloatTensor(ba)
                bd = torch.FloatTensor(bd)

            bf = torch.FloatTensor(bf)
            idx = np.random.permutation(sample_size)
            shuf_fts = bf[:, idx, :]

            bf = bf.to(self.device)
            ba = ba.to(self.device)
            bd = bd.to(self.device)
            shuf_fts = shuf_fts.to(self.device)

            logits, _, _ = self.model(bf, shuf_fts, ba, bd, sparse, None, None, None)

            loss = b_xent(logits, lbl)

            epoch_iter.set_description(f'Epoch: {epoch:03d}, Loss: {loss.item()}')

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                print('Early stopping!')
                break

            loss.backward()
            optimizer.step()
        
        if sparse:
            adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
            diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

        features = torch.FloatTensor(features[np.newaxis]).to(self.device)
        adj = torch.FloatTensor(adj[np.newaxis]).to(self.device)
        diff = torch.FloatTensor(diff[np.newaxis]).to(self.device)

        embeds, _ = self.model.embed(features, adj, diff, sparse, None)

        idx_train = data.train_mask.to(self.device)
        idx_val = data.val_mask.to(self.device)
        idx_test = data.test_mask.to(self.device)
        labels = data.y.to(self.device)

        train_embs = embeds[0, idx_train]
        val_embs = embeds[0, idx_val]
        test_embs = embeds[0, idx_test]

        train_lbls = labels[idx_train]
        val_lbls = labels[idx_val]
        test_lbls = labels[idx_test]

        tot = 0

        xent = nn.CrossEntropyLoss()
        wd = 0.01 if dataset_name == 'citeseer' else 0.0
        for _ in range(50):
            log = LogReg(self.nhid, self.nclass)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=wd)
            log.to(self.device)

            for _ in range(300):
                log.train()
                opt.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)
                
                loss.backward()
                opt.step()

            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            tot += acc.item()

        return tot / 50
