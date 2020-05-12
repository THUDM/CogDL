import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .. import BaseModel, register_model


@register_model("sdne")
class SDNE(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off  
        parser.add_argument("--hidden-size1", type=int, default=1000, help="Hidden size in first layer of Auto-Encoder")
        parser.add_argument("--hidden-size2", type=int, default=128, help="Hidden size in second layer of Auto-Encoder")
        parser.add_argument("--droput", type=float, default=0.5, help="Dropout rate")
        parser.add_argument("--alpha", type=float, default=1e-1, help="alhpa is a hyperparameter in SDNE")
        parser.add_argument("--beta", type=float, default=5, help="beta is a hyperparameter in SDNE")
        parser.add_argument("--nu1", type=float, default=1e-4, help="nu1 is a hyperparameter in SDNE")
        parser.add_argument("--nu2", type=float, default=1e-3, help="nu2 is a hyperparameter in SDNE")

        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size1, args.hidden_size2, args.droput, args.alpha, args.beta, args.nu1, args.nu2, args.max_epoch, args.lr)

    def __init__(self, hidden_size1, hidden_size2, droput, alpha, beta, nu1, nu2, max_epoch, lr):
        super(SDNE, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.droput = droput
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.max_epoch = max_epoch
        self.lr = lr
        

    def build_nn(self, ):
        self.encode0 = nn.Linear(self.num_node, self.hidden_size1)
        self.encode1 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.decode0 = nn.Linear(self.hidden_size2, self.hidden_size1)
        self.decode1 = nn.Linear(self.hidden_size1, self.num_node)


    def forward(self, adj_mat, l_mat):
        t0 = F.leaky_relu(self.encode0(adj_mat))
        t0 = F.leaky_relu(self.encode1(t0))
        self.embedding = t0
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        
        # embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)
        # L_1st = torch.sum(adj_mat * (embedding_norm -
        #                              2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
        #                              + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        
        L_1st = 2 * torch.trace(torch.mm(torch.mm(torch.t(self.embedding), l_mat), self.embedding))
        L_2nd = torch.sum(((adj_mat - t0) * adj_mat * self.beta) * ((adj_mat - t0) *  adj_mat * self.beta))
        return self.alpha * L_1st,  L_2nd,  self.alpha * L_1st + L_2nd

    def get_emb(self, adj):
        t0 = self.encode0(adj)
        t0 = self.encode1(t0)
        return t0
   
    def train(self, G):
        self.num_node = G.number_of_nodes()
        self = self.cuda()
        A = torch.from_numpy(nx.adjacency_matrix(G).todense().astype(np.float32))
        L = torch.from_numpy(nx.laplacian_matrix(G).todense().astype(np.float32))
        self.build_nn()
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:            
            opt.zero_grad()
            L_1st, L_2nd, L_all = self.forward(A, L)
            L_reg = 0
            for param in self.parameters():
                L_reg += self.nu1 * torch.sum(torch.abs(param)) + self.nu2 * torch.sum(param * param)
            Loss = L_all + L_reg
            Loss.backward()
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, L_1st: {L_1st:.4f}, L_2nd: {L_2nd:.4f}, L_reg: {L_reg:.4f}"
            )
            opt.step()
        embedding = self.get_emb(A)
        return embedding.detach().numpy()
