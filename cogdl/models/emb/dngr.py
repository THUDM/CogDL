import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .. import BaseModel, register_model


class DNGR_layer(nn.Module):
    def __init__(self, num_node, hidden_size1, hidden_size2):
        super(DNGR_layer, self).__init__()
        self.num_node = num_node
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.encoder = nn.Sequential(
            nn.Linear(self.num_node, self.hidden_size1),
            nn.Tanh(),
            nn.Linear(self.hidden_size1, self.hidden_size2),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size2, self.hidden_size1),
            nn.Tanh(),
            nn.Linear(self.hidden_size1, self.num_node),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded    


@register_model("dngr")
class DNGR(BaseModel):
    r"""The DNGR model from the `"Deep Neural Networks for Learning Graph Representations"
    <https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12423/11715>`_ paper
    
    Args:
        hidden_size1 (int) : The size of the first hidden layer.
        hidden_size2 (int) : The size of the second hidden layer.
        noise (float) : Denoise rate of DAE.
        alpha (float) : Parameter in DNGR.
        step (int) : The max step in random surfing.
        max_epoch (int) : The max epoches in training step.
        lr (float) : Learning rate in DNGR.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size1", type=int, default=1000, help="Hidden size in first layer of Auto-Encoder")
        parser.add_argument("--hidden-size2", type=int, default=128, help="Hidden size in second layer of Auto-Encoder")
        parser.add_argument("--noise", type=float, default=0.2, help="denoise rate of DAE")
        parser.add_argument("--alpha", type=float, default=0.98, help="alhpa is a hyperparameter in DNGR")
        parser.add_argument("--step", type=int, default=10, help="step is a hyperparameter in DNGR")

        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size1, args.hidden_size2, args.noise, args.alpha, args.step, args.max_epoch, args.lr, args.cpu)

    def __init__(self, hidden_size1, hidden_size2, noise, alpha, step, max_epoch, lr, cpu):
        super(DNGR, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.noise = noise
        self.alpha = alpha
        self.step = step
        self.max_epoch = max_epoch
        self.lr = lr
        self.cpu = cpu
        self.device = torch.device('cpu' if self.cpu else 'cuda')
   
    def scale_matrix(self, mat):
        mat = mat - np.diag(np.diag(mat))
        D_inv = np.diagflat(np.reciprocal(np.sum(mat, axis=0)))
        mat = np.dot(D_inv, mat)
        return mat
 
    def random_surfing(self, adj_matrix):
        # Random Surfing
	    adj_matrix = self.scale_matrix(adj_matrix)
	    P0 = np.eye(self.num_node, dtype='float32')
	    M = np.zeros((self.num_node, self.num_node), dtype='float32')
	    P = np.eye(self.num_node, dtype='float32')
	    for i in range(0, self.step):
		    P = self.alpha * np.dot(P, adj_matrix) + (1 - self.alpha) * P0
		    M = M + P
	    return M
   
    def get_ppmi_matrix(self, mat):
        # Get Positive Pairwise Mutual Information(PPMI) matrix
        mat = self.random_surfing(mat)
        M = self.scale_matrix(mat)
        col_s = np.sum(M, axis=0).reshape(1, self.num_node)
        row_s = np.sum(M, axis=1).reshape(self.num_node, 1)
        D = np.sum(col_s)
        rowcol_s = np.dot(row_s, col_s)
        PPMI = np.log(np.divide(D * M, rowcol_s))
        
        PPMI[np.isnan(PPMI)] = 0.0
        PPMI[np.isinf(PPMI)] = 0.0
        PPMI[np.isneginf(PPMI)] = 0.0
        PPMI[PPMI < 0] = 0.0
        return PPMI
    
    def get_denoised_matrix(self, mat):
        return mat * (np.random.random(mat.shape) > self.noise)
   
    def get_emb(self, matrix):
        ut, s, _ = sp.linalg.svds(matrix, self.hidden_size2)
        emb_matrix = ut * np.sqrt(s)
        emb_matrix = preprocessing.normalize(emb_matrix, "l2")
        return emb_matrix
   
    def train(self, G):
        self.num_node = G.number_of_nodes()
        A = nx.adjacency_matrix(G).todense()
        PPMI = self.get_ppmi_matrix(A)
        print("PPMI matrix compute done")
        # return self.get_emb(PPMI)
        
        input_mat = torch.from_numpy(self.get_denoised_matrix(PPMI).astype(np.float32))
        model = DNGR_layer(self.num_node, self.hidden_size1, self.hidden_size2)
        
        input_mat = input_mat.to(self.device)
        model = model.to(self.device)
        
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_func = nn.MSELoss()

        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:            
            opt.zero_grad()
            encoded, decoded = model.forward(input_mat)
            Loss = loss_func(decoded, input_mat)
            Loss.backward()
            epoch_iter.set_description(
                f"Epoch: {epoch:03d},  Loss: {Loss:.8f}"
            )
            opt.step()
        embedding, _ = model.forward(input_mat)
        return embedding.detach().cpu().numpy()
