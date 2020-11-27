import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import BaseModel, register_model
from cogdl.utils import add_remaining_self_loops, spmm, spmm_adj

from .gcn import TKipfGCN

@register_model('ppnp')
class PPNP(BaseModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--propagation-type", type=str, default='appnp')
        parser.add_argument("--alpha", type=float, default=0.1)
        parser.add_argument("--num-iterations", type=int, default=10) # only for appnp
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, 
                    args.dropout, args.propagation_type, args.alpha, args.num_iterations)

    def __init__(self, nfeat, nhid, nclass, dropout, propagation, alpha, niter):
        super(PPNP, self).__init__()

        # GCN as a prediction and then apply the personalized page rank on the results
        self.nn = TKipfGCN(nfeat, nhid, nclass, dropout)
        
        if propagation not in ('appnp', 'ppnp'):
            print('Invalid propagation type, using default appnp')
            propagation = 'appnp'
        
        self.propagation = propagation
        self.alpha = alpha
        self.niter = niter
        self.dropout = dropout

        self.vals = None # speedup for ppnp
        

    def _calculate_A_hat(self,x,adj):
        device = x.device
        adj_values = torch.ones(adj.shape[1]).to(device)
        adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, x.shape[0])
        deg = spmm(adj, adj_values, torch.ones(x.shape[0], 1).to(device)).squeeze()
        deg_sqrt = deg.pow(-1/2)
        adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
        return adj, adj_values

    def forward(self, x, adj):
        def get_ready_format(input, edge_index, edge_attr=None):
            if edge_attr is None:
                edge_attr = torch.ones(edge_index.shape[1]).float().to(input.device)
            adj = torch.sparse_coo_tensor(
                edge_index,
                edge_attr,
                (input.shape[0], input.shape[0]),
            ).to(input.device)
            return adj
        
        
        #get prediction
        local_preds = self.nn.forward(x, adj)    
        
        adj, A_hat = self._calculate_A_hat(x,adj)
        #apply personalized pagerank
        if self.propagation == 'ppnp':
            if self.vals is None:
                self.vals = self.alpha * torch.inverse(torch.eye(x.shape[0]).to(x.device) - (1-self.alpha)* get_ready_format(x, adj, A_hat))
            final_preds = F.dropout(self.vals) @ local_preds
        else: #appnp
            preds = local_preds
            for i in range(1,self.niter+1):
                A_dropped = get_ready_format(x, adj, F.dropout(A_hat, self.dropout, training=self.training))
                preds = torch.spmm((1-self.alpha) * A_dropped,preds) + self.alpha * local_preds
            final_preds = preds
        
        return F.log_softmax(final_preds, dim=-1)
            


    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data.x, data.edge_index)

        