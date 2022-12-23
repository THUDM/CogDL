import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from cogdl.layers import GCNLayer, GATLayer
from cogdl.models import BaseModel
import cogdl.utils.grb_utils as utils
import types


class GCNGuard(BaseModel):
    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser."""
    #     # fmt: off
    #     parser.add_argument("--num-features", type=int)
    #     parser.add_argument("--num-classes", type=int)
    #     parser.add_argument("--num-layers", type=int, default=2)
    #     parser.add_argument("--hidden-size", type=int, default=64)
    #     parser.add_argument("--dropout", type=float, default=0.5)
    #     parser.add_argument("--norm", type=str, default=None)
    #     parser.add_argument("--feat-norm", type=types.FunctionType, default=None)
    #     parser.add_argument("--adj-norm", type=types.FunctionType, default=utils.GCNAdjNorm)
    #     parser.add_argument("--activation", type=str, default="relu")
    #     parser.add_argument("--attention", type=bool, default=True)
    #     parser.add_argument("--drop", type=bool, default=False)
    #     # fmt: on

    # @classmethod
    # def build_model_from_args(cls, args):
    #     return cls(
    #         args.num_features,
    #         args.hidden_size,
    #         args.num_classes,
    #         args.num_layers,
    #         args.dropout,
    #         args.activation,
    #         args.norm,
    #         args.feat_norm,
    #         args.adj_norm,
    #         args.attention,
    #         args.drop,
    #     )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout=0.0,
        activation=F.relu,
        norm=None,
        feat_norm=None,
        adj_norm_func=utils.GCNAdjNorm,
        attention=True,
        drop=False,
    ):
        super(GCNGuard, self).__init__()
        self.in_features = in_feats
        self.out_features = out_feats
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_size) is int:
            hidden_size = [hidden_size] * (num_layers - 1)
        elif type(hidden_size) is list or type(hidden_size) is tuple:
            assert len(hidden_size) == (num_layers - 1), "Incompatible sizes between hidden_size and n_layers."
        n_features = [in_feats] + hidden_size + [out_feats]

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if norm == "layernorm" and i == 0:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(
                GCNLayer(
                    in_features=n_features[i],
                    out_features=n_features[i + 1],
                    activation=activation if i != num_layers - 1 else None,
                    dropout=dropout if i != num_layers - 1 else 0.0,
                    norm=norm if i != num_layers - 1 else None,
                )
            )
        self.reset_parameters()
        self.drop = drop
        self.drop_learn = torch.nn.Linear(2, 1)
        self.attention = attention

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph):
        adj, h = utils.getGRBGraph(graph)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                h = layer(h)
            else:
                if self.attention:
                    adj = self.att_coef(h, adj)
                h = layer(graph, h)

        return h

    def att_coef(self, features, adj):
        edge_index = adj._indices()

        n_node = features.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        features_copy = features.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=features_copy, Y=features_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm="l1")

        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1, att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T).to(features.device)
            drop_score = self.drop_learn(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)  # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)  # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32).to(features.device)
        att_adj = torch.tensor(att_adj, dtype=torch.int64).to(features.device)

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)

        return new_adj


class GATGuard(nn.Module):
    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser."""
    #     # fmt: off
    #     parser.add_argument("--num-features", type=int)
    #     parser.add_argument("--num-classes", type=int)
    #     parser.add_argument("--num-layers", type=int, default=2)
    #     parser.add_argument("--nhead", type=int, default=8)
    #     parser.add_argument("--hidden-size", type=int, default=64)
    #     parser.add_argument("--dropout", type=float, default=0.5)
    #     parser.add_argument("--norm", type=str, default=None)
    #     parser.add_argument("--feat-norm", type=types.FunctionType, default=None)
    #     parser.add_argument("--adj-norm", type=types.FunctionType, default=utils.GCNAdjNorm)
    #     parser.add_argument("--activation", type=str, default="relu")
    #     parser.add_argument("--attention", type=bool, default=True)
    #     parser.add_argument("--drop", type=bool, default=False)
    #     # fmt: on

    # @classmethod
    # def build_model_from_args(cls, args):
    #     return cls(
    #         args.num_features,
    #         args.hidden_size,
    #         args.num_classes,
    #         args.num_layers,
    #         args.nhead,
    #         args.dropout,
    #         args.activation,
    #         args.norm,
    #         args.feat_norm,
    #         args.adj_norm,
    #         args.attention,
    #         args.drop,
    #     )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        num_heads,
        dropout=0.0,
        activation=F.leaky_relu,
        norm=None,
        feat_norm=None,
        adj_norm_func=utils.GCNAdjNorm,
        attention=True,
        drop=False,
    ):

        super(GATGuard, self).__init__()
        self.in_features = in_feats
        self.out_features = out_feats
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_size) is int:
            hidden_size = [hidden_size] * (num_layers - 1)
        elif type(hidden_size) is list or type(hidden_size) is tuple:
            assert len(hidden_size) == (num_layers - 1), "Incompatible sizes between hidden_size and n_layers."
        n_features = [in_feats] + hidden_size + [out_feats]

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if norm == "layernorm" and i == 0:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(
                GATLayer(
                    in_feats=n_features[i] * num_heads if i != 0 else n_features[i],
                    out_feats=n_features[i + 1],
                    nhead=num_heads if i != num_layers - 1 else 1,
                    activation=activation if i != num_layers - 1 else None,
                    norm=norm if i != num_layers - 1 else None,
                )
            )
        self.drop = drop
        self.drop_learn = torch.nn.Linear(2, 1)
        self.attention = attention
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, graph):
        adj, x = utils.getGRBGraph(graph)

        for i, layer in enumerate(self.layers):
            if self.attention:
                adj = self.att_coef(x, adj)
                graph = utils.getGraph(adj, x, device=graph.device)
            x = layer(graph, x).flatten(1)
            if i != len(self.layers) - 1:
                if self.dropout is not None:
                    x = self.dropout(x)

        return x

    def att_coef(self, features, adj):
        if type(adj) == torch.Tensor:
            adj = sp.coo_matrix(adj.to_dense().detach().cpu().numpy())
        else:
            adj = adj.tocoo()
        n_node = features.shape[0]
        row, col = adj.row, adj.col

        features_copy = features.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=features_copy, Y=features_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm="l1")

        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1, att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T).to(features.device)
            drop_score = self.drop_learn(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)  # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)
        att_edge_weight = np.asarray(att_edge_weight.ravel())[0]
        new_adj = sp.csr_matrix((att_edge_weight, (row, col)))

        return new_adj
