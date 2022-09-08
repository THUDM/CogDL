import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import SELayer
from .. import BaseModel

from cogdl.layers import MLP, GATLayer, GINLayer
from cogdl.utils import batch_sum_pooling, batch_mean_pooling, batch_max_pooling
from cogdl.layers import Set2Set


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp, use_selayer):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = (
            SELayer(self.mlp.output_dim, int(np.sqrt(self.mlp.output_dim)))
            if use_selayer
            else nn.BatchNorm1d(self.mlp.output_dim)
        )

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_layers, nhead, dropout=0.0, attn_drop=0.0, alpha=0.2, residual=False):
        super(GATModel, self).__init__()
        assert hidden_size % nhead == 0
        self.layers = nn.ModuleList(
            [
                GATLayer(
                    in_feats=in_feats if i > 0 else hidden_size // nhead,
                    out_feats=hidden_size // nhead,
                    nhead=nhead,
                    attn_drop=0.0,
                    alpha=0.2,
                    residual=False,
                    activation=F.leaky_relu if i + 1 < num_layers else None,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, graph, x):
        for i, layer in enumerate(self.layers):
            x = layer(graph, x)
        return x


class GINModel(nn.Module):
    def __init__(
        self,
        num_layers,
        in_feats,
        hidden_dim,
        out_feats,
        num_mlp_layers,
        eps=0,
        pooling="sum",
        train_eps=False,
        dropout=0.5,
        final_dropout=0.2,
        use_selayer=False,
    ):
        super(GINModel, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            if i == 0:
                mlp = MLP(in_feats, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            self.gin_layers.append(GINLayer(ApplyNodeFunc(mlp, use_selayer), eps, train_eps))
            self.batch_norm.append(nn.BatchNorm1d(hidden_dim))

        self.linear_prediction = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.linear_prediction.append(nn.Linear(in_feats, out_feats))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, out_feats))
        self.dropout = nn.Dropout(dropout)

        if pooling == "sum":
            self.pool = batch_sum_pooling
        elif pooling == "mean":
            self.pool = batch_mean_pooling
        elif pooling == "max":
            self.pool = batch_max_pooling
        else:
            raise NotImplementedError
        self.final_drop = nn.Dropout(final_dropout)

    def forward(self, batch, n_feat):
        h = n_feat
        # device = h.device
        # batchsize = int(torch.max(batch.batch)) + 1

        layer_rep = [h]
        for i in range(self.num_layers - 1):
            h = self.gin_layers[i](batch, h)
            h = self.batch_norm[i](h)
            h = F.relu(h)
            layer_rep.append(h)

        score_over_layer = 0

        all_outputs = []
        for i, h in enumerate(layer_rep):
            pooled_h = self.pool(h, batch.batch)
            all_outputs.append(pooled_h)
            score_over_layer += self.final_drop(self.linear_prediction[i](pooled_h))

        return score_over_layer, all_outputs[1:]


class GCCModel(BaseModel):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__
    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 15.
    edge_input_dim : int
        Dimension of input edge feature, default to be 15.
    output_dim : int
        Dimension of prediction, default to be 12.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 64.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    num_step_set2set : int
        Number of set2set steps
    num_layer_set2set : int
        Number of set2set layers
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--positional-embedding-size", type=int, default=32)
        parser.add_argument("--degree-embedding-size", type=int, default=16)
        parser.add_argument("--max-node-freq", type=int, default=16)
        parser.add_argument("--max-edge-freq", type=int, default=16)
        parser.add_argument("--max-degree", type=int, default=512)
        parser.add_argument("--freq-embedding-size", type=int, default=16)
        parser.add_argument("--num-layers", type=int, default=5)
        parser.add_argument("--num-heads", type=int, default=2)
        parser.add_argument("--output-size", type=int, default=64)
        parser.add_argument("--norm", type=bool, default=True)
        parser.add_argument("--gnn-model", type=str, default="gin")
        parser.add_argument("--degree-input", type=bool, default=True)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            positional_embedding_size=args.positional_embedding_size,
            max_node_freq=args.max_node_freq,
            max_edge_freq=args.max_edge_freq,
            max_degree=args.max_degree,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            degree_embedding_size=args.degree_embedding_size,
            node_hidden_dim=args.hidden_size,
            norm=args.norm,
            gnn_model=args.gnn_model,
            output_dim=args.output_size,
            degree_input=args.degree_input
        )

    def __init__(
        self,
        positional_embedding_size=32,
        max_node_freq=8,
        max_edge_freq=8,
        max_degree=128,
        freq_embedding_size=32,
        degree_embedding_size=32,
        output_dim=32,
        node_hidden_dim=32,
        edge_hidden_dim=32,
        num_layers=6,
        num_heads=4,
        num_step_set2set=6,
        num_layer_set2set=3,
        norm=False,
        gnn_model="gin",
        degree_input=True,
    ):
        super(GCCModel, self).__init__()

        if degree_input:
            node_input_dim = positional_embedding_size + degree_embedding_size + 1
        else:
            node_input_dim = positional_embedding_size + 1
        # node_input_dim = (
        #     positional_embedding_size + freq_embedding_size + degree_embedding_size + 3
        # )
        # edge_input_dim = freq_embedding_size + 1
        if gnn_model == "gat":
            self.gnn = GATModel(
                in_feats=node_input_dim,
                hidden_size=node_hidden_dim,
                num_layers=num_layers,
                nhead=num_heads,
                dropout=0.0,
            )
        elif gnn_model == "gin":
            self.gnn = GINModel(
                num_layers=num_layers,
                num_mlp_layers=2,
                in_feats=node_input_dim,
                hidden_dim=node_hidden_dim,
                out_feats=output_dim,
                final_dropout=0.5,
                train_eps=False,
                pooling="sum",
                # neighbor_pooling_type="sum",
                # use_selayer=False,
            )
        self.gnn_model = gnn_model

        self.max_node_freq = max_node_freq
        self.max_edge_freq = max_edge_freq
        self.max_degree = max_degree
        self.degree_input = degree_input
        self.output_dim = output_dim
        self.hidden_size = node_hidden_dim

        # self.node_freq_embedding = nn.Embedding(
        #     num_embeddings=max_node_freq + 1, embedding_dim=freq_embedding_size
        # )
        if degree_input:
            self.degree_embedding = nn.Embedding(num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size)

        # self.edge_freq_embedding = nn.Embedding(
        #     num_embeddings=max_edge_freq + 1, embedding_dim=freq_embedding_size
        # )

        self.set2set = Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)
        if gnn_model != "gin":
            self.lin_readout = nn.Sequential(
                nn.Linear(2 * node_hidden_dim, node_hidden_dim), nn.ReLU(), nn.Linear(node_hidden_dim, output_dim),
            )
        else:
            self.lin_readout = None
        self.norm = norm

    def forward(self, g, return_all_outputs=False):
        """Predict molecule labels
        Parameters
        ----------
        g : Graph
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.
        Returns
        -------
        res : Predicted labels
        """

        # nfreq = g.ndata["nfreq"]
        device = self.device
        pos_undirected = g.pos_undirected
        seed_emb = g.seed.unsqueeze(1).float()
        if not torch.is_tensor(seed_emb):
            seed_emb = torch.Tensor(seed_emb)

        if self.degree_input:
            degrees = g.degrees()
            if device != torch.device("cpu"):
                degrees = degrees.cuda(device)

            degrees = degrees.long()
            deg_emb = self.degree_embedding(degrees.clamp(0, self.max_degree))

            n_feat = torch.cat((pos_undirected, deg_emb, seed_emb), dim=-1)
        else:
            n_feat = torch.cat(
                (
                    pos_undirected,
                    # self.node_freq_embedding(nfreq.clamp(0, self.max_node_freq)),
                    # self.degree_embedding(degrees.clamp(0, self.max_degree)),
                    seed_emb,
                    # nfreq.unsqueeze(1).float() / self.max_node_freq,
                    # degrees.unsqueeze(1).float() / self.max_degree,
                ),
                dim=-1,
            )

        if self.gnn_model == "gin":
            x, all_outputs = self.gnn(g, n_feat)
        else:
            x, all_outputs = self.gnn(g, n_feat), None
            x = self.set2set(g, x)
            x = self.lin_readout(x)
        if self.norm:
            x = F.normalize(x, p=2, dim=-1, eps=1e-5)
        if return_all_outputs:
            return x, all_outputs
        else:
            return x


# --------------------------------------------------------------------
# --------------------------------------------------------------------


def warmup_linear(x, warmup=0.002):
    """Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
    After `t_total`-th training step, learning rate is zero."""
    if x < warmup:
        return x / warmup
    return max((x - 1.0) / (warmup - 1.0), 0)
