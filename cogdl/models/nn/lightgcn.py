"""
from from https://github.com/huangtinglin/MixGCF

Created on October 1, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
"""
import torch
import torch.nn as nn

from cogdl.models import BaseModel, register_model


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, n_hops, n_users, interact_mat, edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.interact_mat = self.interact_mat.to(*args, **kwargs)
        return self

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1.0 / (1 - rate))

    def forward(self, user_embed, item_embed, mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = (
                self._sparse_dropout(self.interact_mat, self.edge_dropout_rate) if edge_dropout else self.interact_mat
            )

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[: self.n_users, :], embs[self.n_users :, :]


@register_model("lightgcn")
class LightGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dim', type=int, default=64, help='embedding size')
        parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight, 1e-5 for NGCF')
        parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
        parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
        parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
        parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
        parser.add_argument("--ns", type=str, default='mixgcf', help="rns,mixgcf")
        parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")
        parser.add_argument("--n_negs", type=int, default=64, help="number of candidate negative")
        parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")
        parser.add_argument("--context_hops", type=int, default=3, help="hop")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.n_users,
            args.n_items,
            args.l2,
            args.dim,
            args.context_hops,
            args.mess_dropout,
            args.mess_dropout_rate,
            args.edge_dropout,
            args.edge_dropout_rate,
            args.pool,
            args.n_negs,
            args.ns,
            args.K,
            args.adj_mat,
        )

    def __init__(
        self,
        n_users,
        n_items,
        l2,
        dim,
        context_hops,
        mess_dropout,
        mess_dropout_rate,
        edge_dropout,
        edge_dropout_rate,
        pool,
        n_negs,
        ns,
        K,
        adj_mat,
    ):
        super(LightGCN, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.adj_mat = adj_mat

        self.decay = l2
        self.emb_size = dim
        self.context_hops = context_hops
        self.mess_dropout = mess_dropout
        self.mess_dropout_rate = mess_dropout_rate
        self.edge_dropout = edge_dropout
        self.edge_dropout_rate = edge_dropout_rate
        self.pool = pool
        self.n_negs = n_negs
        self.ns = ns
        self.K = K

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat)

    def _init_model(self):
        return GraphConv(
            n_hops=self.context_hops,
            n_users=self.n_users,
            interact_mat=self.sparse_norm_adj,
            edge_dropout_rate=self.edge_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
        )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        # self.sparse_norm_adj = self.sparse_norm_adj.to(*args, **kwargs)
        self.gcn.to(*args, **kwargs)
        return self

    def forward(self, batch=None):
        user = batch["users"]
        pos_item = batch["pos_items"]
        neg_item = batch["neg_items"]  # [batch_size, n_negs * K]

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(
            self.user_embed, self.item_embed, edge_dropout=self.edge_dropout, mess_dropout=self.mess_dropout
        )

        if self.ns == "rns":  # n_negs = 1
            neg_gcn_embs = item_gcn_emb[neg_item[:, : self.K]]
        else:
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(
                    self.negative_sampling(
                        user_gcn_emb, item_gcn_emb, user, neg_item[:, k * self.n_negs : (k + 1) * self.n_negs], pos_item
                    )
                )
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs)

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != "concat":
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == "mean":
            return embeddings.mean(dim=1)
        elif self.pool == "sum":
            return embeddings.sum(dim=1)
        elif self.pool == "concat":
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed, self.item_embed, edge_dropout=False, mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(
            batch_size, self.K, -1
        )

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]

        mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        # cul regularizer
        regularize = (
            torch.norm(user_gcn_emb[:, 0, :]) ** 2
            + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
            + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2
        ) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
