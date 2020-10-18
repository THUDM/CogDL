import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_mrr(embedding, rel_embedding, edge_index, edge_type, scoring, protocol="raw", batch_size=1000, hits=[]):
    with torch.no_grad():
        if protocol == "raw":
            heads = edge_index[0]
            tails = edge_index[1]
            ranks_h = get_raw_rank(heads, tails, edge_type, embedding, rel_embedding, batch_size, scoring)
            ranks_t = get_raw_rank(tails, heads, edge_type, embedding, rel_embedding, batch_size, scoring)
            # ranks = torch.cat((ranks_h, ranks_t)) + 1
            ranks = np.concatenate((ranks_h, ranks_t)) + 1
        elif protocol == "filtered":
            raise NotImplementedError
        else:
            raise ValueError
        mrr = (1. / ranks).mean()
        hits_count = []
        # for hit in hits:
            # hits_count.append(torch.mean((ranks <= hit).float()).item())
        for hit in hits:
            hits_count.append(np.mean((ranks <= hit).astype(np.float)))
    # return mrr.item(), hits_count
    return mrr, hits_count


class DistMultLayer(nn.Module):
    def __init__(self):
        super(DistMultLayer, self).__init__()

    def forward(self, sub_emb, obj_emb, rel_emb):
        return torch.sum(sub_emb * obj_emb * rel_emb, dim=-1)

    def predict(self, sub_emb, obj_emb, rel_emb):
        return torch.matmul(sub_emb * rel_emb, obj_emb.t())


class ConvELayer(nn.Module):
    def __init__(self, dim, num_filter=20, kernel_size=7, k_w=10, dropout=0.3):
        super(ConvELayer, self).__init__()
        assert dim % k_w == 0
        self.k_w = k_w
        self.k_h = dim // k_w
        self.dim = dim
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(num_filter)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.hidden_drop = nn.Dropout(dropout)
        self.hidden_drop2 = nn.Dropout(dropout)
        self.feature_drop = nn.Dropout(dropout)
        self.conv = nn.Conv2d(1, out_channels=num_filter, kernel_size=(kernel_size, kernel_size), stride=1, padding=0, bias=True)

        flat_size_h = int(2*self.k_w) - kernel_size + 1
        flat_size_w = self.k_h - kernel_size + 1
        self.flat_size = flat_size_h * flat_size_w * num_filter
        self.fc = nn.Linear(self.flat_size, dim)

        self.bias = nn.Parameter(torch.zeros(dim))
    
    def concat(self, ent, rel):
        ent = ent.view(-1, 1, self.dim)
        rel = rel.view(-1, 1, self.dim)
        ent_rel = torch.cat([ent, rel], dim=1)
        ent_rel = ent_rel.transpose(2, 1).reshape(-1, 1, 2 * self.k_w, self.k_h)
        return ent_rel

    def forward(self, sub_emb, obj_emb, rel_emb):
        h = self.concat(sub_emb, rel_emb)
        h = self.bn0(h)
        h = self.conv(h)
        h = F.relu(self.bn1(h))
        h = self.feature_drop(h)
        h = h.view(-1, self.flat_size)
        h = self.hidden_drop(self.fc(h))
        h = F.relu(self.bn2(self.hidden_drop2(h)))
        x = torch.sum(h * obj_emb + self.bias, dim=-1)
        return x

    def predict(self, sub_emb, obj_emb, rel_emb):
        h = self.concat(sub_emb, rel_emb)
        h = self.bn0(h)
        h = self.conv(h)
        h = F.relu(self.bn1(h))
        h = h.view(-1, self.flat_size)
        h = self.fc(h)
        h = F.relu(self.bn2(h))
        x = torch.matmul(h, obj_emb.t())
        return x


class GNNLinkPredict(nn.Module):
    def __init__(self, score_func, dim):
        super(GNNLinkPredict, self).__init__()
        self.edge_set = None
        self.score_func = score_func
        if score_func == "distmult":
            self.scoring = DistMultLayer()
        elif score_func == "conve":
            self.scoring = ConvELayer(dim)
        else:
            raise NotImplementedError

    def forward(self, edge_index, edge_type):
        raise NotImplemented

    def get_score(self, heads, tails, rels):
        return self.scoring(heads, tails, rels)

    def get_edge_set(self, edge_index, edge_types):
        if self.edge_set is None:
            edge_list = torch.cat((edge_index, edge_types.unsqueeze(0)), dim=0).T
            edge_list = edge_list.cpu().T.numpy().tolist()
            torch.cuda.empty_cache()
            self.edge_set = set([tuple(x) for x in edge_list])  # tuple(h, t, r)

    def _loss(self, head_embed, tail_embed, rel_embed, labels):
        score = self.get_score(head_embed, tail_embed, rel_embed)
        prediction_loss = F.binary_cross_entropy_with_logits(score, labels.float())
        return prediction_loss

    def _regularization(self, embs):
        loss = 0
        for emb in embs:
            loss += torch.mean(emb.pow(2))
        return loss


def sampling_edge_uniform(edge_index, edge_types, edge_set, sampling_rate, num_rels, label_smoothing=0.0, num_entities=1):
    """
    Args:
        edge_index: edge index of graph
        edge_types:
        edge_set: set of all edges of the graph, (h, t, r)
        sampling_rate:
        num_rels:
        label_smoothing(Optional):
        num_entities (Optional):  

    Returns:
        sampled_edges: sampled existing edges
        rels: types of smapled existing edges
        sampled_edges_all: existing edges with corrupted edges
        sampled_types_all: types of existing and corrupted edges
        labels: 0/1
    """
    num_edges = edge_index.shape[1]
    num_sampled_edges = int(num_edges * sampling_rate)

    selected_edges = np.random.choice(range(num_edges), num_sampled_edges, replace=False)
    sampled_edges = edge_index.t()[selected_edges].t()

    sampled_nodes = torch.unique(sampled_edges).cpu().numpy()

    heads = sampled_edges[0].cpu().numpy()
    tails = sampled_edges[1].cpu().numpy()
    rels = edge_types[selected_edges].cpu().numpy()
    torch.cuda.empty_cache()

    def to_set(head, tail, rel):
        triplets = np.stack((head, tail, rel), axis=0).T
        triplets_set = set([tuple(x) for x in triplets])
        return triplets_set

    corrupt_heads = np.random.choice(sampled_nodes, num_sampled_edges, replace=True)
    corrupt_heads = to_set(corrupt_heads, tails, rels)
    corrupt_tails = np.random.choice(sampled_nodes, num_sampled_edges, replace=True)
    corrupt_tails = to_set(heads, corrupt_tails, rels)
    corrupt_rels = np.random.choice(num_rels, num_sampled_edges, replace=True)
    corrupt_rels = to_set(heads, tails, corrupt_rels)

    corrupt_triplets = corrupt_heads.union(corrupt_tails).union(corrupt_rels)
    corrupt_triplets = corrupt_triplets.difference(edge_set)
    corrupt_triplets = torch.tensor(list(corrupt_triplets)).to(edge_index.device).T

    _edge_index = corrupt_triplets[0:2]
    _edge_types = corrupt_triplets[2]

    sampled_edges_all = torch.cat((sampled_edges, _edge_index), dim=-1)
    edge_types_all = torch.cat((edge_types[selected_edges], _edge_types), dim=-1)
    labels = torch.tensor([1] * num_sampled_edges + [0] * _edge_index.shape[1]).to(edge_index.device)
    if label_smoothing > 0:
        labels = (1.0 - label_smoothing) * labels + 1. / num_entities
    return sampled_edges, torch.from_numpy(rels), sampled_edges_all, edge_types_all, labels


def get_rank(scores, target):
    _, indices = torch.sort(scores, dim=1, descending=True)
    rank = (indices == target.view(-1, 1)).nonzero()[:, 1]
    return rank.view(-1)


def get_raw_rank(heads, tails, rels, embedding, rel_embedding, batch_size, scoring):
    test_size = heads.shape[0]
    num_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for i in range(num_batch):
        start = batch_size * i
        end = start + batch_size
        scores = torch.sigmoid(scoring.predict(embedding[heads[start:end]], embedding, rel_embedding[rels[start:end]]))
        target = tails[start:end]
        rank = get_rank(scores, target).cpu().numpy()
        torch.cuda.empty_cache()
        ranks.append(rank)
    return np.concatenate(ranks).astype(np.float)


def get_filtered_rank(heads, tails, rels, embedding, rel_embedding, batch_size, seen_data):
    pass