import numpy as np
import networkx as nx
from collections import defaultdict
from gensim.models.keyedvectors import Vocab  # Retained for now to ease the loading of older models.
# See: https://radimrehurek.com/gensim/models/keyedvectors.html?highlight=vocab#gensim.models.keyedvectors.CompatVocab
import random
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import BaseModel


class GATNE(BaseModel):
    r"""The GATNE model from the `"Representation Learning for Attributed Multiplex Heterogeneous Network"
    <https://dl.acm.org/doi/10.1145/3292500.3330964>`_ paper

    Args:
        walk_length (int) : The walk length.
        walk_num (int) : The number of walks to sample for each node.
        window_size (int) : The actual context size which is considered in language model.
        worker (int) : The number of workers for word2vec.
        epochs (int) : The number of training epochs.
        batch_size (int) : The size of each training batch.
        edge_dim (int) : Number of edge embedding dimensions.
        att_dim (int) : Number of attention dimensions.
        negative_samples (int) : Negative samples for optimization.
        neighbor_samples (int) : Neighbor samples for aggregation
        schema (str) : The metapath schema used in model. Metapaths are splited with ",",
        while each node type are connected with "-" in each metapath. For example:"0-1-0,0-1-2-1-0"
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--walk-length', type=int, default=10,
                            help='Length of walk per source. Default is 10.')
        parser.add_argument('--walk-num', type=int, default=10,
                            help='Number of walks per source. Default is 10.')
        parser.add_argument('--window-size', type=int, default=5,
                            help='Window size of skip-gram model. Default is 5.')
        parser.add_argument('--worker', type=int, default=10,
                            help='Number of parallel workers. Default is 10.')
        parser.add_argument('--epochs', type=int, default=20,
                            help='Number of epochs. Default is 20.')
        parser.add_argument('--batch-size', type=int, default=256,
                            help='Number of batch_size. Default is 256.')
        parser.add_argument('--edge-dim', type=int, default=10,
                            help='Number of edge embedding dimensions. Default is 10.')
        parser.add_argument('--att-dim', type=int, default=20,
                            help='Number of attention dimensions. Default is 20.')
        parser.add_argument('--negative-samples', type=int, default=5,
                            help='Negative samples for optimization. Default is 5.')
        parser.add_argument('--neighbor-samples', type=int, default=10,
                            help='Neighbor samples for aggregation. Default is 10.')
        parser.add_argument('--schema', type=str, default=None,
                            help="Input schema for metapath random walk.")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.walk_length,
            args.walk_num,
            args.window_size,
            args.worker,
            args.epochs,
            args.batch_size,
            args.edge_dim,
            args.att_dim,
            args.negative_samples,
            args.neighbor_samples,
            args.schema,
        )

    def __init__(
        self,
        dimension,
        walk_length,
        walk_num,
        window_size,
        worker,
        epochs,
        batch_size,
        edge_dim,
        att_dim,
        negative_samples,
        neighbor_samples,
        schema,
    ):
        super(GATNE, self).__init__()
        self.embedding_size = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_u_size = edge_dim
        self.dim_att = att_dim
        self.num_sampled = negative_samples
        self.neighbor_samples = neighbor_samples
        self.schema = schema
        self.multiplicity = True

    def forward(self, network_data):
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        all_walks = generate_walks(network_data, self.walk_num, self.walk_length, schema=self.schema)
        vocab, index_to_key = generate_vocab(all_walks)
        train_pairs = generate_pairs(all_walks, vocab)

        edge_types = list(network_data.keys())

        num_nodes = len(index_to_key)
        edge_type_count = len(edge_types)

        epochs = self.epochs
        batch_size = self.batch_size
        embedding_size = self.embedding_size
        embedding_u_size = self.embedding_u_size
        num_sampled = self.num_sampled
        dim_att = self.dim_att
        neighbor_samples = self.neighbor_samples

        neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
        for r in range(edge_type_count):
            g = network_data[edge_types[r]]
            for (x, y) in g:
                ix = vocab[x].index
                iy = vocab[y].index
                neighbors[ix][r].append(iy)
                neighbors[iy][r].append(ix)
            for i in range(num_nodes):
                if len(neighbors[i][r]) == 0:
                    neighbors[i][r] = [i] * neighbor_samples
                elif len(neighbors[i][r]) < neighbor_samples:
                    neighbors[i][r].extend(
                        list(np.random.choice(neighbors[i][r], size=neighbor_samples - len(neighbors[i][r]),))
                    )
                elif len(neighbors[i][r]) > neighbor_samples:
                    neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))

        model = GATNEModel(num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_att)
        nsloss = NSLoss(num_nodes, num_sampled, embedding_size)

        model.to(device)
        nsloss.to(device)

        optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": nsloss.parameters()}], lr=1e-4)

        for epoch in range(epochs):
            random.shuffle(train_pairs)
            batches = get_batches(train_pairs, neighbors, batch_size)

            data_iter = tqdm.tqdm(
                batches,
                desc="epoch %d" % (epoch),
                total=(len(train_pairs) + (batch_size - 1)) // batch_size,
                bar_format="{l_bar}{r_bar}",
            )
            avg_loss = 0.0

            for i, data in enumerate(data_iter):
                optimizer.zero_grad()
                embs = model(data[0].to(device), data[2].to(device), data[3].to(device),)
                loss = nsloss(data[0].to(device), embs, data[1].to(device))
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()

                if i % 5000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss.item(),
                    }
                    data_iter.write(str(post_fix))

        final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
        for i in range(num_nodes):
            train_inputs = torch.tensor([i for _ in range(edge_type_count)]).to(device)
            train_types = torch.tensor(list(range(edge_type_count))).to(device)
            node_neigh = torch.tensor([neighbors[i] for _ in range(edge_type_count)]).to(device)
            node_emb = model(train_inputs, train_types, node_neigh)
            for j in range(edge_type_count):
                final_model[edge_types[j]][index_to_key[i]] = node_emb[j].cpu().detach().numpy()
        return final_model


class GATNEModel(nn.Module):
    def __init__(self, num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a):
        super(GATNEModel, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a

        self.node_embeddings = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.node_type_embeddings = Parameter(torch.FloatTensor(num_nodes, edge_type_count, embedding_u_size))
        self.trans_weights = Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size))
        self.trans_weights_s1 = Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, dim_a))
        self.trans_weights_s2 = Parameter(torch.FloatTensor(edge_type_count, dim_a, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.node_embeddings.data.uniform_(-1.0, 1.0)
        self.node_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, train_inputs, train_types, node_neigh):
        node_embed = self.node_embeddings[train_inputs]
        node_embed_neighbors = self.node_type_embeddings[node_neigh]
        node_embed_tmp = torch.cat(
            [node_embed_neighbors[:, i, :, i, :].unsqueeze(1) for i in range(self.edge_type_count)], dim=1,
        )
        node_type_embed = torch.sum(node_embed_tmp, dim=2)

        trans_w = self.trans_weights[train_types]
        trans_w_s1 = self.trans_weights_s1[train_types]
        trans_w_s2 = self.trans_weights_s2[train_types]

        attention = F.softmax(
            torch.matmul(F.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2).squeeze()
        ).unsqueeze(1)
        node_type_embed = torch.matmul(attention, node_type_embed)
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze()

        last_node_embed = F.normalize(node_embed, dim=1)

        return last_node_embed


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(
            torch.Tensor([(math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1) for k in range(num_nodes)]),
            dim=0,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1)))
        negs = torch.multinomial(self.sample_weights, self.num_sampled * n, replacement=True).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


class RWGraph:
    def __init__(self, nx_G, node_type=None):
        self.G = nx_G
        self.node_type = node_type

    def walk(self, walk_length, start, schema=None):
        # Simulate a random walk starting from start node.
        G = self.G

        rand = random.Random()

        if schema:
            schema_items = schema.split("-")
            assert schema_items[0] == schema_items[-1]

        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            for node in G[cur].keys():
                if schema is None or self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, schema=None):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        if schema is not None:
            schema_list = schema.split(",")
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if schema is None:
                    walks.append(self.walk(walk_length=walk_length, start=node))
                else:
                    for schema_iter in schema_list:
                        if schema_iter.split("-")[0] == self.node_type[node]:
                            walks.append(self.walk(walk_length=walk_length, start=node, schema=schema_iter,))

        return walks


def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + "_" + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.Graph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = int(edge_key.split("_")[0])
        y = int(edge_key.split("_")[1])
        tmp_G.add_edge(x, y)
        tmp_G[x][y]["weight"] = weight
    return tmp_G


def generate_pairs(all_walks, vocab, window_size=5):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        for walk in walks:
            for i in range(len(walk)):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))
    return pairs


def generate_vocab(all_walks):
    index_to_key = []
    raw_vocab = defaultdict(int)

    for walks in all_walks:
        for walk in walks:
            for word in walk:
                raw_vocab[word] += 1

    vocab = {}
    for word, v in raw_vocab.items():
        vocab[word] = Vocab(count=v, index=len(index_to_key))
        index_to_key.append(word)

    index_to_key.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index_to_key):
        vocab[word].index = i

    return vocab, index_to_key


def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    # result = []
    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield torch.tensor(x), torch.tensor(y), torch.tensor(t), torch.tensor(neigh)


def generate_walks(network_data, num_walks, walk_length, schema=None):
    # if schema is not None:
    #     pass
    # else:
    #     node_type = None

    all_walks = []
    for layer_id in network_data:
        tmp_data = network_data[layer_id]
        # start to do the random walk on a layer

        layer_walker = RWGraph(get_G_from_edges(tmp_data))
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)

        all_walks.append(layer_walks)

    return all_walks
