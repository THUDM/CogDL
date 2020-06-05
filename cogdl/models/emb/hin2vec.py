import hashlib
import networkx as nx
import numpy as np
import random
from .. import BaseModel, register_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm


@register_model("hin2vec")
class Hin2vec(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument('--walk-length', type=int, default=80,
                            help='Length of walk per source. Default is 80.')
        parser.add_argument('--walk-num', type=int, default=40,
                            help='Number of walks per source. Default is 40.')
        parser.add_argument('--batch-size', type=int, default=1000,
                            help='Batch size in SGD training process. Default is 1000.')
        parser.add_argument("--hop", type=int, default=2)
        parser.add_argument("--negative", type=int, default=5)
        parser.add_argument("--epoches", type=int, default=20)


    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.walk_length,
            args.walk_num,
            args.batch_size,
            args.hop,
            args.negative,
            args.epoches,
            args.lr
        )

    def __init__(self, hidden_dim, walk_length, walk_num, batch_size, hop, negative, epoches, lr):
        super(Hin2vec, self).__init__()
        self.hidden_dim = hidden_dim
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.batch_size = batch_size
        self.hop = hop
        self.negative = negative
        self.epoches = epoches
        self.lr = lr


    def train(self, G, node_type):
        self.G = G
        self.node_type = node_type
        self.num_node = G.number_of_nodes()
        
        walks = self._simulate_walks(self.walk_length, self.walk_num)
        pairs, relation = data_preparation(walks, node_type, self.hop, self.negative)
        
        num_batch = int(len(pairs) / self.batch_size)
        self.num_relation = len(relation)
        
        self.build_nn(self.num_node, self.num_relation, self.hidden_dim)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        epoch_iter = tqdm(range(self.epoches))
        for epoch in epoch_iter:
            total_loss = 0
            for i in range(num_batch):
                batch_pairs = pairs[i *self.batch_size :(i+1) * self.batch_size]
                opt.zero_grad()
                logists, loss = self.forward(batch_pairs)
                total_loss += loss
                loss.backward()
                opt.step()
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Loss: {total_loss/num_batch:.5f}"
            )
        embedding = self.get_emb()
        return embedding.detach().numpy()


    def build_nn(self, num_node, num_relation, hidden_size):
        self.Wx = Parameter(torch.randn(num_node, hidden_size))
        self.Wr = Parameter(torch.randn(num_relation, hidden_size))
        self.criterion = nn.CrossEntropyLoss()

    def to_one_hot(self, batch_pairs):
        batch_pairs = batch_pairs.T
        x, y, r, l = batch_pairs[0].tolist()[0], batch_pairs[1].tolist()[0], batch_pairs[2].tolist()[0], batch_pairs[3].tolist()[0]
        x = F.one_hot(torch.from_numpy(np.array(x)), self.num_node).float()
        y = F.one_hot(torch.from_numpy(np.array(y)), self.num_node).float()
        r = F.one_hot(torch.from_numpy(np.array(r)), self.num_relation).float()
        l = torch.from_numpy(np.array(l))
        return x, y, r, l

    def regulartion(self, embr):
        clamp_embr = torch.clamp(embr, -6.0, 6.0)
        sigmod1 = torch.sigmoid(clamp_embr)
        # return sigmod1
        re_embr = torch.mul(sigmod1, 1-sigmod1)
        return re_embr 

    def forward(self, batch_pairs):
        x, y, r, l = self.to_one_hot(batch_pairs)
        self.embx, self.emby, self.embr = torch.mm(x, self.Wx), torch.mm(y, self.Wx), torch.mm(r, self.Wr)
        self.re_embr = self.regulartion(self.embr)
        self.preds = torch.unsqueeze(torch.sigmoid(torch.sum(torch.mul(torch.mul(self.embx, self.emby), self.re_embr), 1)),1)
        self.logits = torch.cat((self.preds, 1- self.preds), 1)
        return self.logits, self.criterion(self.logits, l)


    def get_emb(self,):
        x = F.one_hot(torch.arange(0, self.num_node), num_classes=self.num_node).float()
        return torch.mm(x, self.Wx)

    def _walk(self, start_node, walk_length):
        # Simulate a random walk starting from start node.
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) == 0:
                break
            k = int(np.floor(np.random.rand() * len(cur_nbrs)))
            walk.append(cur_nbrs[k])
        return walk

    def _simulate_walks(self, walk_length, num_walks):
        # Repeatedly simulate random walks from each node.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("node number:", len(nodes))
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), "/", str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._walk(node, walk_length))
        return walks

def data_preparation(walks, node_type, hop, negative):
    num_node_type = len(set(node_type))
    type2list = [[] for _ in range(num_node_type)]
    for node, nt in enumerate(node_type):
        type2list[nt].append(node)
    print("number of type2list", num_node_type)
    relation = dict()
    pairs = []
    for walk in walks:
        for i in range(len(walk) - hop):
            for j in range(1, hop+1):
                x, y = walk[i], walk[i+j]
                tx, ty = node_type[x], node_type[y]
                if x ==y: continue
                meta_str = "-".join([str(node_type[a]) for a in walk[i:i+j+1]])
                if meta_str not in relation:
                    relation[meta_str] = len(meta_str)
                pairs.append([x, y, relation[meta_str], 1])
                for k in range(negative):
                    if random.random() > 0.5:
                        fx = random.choice(type2list[node_type[x]])
                        while fx == x:
                            fx = random.choice(type2list[node_type[x]])
                        pairs.append([fx, y, relation[meta_str], 0])
                    else:
                        fy = random.choice(type2list[node_type[y]])
                        while fy == y:
                            fy = random.choice(type2list[node_type[y]])
                        pairs.append([x, fy, relation[meta_str], 0])                  
    print("number of relation", len(relation))
    return np.matrix(pairs), relation