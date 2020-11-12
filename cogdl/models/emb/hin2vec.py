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


class Hin2vec_layer(nn.Module):
    def __init__(self, num_node, num_relation, hidden_size, cpu):
        super(Hin2vec_layer, self).__init__()
        
        self.num_node = num_node
        
        self.Wx = Parameter(torch.randn(num_node, hidden_size))
        self.Wr = Parameter(torch.randn(num_relation, hidden_size))
        
        self.device = torch.device('cpu' if cpu else 'cuda')
        
        self.X = F.one_hot(torch.arange(num_node), num_node).float().to(self.device)
        self.R = F.one_hot(torch.arange(num_relation), num_relation).float().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def regulartion(self, embr):
        clamp_embr = torch.clamp(embr, -6.0, 6.0)
        sigmod1 = torch.sigmoid(clamp_embr)
        # return sigmod1
        re_embr = torch.mul(sigmod1, 1-sigmod1)
        return re_embr 


    def forward(self, x, y, r, l):
        x_one, y_one, r_one = torch.index_select(self.X, 0, x), torch.index_select(self.X, 0, y), torch.index_select(self.R, 0, r)
        self.embx, self.emby, self.embr = torch.mm(x_one, self.Wx), torch.mm(y_one, self.Wx), torch.mm(r_one, self.Wr)
        self.re_embr = self.regulartion(self.embr)
        self.preds = torch.unsqueeze(torch.sigmoid(torch.sum(torch.mul(torch.mul(self.embx, self.emby), self.re_embr), 1)),1)
        self.logits = torch.cat((self.preds, 1- self.preds), 1)
        return self.logits, self.criterion(self.logits, l)


    def get_emb(self,):
        x = F.one_hot(torch.arange(0, self.num_node), num_classes=self.num_node).float().to(self.device)
        return torch.mm(x, self.Wx)


class RWgraph():
    def __init__(self, nx_G, node_type=None):
        self.G = nx_G
        self.node_type = node_type

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
        walks = []
        nodes = list(self.G.nodes())
        print("node number:", len(nodes))
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), "/", str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._walk(node, walk_length))
        return walks

    def data_preparation(self, walks, hop, negative):
        # data preparation via process walks and negative sampling
        node_type = self.node_type
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
        return np.asarray(pairs), relation


@register_model("hin2vec")
class Hin2vec(BaseModel):
    r"""The Hin2vec model from the `"HIN2Vec: Explore Meta-paths in Heterogeneous Information Networks for Representation Learning"
    <https://dl.acm.org/doi/10.1145/3132847.3132953>`_ paper.
    
    Args:
        hidden_size (int) : The dimension of node representation.
        walk_length (int) : The walk length.
        walk_num (int) : The number of walks to sample for each node.
        batch_size (int) : The batch size of training in Hin2vec.
        hop (int) : The number of hop to construct training samples in Hin2vec.
        negative (int) : The number of nagative samples for each meta2path pair.
        epochs (int) : The number of training iteration.
        lr (float) : The initial learning rate of SGD.
        cpu (bool) : Use CPU or GPU to train hin2vec.
    """
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
        parser.add_argument("--epochs", type=int, default=1)


    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.walk_length,
            args.walk_num,
            args.batch_size,
            args.hop,
            args.negative,
            args.epochs,
            args.lr,
            args.cpu
        )

    def __init__(self, hidden_dim, walk_length, walk_num, batch_size, hop, negative, epochs, lr, cpu=True):
        super(Hin2vec, self).__init__()
        self.hidden_dim = hidden_dim
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.batch_size = batch_size
        self.hop = hop
        self.negative = negative
        self.epochs = epochs
        self.lr = lr
        self.cpu = cpu
        self.device = torch.device('cpu' if self.cpu else 'cuda')
        

    def train(self, G, node_type):
        self.num_node = G.number_of_nodes()
        rw = RWgraph(G, node_type)
        walks = rw._simulate_walks(self.walk_length, self.walk_num)
        pairs, relation = rw.data_preparation(walks, self.hop, self.negative)
                
        self.num_relation = len(relation)
        model = Hin2vec_layer(self.num_node, self.num_relation, self.hidden_dim, self.cpu)
        self.model = model.to(self.device)
        
        num_batch = int(len(pairs) / self.batch_size)
        print_num_batch = 100
        print("number of batch", num_batch)
        
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        epoch_iter = tqdm(range(self.epochs))
        for epoch in epoch_iter:
            loss_n, pred, label = [], [], []
            for i in range(num_batch):
                batch_pairs = torch.from_numpy(pairs[i *self.batch_size :(i+1) * self.batch_size])
                batch_pairs = batch_pairs.to(self.device)
                batch_pairs = batch_pairs.T
                x, y, r, l = batch_pairs[0], batch_pairs[1], batch_pairs[2], batch_pairs[3]
                opt.zero_grad()
                logits, loss = self.model.forward(x, y, r, l)
                
                loss_n.append(loss.item())
                label.append(l)
                pred.extend(logits)
                if i% print_num_batch ==0 and i!=0:
                    label = torch.cat(label).to(self.device)
                    pred = torch.stack(pred, dim=0)
                    pred = pred.max(1)[1]
                    acc = pred.eq(label).sum().item() / len(label)
                    epoch_iter.set_description(
                    f"Epoch: {i:03d}, Loss: {sum(loss_n)/print_num_batch:.5f}, Acc: {acc:.5f}"
                    )
                    loss_n, pred, label = [], [], []
                    
                loss.backward()
                opt.step()

        embedding = self.model.get_emb()
        return embedding.cpu().detach().numpy()






