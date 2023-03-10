import numpy as np
from cogdl import function as BF
from cogdl.backend import BACKEND
if BACKEND == 'jittor':
    import jittor as tj
    from jittor import nn, Module
elif BACKEND == 'torch':
    import torch as tj
    from torch import nn
    from torch.nn import Module

from .. import UnsupervisedModelWrapper
from cogdl.wrappers.tools.wrapper_utils import evaluate_node_embeddings_using_logreg


class DGIModelWrapper(UnsupervisedModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=512)
        # fmt: on

    def __init__(self, model, optimizer_cfg):
        super(DGIModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

        hidden_size = optimizer_cfg["hidden_size"]
        assert hidden_size > 0
        self.disc = Discriminator(hidden_size)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.act = nn.PReLU()

    def train_step(self, subgraph):
        graph = subgraph
        graph.sym_norm()
        x = graph.x
        shuffle_x = self.augment(graph)

        graph.x = x
        h_pos = self.act(self.model(graph))
        c = self.read(h_pos)
        c = self.sigm(c)

        graph.x = shuffle_x
        h_neg = self.act(self.model(graph))
        logits = self.disc(c, h_pos, h_neg)
        graph.x = x

        num_nodes = x.shape[0]
        labels = BF.zeros(num_nodes * 2, device=BF.device(x))
        # labels = tj.zeros((num_nodes * 2,), device=x.device)
        labels[:num_nodes] = 1
        loss = self.loss_fn(logits, labels)
        return loss

    def test_step(self, graph):
        with tj.no_grad():
            pred = self.act(self.model(graph))
        y = graph.y
        result = evaluate_node_embeddings_using_logreg(pred, y, graph.train_mask, graph.test_mask)
        self.note("test_acc", result)

    @staticmethod
    def augment(graph):
        idx = np.random.permutation(graph.num_nodes)
        shuffle_x = graph.x[idx, :]
        return shuffle_x

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return tj.optim.Adam(self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


# Borrowed from https://github.com/PetarV-/DGI
class AvgReadout(Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    
    if BACKEND == 'jittor':
        def execute(self, seq, msk=None):
            dim = len(seq.shape) - 2
            if msk is None:
                return BF.mean(seq, dim)
            else:
                return BF.sum(seq * msk, dim) / BF.sum(msk)
    elif BACKEND == 'torch':
        def forward(self, seq, msk=None):
            dim = len(seq.shape) - 2
            if msk is None:
                return BF.mean(seq, dim)
            else:
                return BF.sum(seq * msk, dim) / BF.sum(msk)
        


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            BF.xavier_uniform_(m.weight)
            if m.bias is not None:
                BF.fill_(m.bias, 0.0)
            

    def temp(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = BF.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = BF.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = BF.squeeze(self.f_k(h_mi, c_x), 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = BF.cat((sc_1, sc_2))
        return logits

    if BACKEND == 'jittor':
        def execute(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
             return self.temp(c, h_pl, h_mi, s_bias1=None, s_bias2=None)
    elif BACKEND == 'torch':
        def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
            return self.temp(c, h_pl, h_mi, s_bias1=None, s_bias2=None)

