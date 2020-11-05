import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk

from .dgi import LogRegTrainer
from .. import register_model, BaseModel
from cogdl.models.nn.graphsage import sage_sampler, GraphSAGELayer


class SAGE(nn.Module):
    def sampling(self, edge_index, num_sample):
        return sage_sampler(self.adjlist, edge_index, num_sample)

    def __init__(
        self, num_features, hidden_size, num_layers, sample_size, dropout, walk_length, negative_samples
    ):
        super(SAGE, self).__init__()
        self.adjlist = {}
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sample_size = sample_size
        self.dropout = dropout
        self.walk_length = walk_length
        self.num_negative_samples = negative_samples
        self.walk_res = None
        self.num_nodes = None
        self.negative_samples = None

        shapes = [num_features] + [hidden_size] * num_layers

        self.convs = nn.ModuleList(
            [
                GraphSAGELayer(shapes[layer], shapes[layer+1])
                for layer in range(num_layers)
            ]
        )

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            edge_index_sp = self.sampling(edge_index, self.sample_size[i]).to(x.device)
            x = self.convs[i](x, edge_index_sp)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def loss(self, data):
        x = self.forward(data.x, data.edge_index)
        device = x.device
        # if self.walk_res is None:
        self.walk_res = random_walk(data.edge_index[0], data.edge_index[1],
                                    start=torch.arange(0, x.shape[0]).to(device),
                                    walk_length=self.walk_length)[:, 1:]

        if not self.num_nodes:
            self.num_nodes = int(torch.max(data.edge_index)) + 1

        # if self.negative_samples is None:
        self.negative_samples = torch.from_numpy(
            np.random.choice(self.num_nodes, (self.num_nodes, self.num_negative_samples))
        ).to(device)

        pos_loss = -torch.log(
            torch.sigmoid(
                torch.sum(x.unsqueeze(1).repeat(1, self.walk_length, 1) * x[self.walk_res], dim=-1)
            )
        ).mean()
        neg_loss = -torch.log(
            torch.sigmoid(-torch.sum(x.unsqueeze(1).repeat(1, self.num_negative_samples, 1) * x[self.negative_samples], dim=-1))
        ).mean()
        return (pos_loss + neg_loss)/2

    def embed(self, data):
        emb = self.forward(data.x, data.edge_index)
        return emb


@register_model("unsup_graphsage")
class Graphsage(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--sample-size", type=int, nargs='+', default=[10, 10])
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--walk-length", type=int, default=10)
        parser.add_argument("--negative-samples", type=int, default=30)
        parser.add_argument("--lr", type=float, default=0.001)

        parser.add_argument("--max-epochs", type=int, default=3000)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.sample_size,
            args.dropout,
            args.walk_length,
            args.negative_samples,
            args.lr,
            args.max_epochs,
            args.patience,
        )

    def __init__(
            self, num_features, hidden_size, num_classes, num_layers,
            sample_size, dropout, walk_length, negative_samples, lr, epochs, patience
    ):
        super(Graphsage, self).__init__()
        self.model = SAGE(num_features, hidden_size, num_layers, sample_size, dropout, walk_length, negative_samples)
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.nhid = hidden_size
        self.nclass = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, data):
        data.apply(lambda x: x.to(self.device))
        self.model.to(self.device)
        device = data.x.device
        best = 1e9
        cnt_wait = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)

        epoch_iter = tqdm.tqdm(range(self.epochs))
        for epoch in epoch_iter:
            self.model.train()
            optimizer.zero_grad()

            loss = self.model.loss(data)
            epoch_iter.set_description(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}')

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                print('Early stopping!')
                break

            loss.backward()
            optimizer.step()
        self.model.eval()
        embeds = self.model.embed(data).detach()

        opt = {
            "idx_train": data.train_mask,
            "idx_val": data.val_mask,
            "idx_test": data.test_mask,
            "num_classes": self.nclass
        }
        result = LogRegTrainer().train(embeds, data.y, opt)
        return result

