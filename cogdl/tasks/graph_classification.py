import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from tqdm import tqdm


from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.data import DataLoader, Data
from cogdl.models import build_model

from . import BaseTask, register_task

def node_degree_as_feature(data):
    max_degree = 0
    degrees = []
    for graph in data:
        edge_index =graph.edge_index
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        fill_value = 1
        num_nodes = graph.num_nodes
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).long()
        degrees.append(deg.cpu()-1)
        max_degree = max(torch.max(deg), max_degree)
    max_degree = int(max_degree)
    for i in range(len(data)):
        one_hot = torch.zeros(data[i].num_nodes, max_degree).scatter_(1, degrees[i].unsqueeze(1), 1)
        data[i].x = one_hot.cuda()
    return data


def uniform_node_feature(data):
    feat_dim = 2
    init_feat = torch.rand(1, feat_dim)
    for i in range(len(data)):
        data[i].x =init_feat.repeat(1, data[i].num_nodes)
    return data


@register_task("graph_classification")
class GraphClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument("--num-features", type=int)
        # fmt: on
        # batch_size \in {32, 128}
        parser.add_argument("--degree-feature", dest="degree_feature", action="store_true")

    def __init__(self, args):
        super(GraphClassification, self).__init__(args)

        dataset = build_dataset(args)
        self.data = [
            Data(x=data.x, y=data.y, edge_index=data.edge_index, edge_attr=data.edge_attr, pos=data.pos).apply(lambda x:x.cuda())
            for data in dataset
        ]
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        args.use_unsup = False
        if args.degree_feature:
            self.data = node_degree_as_feature(self.data)
            args.num_features = self.data[0].num_features


        model = build_model(args)
        self.model = model.cuda()
        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.train_loader, self.val_loader, self.test_loader = self.model.split_dataset(self.data, args)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=50,
            gamma=0.5
        )

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 0
        best_model = None
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf

        for epoch in epoch_iter:
            self.scheduler.step()
            self._train_step()
            train_acc, train_loss = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Train_loss: {train_loss:.4f}, Val_loss: {val_loss:.4f}"
            )
            if val_loss < min_loss or val_acc > max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_score = val_acc
                    best_model = copy.deepcopy(self.model)
                min_loss = np.min((min_loss, val_loss))
                max_score = np.max((max_score, val_acc))
                patience = 0
            else:
                patience += 1
                if patience == self.patience:
                    self.model = best_model
                    epoch_iter.close()
                    break
        test_acc, _ = self._test_step(split="test")
        print(f"Test accuracy = {test_acc}")
        return dict(Acc=test_acc)

    def _train_step(self):
        self.model.train()
        loss_n = 0
        for batch in self.train_loader:
            batch = batch.cuda()
            self.optimizer.zero_grad()
            batch_data = batch if self.model.__class__.__name__ == "PatchySAN" else batch.batch
            predict, loss = self.model(batch.x, batch.edge_index, batch_data, label=batch.y)
            loss_n += loss.item()
            loss.backward()
            self.optimizer.step()


    def _test_step(self, split="val"):
        self.model.eval()
        if split == "train":
            loader = self.train_loader
        elif split == "valid":
            loader = self.val_loader
        else:
            loader = self.test_loader
        loss_n = 0
        pred = []
        y = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.cuda()
                batch_data = batch if self.model.__class__.__name__ == "PatchySAN" else batch.batch
                predict, loss = self.model(batch.x, batch.edge_index, batch_data, label=batch.y)
                loss_n += loss.item()
                y.append(batch.y)
                pred.extend(predict)

        y = torch.cat(y).cuda()
        pred = torch.stack(pred, dim=0)
        pred = pred.max(1)[1]
        acc = pred.eq(y).sum().item() / len(y)
        return acc, loss_n
