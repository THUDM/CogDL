from sklearn.metrics import roc_auc_score

import torch
from .. import ModelWrapper
from cogdl.utils import negative_edge_sampling


class GNNLinkPredictionModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg):
        super(GNNLinkPredictionModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.loss_fn = torch.nn.BCELoss()

    def train_step(self, subgraph):
        graph = subgraph

        train_neg_edges = negative_edge_sampling(graph.train_edges, graph.num_nodes).to(self.device)
        train_pos_edges = graph.train_edges
        edge_index = torch.cat([train_pos_edges, train_neg_edges], dim=1)
        labels = self.get_link_labels(train_pos_edges.shape[1], train_neg_edges.shape[1], self.device)

        # link prediction loss
        with graph.local_graph():
            graph.edge_index = edge_index
            emb = self.model(graph)
        pred = (emb[edge_index[0]] * emb[edge_index[1]]).sum(1)
        pred = torch.sigmoid(pred)
        loss = self.loss_fn(pred, labels)
        return loss

    def val_step(self, subgraph):
        graph = subgraph
        pos_edges = graph.val_edges
        neg_edges = graph.val_neg_edges
        train_edges = graph.train_edges
        edges = torch.cat([pos_edges, neg_edges], dim=1)
        labels = self.get_link_labels(pos_edges.shape[1], neg_edges.shape[1], self.device).long()
        with graph.local_graph():
            graph.edge_index = train_edges
            with torch.no_grad():
                emb = self.model(graph)
                pred = (emb[edges[0]] * emb[edges[1]]).sum(-1)
        pred = torch.sigmoid(pred)

        auc_score = roc_auc_score(labels.cpu().numpy(), pred.cpu().numpy())

        self.note("auc", auc_score)

    def test_step(self, subgraph):
        graph = subgraph
        pos_edges = graph.test_edges
        neg_edges = graph.test_neg_edges
        train_edges = graph.train_edges
        edges = torch.cat([pos_edges, neg_edges], dim=1)
        labels = self.get_link_labels(pos_edges.shape[1], neg_edges.shape[1], self.device).long()
        with graph.local_graph():
            graph.edge_index = train_edges
            with torch.no_grad():
                emb = self.model(graph)
                pred = (emb[edges[0]] * emb[edges[1]]).sum(-1)
        pred = torch.sigmoid(pred)

        auc_score = roc_auc_score(labels.cpu().numpy(), pred.cpu().numpy())

        self.note("auc", auc_score)

    @staticmethod
    def get_link_labels(num_pos, num_neg, device=None):
        labels = torch.zeros(num_pos + num_neg)
        labels[:num_pos] = 1
        if device is not None:
            labels = labels.to(device)
        return labels.float()

    def setup_optimizer(self):
        lr, wd = self.optimizer_cfg["lr"], self.optimizer_cfg["weight_decay"]
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def set_early_stopping(self):
        return "auc", ">"
