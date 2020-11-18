import copy

import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from cogdl.utils import add_remaining_self_loops


def random_partition_graph(num_nodes, cluster_number=10):
    return np.random.randint(cluster_number, size=num_nodes)


def generate_subgraphs(edge_index, parts, cluster_number=10, batch_size=1):
    num_nodes = torch.max(edge_index) + 1
    num_edges = edge_index.shape[1]
    device = edge_index.device
    edge_index_np = edge_index.cpu().numpy()
    adj = sparse.csr_matrix(
            (np.ones(num_edges), (edge_index_np[0], edge_index_np[1])),
            shape=(num_nodes, num_nodes)
        )
    num_batches = cluster_number // batch_size
    sg_nodes = []
    sg_edges = []

    for cluster in range(num_batches):
        node_cluster = np.where((parts == cluster))[0]
        part_adj = adj[node_cluster, :][:, node_cluster]
        part_adj_coo = sparse.coo_matrix(part_adj)
        row, col = part_adj_coo.row, part_adj_coo.col

        sg_nodes.append(node_cluster)
        sg_edges.append(torch.Tensor([row, col]).long())
    return sg_nodes, sg_edges


class DeeperGCNTrainer(BaseTrainer):
    def __init__(self, args):
        super(DeeperGCNTrainer, self).__init__()
        self.device = args.device_id[0] if not args.cpu else "cpu"
        self.patience = args.patience // 5
        self.max_epoch = args.max_epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.cluster_number = args.cluster_number
        self.batch_size = args.batch_size
        self.data = None
        self.optimizer = None
        self.edge_index, self.train_index = None, None

    def fit(self, model, data):
        data = data[0]
        self.model = model.to(self.device)
        self.data = data
        self.test_gpu_volume()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.edge_index, _ = add_remaining_self_loops(
            data.edge_index,
            torch.ones(data.edge_index.shape[1]).to(data.x.device),
            1,
            data.x.shape[0]
        )
        self.train_index = torch.where(data.train_mask)[0].tolist()

        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        for epoch in epoch_iter:
            self._train_step()
            if epoch % 5 == 0:
                val_acc, val_loss = self._test_step(split="val")
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, Val: {val_acc:.4f}"
                )
                if val_loss <= min_loss or val_acc >= max_score:
                    if val_acc >= best_score:  # SAINT loss is not accurate
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
        val_acc, _ = self._test_step(split="val")
        print(f"Test accuracy = {test_acc}")
        return dict(Acc=test_acc, ValAcc=val_acc)

    def test_gpu_volume(self):
        try:
            self.data.apply(lambda x: x.to(self.device))
            self.data_device = self.device
        except Exception as e:
            print(e)
            self.data_device = "cpu"
        print(f"device(model):{self.device}, device(data):{self.data_device}")

    def _train_step(self):
        self.model.train()
        x = self.data.x
        y = self.data.y
        num_nodes = x.shape[1]

        parts = random_partition_graph(
            num_nodes=num_nodes,
            cluster_number=self.cluster_number
        )
        subgraph_nodes, subgraph_edges = generate_subgraphs(
            self.edge_index,
            parts,
            self.cluster_number,
            batch_size=self.batch_size
        )

        idx_clusters = np.arange(len(subgraph_nodes))
        np.random.shuffle(idx_clusters)

        for idx in idx_clusters:
            self.optimizer.zero_grad()

            nodes, edges = subgraph_nodes[idx], subgraph_edges[idx].to(self.device)
            _x = x[nodes].to(self.device)
            mapper = {val: idx for idx, val in enumerate(nodes)}
            intersection_index = list(set(self.train_index) & set(nodes))
            training_index = [mapper[k] for k in intersection_index]

            targets = y[intersection_index].to(self.device)

            loss_n = self.model.loss(_x, edges, targets, training_index)
            loss_n.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()

    def _test_step(self, split="val"):
        self.model.eval()
        if self.data_device == "cpu":
            self.model.to("cpu")

        logits = self.model.predict(self.data.x, self.data.edge_index)
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        loss = F.nll_loss(logits[mask], self.data.y[mask]).item()

        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()

        self.data.apply(lambda x: x.to(self.data_device))
        self.model.to(self.device)
        return acc, loss

    def loss(self, data):
        return F.nll_loss(
            self.model(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask]
        )

    def predict(self, data):
        return self.model(data.x, data.edge_index)

    @classmethod
    def build_trainer_from_args(cls, args):
        return DeeperGCNTrainer(args)