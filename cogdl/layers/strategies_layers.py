import copy
import math
import os
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import (
    add_self_loops,
    cycle_index,
    batch_sum_pooling,
    batch_mean_pooling
)
from cogdl.datasets import build_dataset_from_name
from cogdl.datasets.pyg_strategies_data import *

from torch_geometric.data import DataLoader


class GINConv(nn.Module):
    def __init__(
            self,
            hidden_size,
            input_layer=None,
            edge_emb=None,
            edge_encode=None,
            pooling="sum",
            feature_concat=False
    ):
        super(GINConv, self).__init__()
        in_feat = 2 * hidden_size if feature_concat else hidden_size
        self.mlp = nn.Sequential(
            torch.nn.Linear(in_feat, 2 * hidden_size),
            torch.nn.BatchNorm1d(2 * hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_size, hidden_size)
        )

        self.input_node_embeddings = input_layer
        self.edge_embeddings = edge_emb
        self.edge_encoder = edge_encode
        self.feature_concat = feature_concat
        self.pooling = pooling

        if edge_emb is not None:
            self.edge_embeddings = [
                nn.Embedding(num, hidden_size)
                for num in edge_emb
            ]
        if input_layer is not None:
            self.input_node_embeddings = nn.Embedding(input_layer, hidden_size)
            nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)
        if edge_encode is not None:
            self.edge_encoder = nn.Linear(edge_encode, hidden_size)

    def forward(self, x, edge_index, edge_attr, self_loop_index=None, self_loop_type=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if self_loop_index is not None:
            self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
            self_loop_attr[:, self_loop_index] = self_loop_type
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            self_loop_attr.to(x.device)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        if self.edge_embeddings is not None:
            for i in range(edge_index.shape[0]):
                self.edge_embeddings[i].to(x.device)
            edge_embeddings = sum([
                self.edge_embeddings[i](edge_attr[:, i])
                for i in range(edge_index.shape[0])
            ])
        elif self.edge_encoder is not None:
            edge_embeddings = self.edge_encoder(edge_attr)
        else:
            raise NotImplementedError
        if self.input_node_embeddings is not None:
            x = self.input_node_embeddings(x.long().view(-1))
        if self.feature_concat:
            h = torch.cat((x[edge_index[1]], edge_embeddings), dim=1)
        else:
            h = x[edge_index[1]] + edge_embeddings

        h = self.aggr(h, edge_index, x.size(0))
        h = self.mlp(h)
        return h

    def aggr(self, x, edge_index, num_nodes):
        if self.pooling == "mean":
            return batch_mean_pooling(x, edge_index[0])
        elif self.pooling == "sum":
            return batch_sum_pooling(x, edge_index[0])
        else:
            raise NotImplementedError


class GNN(nn.Module):
    def __init__(
            self,
            num_layers,
            hidden_size,
            JK="last",
            dropout=0.5,
            input_layer=None,
            edge_encode=None,
            edge_emb=None,
            num_atom_type=None,
            num_chirality_tag=None,
            concat=False,
    ):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.JK = JK
        self.atom_type_embedder = num_atom_type
        self.chirality_tag_embedder = num_chirality_tag

        self.gnn = nn.ModuleList()
        if num_atom_type is not None:
            self.atom_type_embedder = torch.nn.Embedding(num_atom_type, hidden_size)
            torch.nn.init.xavier_uniform_(self.atom_type_embedder.weight.data)
        if num_chirality_tag is not None:
            self.chirality_tag_embedder = torch.nn.Embedding(num_chirality_tag, hidden_size)
            torch.nn.init.xavier_uniform_(self.chirality_tag_embedder.weight.data)
        for i in range(num_layers):
            if i == 0:
                self.gnn.append(
                    GINConv(
                        hidden_size=hidden_size,
                        input_layer=input_layer,
                        edge_emb=edge_emb,
                        edge_encode=edge_encode,
                        feature_concat=concat
                    )
                )
            else:
                self.gnn.append(
                    GINConv(
                        hidden_size=hidden_size,
                        edge_emb=edge_emb,
                        edge_encode=edge_encode,
                        feature_concat=True
                    )
                )

    def forward(self, x, edge_index, edge_attr, self_loop_index=None, self_loop_type=None):
        if self.atom_type_embedder is not None and self.chirality_tag_embedder is not None:
            x = self.atom_type_embedder(x[:,0]) + self.chirality_tag_embedder(x[:,1])
        h_list = [x]
        for i in range(self.num_layers):
            h = self.gnn[i](h_list[i], edge_index, edge_attr, self_loop_index, self_loop_type)
            if i == self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            node_rep = h_list[-1]
        elif self.JK == "sum":
            node_rep = sum(h_list[1:])
        else:
            node_rep = torch.cat(h_list, dim=-1)
        return node_rep


class GNNPred(nn.Module):
    def __init__(
            self,
            num_layers,
            hidden_size,
            num_tasks,
            JK="last",
            dropout=0,
            graph_pooling="mean",
            input_layer=None,
            edge_encode=None,
            edge_emb=None,
            num_atom_type=None,
            num_chirality_tag=None,
            concat=True,
    ):
        super(GNNPred, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.JK = JK
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        """
            Bio:    input_layer = 2
                    edge_encode = 9
                    self_loop_index = 7
                    self_loop_type = 1
            Chem:   edge_emb = [num_bond_type, num_bond_direction]
                    self_loop_index = 0
                    self_loop_type = 4
        """

        self.gnn = GNN(
            num_layers=num_layers,
            hidden_size=hidden_size,
            JK=JK,
            dropout=dropout,
            input_layer=input_layer,
            edge_encode=edge_encode,
            edge_emb=edge_emb,
            num_atom_type=num_atom_type,
            num_chirality_tag=num_chirality_tag,
            concat=concat
        )

        self.graph_pred_linear = torch.nn.Linear(2 * self.hidden_size, self.num_tasks)

    def load_from_pretrained(self, path):
        self.gnn.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def forward(self, data, self_loop_index, self_loop_type):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_representation = self.gnn(
            x,
            edge_index,
            edge_attr,
            self_loop_index,
            self_loop_type
        )

        pooled = self.pool(node_representation, batch)
        if hasattr(data, "center_node_idx"):
            center_node_rep = node_representation[data.center_node_idx]

            graph_rep = torch.cat([pooled, center_node_rep], dim=1)
        else:
            graph_rep = torch.cat([pooled, pooled], dim=1)

        return self.graph_pred_linear(graph_rep)

    def pool(self, x, batch):
        if self.graph_pooling == "mean":
            return batch_mean_pooling(x, batch)
        elif self.graph_pooling == "sum":
            return batch_sum_pooling(x, batch)
        else:
            raise NotImplementedError


class Pretrainer(nn.Module):
    def __init__(self, args, transform=None):
        super(Pretrainer, self).__init__()
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.JK = args.JK
        self.weight_decay = args.weight_decay
        self.max_epoch = args.max_epoch
        self.device = torch.device("cpu" if args.cpu else "cuda")
        self.data_type = args.data_type
        self.dataset_name = args.dataset
        self.num_workers = args.num_workers
        self.output_model_file = os.path.join(args.output_model_file, args.pretrain_task)
        self.finetune = args.finetune

        self.dataset, self.opt = self.get_dataset(
            dataset_name=args.dataset,
            transform=transform
        )

        if self.dataset_name in ("bio", "chem", "test_bio", "test_chem", "bbbp", "bace"):
            self.self_loop_index = self.opt["self_loop_index"]
            self.self_loop_type = self.opt["self_loop_type"]

    def get_dataset(self, dataset_name, transform=None):
        assert dataset_name in ("bio", "chem", "test_bio", "test_chem", "bbbp", "bace")
        if dataset_name == "bio":
            dataset = BioDataset(self.data_type, transform=transform)  # BioDataset
            opt = {
                "input_layer": 2,
                "edge_encode": 9,
                "self_loop_index": 7,
                "self_loop_type": 1,
                "concat": True,
            }
        elif dataset_name == "chem":
            dataset = MoleculeDataset(self.data_type, transform=transform)  # MoleculeDataset
            opt = {
                "edge_emb": [6, 3],
                "num_atom_type": 120,
                "num_chirality_tag": 3,
                "self_loop_index": 0,
                "self_loop_type": 4,
                "concat": False,
            }
        elif dataset_name == "test_bio":
            dataset = TestBioDataset(data_type=self.data_type, transform=transform)
            opt = {
                "input_layer": 2,
                "edge_encode": 9,
                "self_loop_index": 0,
                "self_loop_type": 1,
                "concat": True,
            }
        elif dataset_name == "test_chem":
            dataset = TestChemDataset(data_type=self.data_type, transform=transform)
            opt = {
                "edge_emb": [6, 3],
                "num_atom_type": 120,
                "num_chirality_tag": 3,
                "self_loop_index": 0,
                "self_loop_type": 4,
                "concat": False,
            }
        else:
            dataset = build_dataset_from_name(self.dataset_name)
            opt = {
                "edge_emb": [6, 3],
                "num_atom_type": 120,
                "num_chirality_tag": 3,
                "self_loop_index": 0,
                "self_loop_type": 4,
                "concat": False,
            }
        return dataset, opt

    def fit(self):
        print("Start training...")
        train_acc = 0.
        for i in range(self.max_epoch):
            train_loss, train_acc = self._train_step()
            if self.device != "cpu":
                torch.cuda.empty_cache()
            print(f"#epoch {i} : train_loss: {train_loss}, train_acc: {train_acc}")
        if not self.output_model_file == "":
            if not os.path.exists("./saved"):
                os.mkdir("./saved")

            if isinstance(self.model, GNNPred):
                model = self.model.gnn
            else:
                model = self.model
            if self.finetune:
                torch.save(model.state_dict(), self.output_model_file + "_ft.pth")
            else:
                torch.save(model.state_dict(), self.output_model_file + ".pth")

        return dict(Acc=train_acc.item())


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        nn.init.xavier_uniform_(self.weight, gain=1. / math.sqrt(size))

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x * h, dim=1)


class InfoMaxTrainer(Pretrainer):
    @staticmethod
    def add_args(parser):
        pass

    def __init__(self, args):
        args.data_type = "unsupervised"
        super(InfoMaxTrainer, self).__init__(args)
        self.hidden_size = args.hidden_size
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        self.model = GNN(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            JK=args.JK,
            dropout=args.dropout,
            input_layer=self.opt.get("input_layer", None),
            edge_encode=self.opt.get("edge_encode", None),
            edge_emb=self.opt.get("edge_emb", None),
            num_atom_type=self.opt.get("num_atom_type", None),
            num_chirality_tag=self.opt.get("num_chirality_tag", None),
            concat=self.opt["concat"],
        )

        self.discriminator = Discriminator(args.hidden_size)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _train_step(self):
        loss_items = []
        acc_items = []

        self.model.train()
        for batch in self.dataloader:
            batch = batch.to(self.device)
            hidden = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                self_loop_index=self.self_loop_index,
                self_loop_type=self.self_loop_type
            )
            summary_h = torch.sigmoid(batch_mean_pooling(hidden, batch.batch))

            pos_summary = summary_h[batch.batch]
            neg_summary_h = summary_h[cycle_index(summary_h.size(0), 1)]
            neg_summary = neg_summary_h[batch.batch]

            pos_scores = self.discriminator(hidden, pos_summary)
            neg_scores = self.discriminator(hidden, neg_summary)

            self.optimizer.zero_grad()
            loss = self.loss_fn(
                pos_scores,
                torch.ones_like(pos_scores)
            ) + \
                   self.loss_fn(
                       neg_scores,
                       torch.zeros_like(neg_scores)
                   )

            loss.backward()
            self.optimizer.step()

            loss_items.append(loss.item())
            acc_items.append(
                ((pos_scores > 0).float().sum() + (neg_scores < 0).float().sum()) / (pos_scores.shape[0] * 2))
        return sum(loss_items) / len(loss_items), sum(acc_items) / len(acc_items)


class ContextPredictTrainer(Pretrainer):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--mode", type=str, default="cbow", help="cbow or skipgram")
        parser.add_argument("--negative-samples", type=int, default=10)
        parser.add_argument("--center", type=int, default=0)
        parser.add_argument("--l1", type=int, default=1)
        parser.add_argument("--l2", type=int, default=2)

    def __init__(self, args):
        if "bio" in args.dataset:
            transform = ExtractSubstructureContextPair(args.l1, args.center)
        elif "chem" in args.dataset:
            transform = ChemExtractSubstructureContextPair(args.num_layers, args.l1, args.l2)
        args.data_type = "unsupervised"
        super(ContextPredictTrainer, self).__init__(args, transform)
        self.mode = args.mode
        self.context_pooling = "sum"
        self.negative_samples = args.negative_samples % args.batch_size if args.batch_size > args.negative_samples else args.negative_samples

        self.dataloader = DataLoaderSubstructContext(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        self.model = GNN(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            JK=args.JK,
            dropout=args.dropout,
            input_layer=self.opt.get("input_layer", None),
            edge_emb=self.opt.get("edge_emb", None),
            edge_encode=self.opt.get("edge_encode", None),
            num_atom_type=self.opt.get("num_atom_type", None),
            num_chirality_tag=self.opt.get("num_chirality_tag", None),
            concat=self.opt["concat"],
        )

        self.model_context = GNN(
            num_layers=3,
            hidden_size=args.hidden_size,
            JK=args.JK,
            dropout=args.dropout,
            input_layer=self.opt.get("input_layer", None),
            edge_emb=self.opt.get("edge_emb", None),
            edge_encode=self.opt.get("edge_encode", None),
            num_atom_type=self.opt.get("num_atom_type", None),
            num_chirality_tag=self.opt.get("num_chirality_tag", None),
            concat=self.opt["concat"],
        )
        self.model.to(self.device)
        self.model_context.to(self.device)
        self.optimizer_neighbor = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.optimizer_context = torch.optim.Adam(
            self.model_context.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def _train_step(self):
        loss_items = []
        acc_items = []
        self.model.train()
        self.model_context.train()
        for batch in self.dataloader:
            batch = batch.to(self.device)
            neighbor_rep = self.model(
                batch.x_substruct,
                batch.edge_index_substruct,
                batch.edge_attr_substruct,
                self.self_loop_index,
                self.self_loop_type
            )[batch.center_substruct_idx]
            overlapped_node_rep = self.model_context(
                batch.x_context,
                batch.edge_index_context,
                batch.edge_attr_context,
                self.self_loop_index,
                self.self_loop_type
            )[batch.overlap_context_substruct_idx]
            if self.mode == "cbow":
                pos_scores, neg_scores = self.get_cbow_pred(
                    overlapped_node_rep,
                    batch.batch_overlapped_context,
                    neighbor_rep
                )
            else:
                pos_scores, neg_scores = self.get_skipgram_pred(
                    overlapped_node_rep,
                    batch.overlapped_context_size,
                    neighbor_rep
                )
            self.optimizer_neighbor.zero_grad()
            self.optimizer_context.zero_grad()

            pos_loss = self.loss_fn(pos_scores.double(), torch.ones_like(pos_scores).double())
            neg_loss = self.loss_fn(neg_scores.double(), torch.zeros_like(neg_scores).double())
            loss = pos_loss + self.negative_samples * neg_loss
            loss.backward()

            self.optimizer_neighbor.step()
            self.optimizer_context.step()

            loss_items.append(loss.item())
            acc_items.append(
                ((pos_scores > 0).float().sum() + (neg_scores < 0).float().sum() / self.negative_samples) / (
                            pos_scores.shape[0] * 2)
            )
        return sum(loss_items) / len(loss_items), sum(acc_items) / len(acc_items)

    def get_cbow_pred(
            self,
            overlapped_rep,
            overlapped_context,
            neighbor_rep
    ):
        if self.context_pooling == "sum":
            context_rep = batch_sum_pooling(overlapped_rep, overlapped_context)
        elif self.context_pooling == "mean":
            context_rep = batch_mean_pooling(overlapped_rep, overlapped_context)
        else:
            raise NotImplementedError

        batch_size = context_rep.size(0)

        neg_context_rep = torch.cat(
            [
                context_rep[cycle_index(batch_size, i + 1)]
                for i in range(self.negative_samples)
            ],
            dim=0
        )

        pos_scores = torch.sum(neighbor_rep * context_rep, dim=1)
        neg_scores = torch.sum(
            neighbor_rep.repeat(self.negative_samples, 1) * neg_context_rep,
            dim=1
        )
        return pos_scores, neg_scores

    def get_skipgram_pred(
            self,
            overlapped_rep,
            overlapped_context_size,
            neighbor_rep
    ):
        expanded_neighbor_rep = torch.cat(
            [
                neighbor_rep[i].repeat(overlapped_context_size[i], 1)
                for i in range(len(neighbor_rep))
            ],
            dim=0
        )
        assert overlapped_rep.shape == expanded_neighbor_rep.shape
        pos_scores = torch.sum(expanded_neighbor_rep * overlapped_rep, dim=1)

        batch_size = neighbor_rep.size(0)
        neg_scores = []
        for i in range(self.negative_samples):
            neg_neighbor_rep = neighbor_rep[cycle_index(batch_size, i + 1)]
            expanded_neg_neighbor_rep = torch.cat(
                [
                    neg_neighbor_rep[i].repeat(overlapped_context_size[k], 1)
                    for k in range(len(neg_neighbor_rep))
                ],
                dim=0
            )
            neg_scores.append(
                torch.sum(expanded_neg_neighbor_rep * overlapped_rep, dim=1)
            )
        neg_scores = torch.cat(neg_scores)
        return pos_scores, neg_scores


class MaskTrainer(Pretrainer):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--mask-rate", type=float, default=0.15)

    def __init__(self, args):
        if "bio" in args.dataset:
            transform = MaskEdge(mask_rate=args.mask_rate)
        elif "chem" in args.dataset:
            transform = MaskAtom(mask_rate=args.mask_rate, num_atom_type=119, num_edge_type=5)
        args.data_type = "unsupervised"
        super(MaskTrainer, self).__init__(args, transform)
        self.dataloader = DataLoaderMasking(
            self.dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        self.model = GNN(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            JK=args.JK,
            dropout=args.dropout,
            input_layer=self.opt.get("input_layer", None),
            edge_encode=self.opt.get("edge_encode", None),
            edge_emb=self.opt.get("edge_emb", None),
            num_atom_type=self.opt.get("num_atom_type", None),
            num_chirality_tag=self.opt.get("num_chirality_tag", None),
            concat=self.opt["concat"],
        )

        edge_attr_dim = self.dataset[0].edge_attr.size(1)
        if "bio" in args.dataset:
            self.linear = nn.Linear(args.hidden_size, edge_attr_dim)
        elif "chem" in args.dataset:
            self.linear = nn.Linear(args.hidden_size, 120)
        else:
            raise NotImplementedError
        self.optmizer_gnn = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.optmizer_linear = torch.optim.Adam(
            self.linear.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        if "chem" in args.dataset:
            self.linear_bonds = torch.nn.Linear(args.hidden_size, 6)
            self.optmizer_linear_bonds = torch.optim.Adam(
                self.linear_bonds.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

        self.loss_fn = nn.CrossEntropyLoss()

    def _train_step(self):
        loss_items = []
        acc_items = []

        for batch in self.dataloader:
            batch = batch.to(self.device)
            hidden = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                self_loop_index=self.self_loop_index,
                self_loop_type=self.self_loop_type
            )
            if "bio" in self.dataset_name:
                masked_edges = batch.edge_index[:, batch.masked_edge_idx]
                masked_edges_rep = hidden[masked_edges[0]] + hidden[masked_edges[1]]
                pred = self.linear(masked_edges_rep)
                labels = torch.argmax(batch.mask_edge_label, dim=1)

                self.optmizer_gnn.zero_grad()
                self.optmizer_linear.zero_grad()

                loss = self.loss_fn(pred, labels)
                loss.backward()
                self.optmizer_gnn.step()
                self.optmizer_linear.step()
                loss_items.append(loss.item())
                acc_items.append(
                    (torch.max(pred, dim=1)[1] == labels).float().sum().cpu().numpy() / len(pred)
                )
            elif "chem" in self.dataset_name:
                def compute_accuracy(pred, target):
                    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)
                pred_node = self.linear(hidden[batch.masked_atom_indices])
                loss = self.loss_fn(pred_node.double(), batch.mask_node_label[:,0])
                acc_node = compute_accuracy(pred_node, batch.mask_node_label[:,0])
                masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
                edge_rep = hidden[masked_edge_index[0]] + hidden[masked_edge_index[1]]
                pred_edge = self.linear_bonds(edge_rep)
                loss += self.loss_fn(pred_edge.double(), batch.mask_edge_label[:,0])
                acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:,0])
                self.optmizer_gnn.zero_grad()
                self.optmizer_linear.zero_grad()
                self.optmizer_linear_bonds.zero_grad()
                loss.backward()
                self.optmizer_gnn.step()
                self.optmizer_linear.step()
                self.optmizer_linear_bonds.step()
                loss_items.append(loss.item())
                acc_items.extend([acc_node, acc_edge])
            else:
                raise NotImplementedError
        return np.mean(loss_items), np.mean(acc_items)


class SupervisedTrainer(Pretrainer):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--pooling", type=str, default="mean")
        parser.add_argument("--checkpoint", type=str, default=None)

    def __init__(self, args):
        args.data_type = "supervised"
        super(SupervisedTrainer, self).__init__(args)
        self.dataloader = self.split_data()
        if "bio" in args.dataset:
            num_tasks = len(self.dataset[0].go_target_downstream)
        elif "chem" in args.dataset:
            num_tasks = len(self.dataset[0].y)
        self.model = GNNPred(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_tasks=num_tasks,
            JK=args.JK,
            dropout=args.dropout,
            graph_pooling=args.pooling,
            input_layer=self.opt.get("input_layer", None),
            edge_emb=self.opt.get("edge_emb", None),
            edge_encode=self.opt.get("edge_encode", None),
            num_atom_type=self.opt.get("num_atom_type", None),
            num_chirality_tag=self.opt.get("num_chirality_tag", None),
            concat=self.opt["concat"],
        )
        if args.checkpoint is not None:
            self.model.load_from_pretrained(args.checkpoint)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def split_data(self):
        length = len(self.dataset)
        indices = np.arange(length)
        np.random.shuffle(indices)
        self.train_ratio = 0.9
        train_index = torch.LongTensor(indices[: int(length * self.train_ratio)])
        dataset = self.dataset[train_index]
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return dataloader

    def _train_step(self):
        loss_items = []
        auc_items = []

        self.model.train()
        for batch in self.dataloader:
            batch = batch.to(self.device)
            pred = self.model(
                batch,
                self.self_loop_index,
                self.self_loop_type
            )
            self.optimizer.zero_grad()
            if "bio" in self.dataset_name:
                target = batch.go_target_pretrain.view(pred.shape).to(torch.float64)
                is_valid = (target != -1)
            elif "chem" in self.dataset_name:
                target = batch.y.view(pred.shape).to(torch.float64)
                is_valid = target**2 > 0
                target = (target + 1) / 2
            else:
                raise NotImplementedError
            
            loss = self.loss_fn(pred, target)
            loss = torch.where(is_valid, loss, torch.zeros(loss.shape).to(loss.device).to(loss.dtype))
            loss = torch.sum(loss) / torch.sum(is_valid)
            loss.backward()
            self.optimizer.step()
            loss_items.append(loss.item())

            with torch.no_grad():
                pred = pred.cpu().detach().numpy()
                if "bio" in self.dataset_name:
                    y_labels = batch.go_target_pretrain.view(pred.shape).cpu().numpy()
                elif "chem" in self.dataset_name:
                    y_labels = batch.y.view(pred.shape).cpu().numpy()
                auc_scores = []
                for i in range(len(pred[0])):
                    if "chem" in self.dataset_name:
                        is_valid = y_labels[:,i]**2 > 0
                        y_labels[:,i] = (y_labels[:,i] + 1) / 2
                    elif "bio" in self.dataset_name:
                        is_valid = y_labels[:,i] != -1
                    else:
                        raise NotImplementedError
                    if (y_labels[is_valid, i] == 1).sum() > 0 and (y_labels[is_valid, i] == 0).sum() > 0:
                        auc_scores.append(roc_auc_score(y_labels[is_valid, i], pred[is_valid, i]))
                    else:
                        # All zeros or all ones
                        auc_scores.append(np.nan)
                auc_scores = np.array(auc_scores)
                auc_items.append(
                    np.mean(auc_scores[np.where(~np.isnan(auc_scores))])
                )
        return np.mean(loss_items), np.mean(auc_items)


class Finetuner(Pretrainer):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--checkpoint", type=str, default="./saved")
        parser.add_argument("--pooling", type=str, default="mean")

    def __init__(self, args):
        args.data_type = "supervised"
        super(Finetuner, self).__init__(args)
        self.model_file = args.checkpoint
        self.patience = args.patience
        self.train_ratio = 0.8
        self.valid_ratio = 0.1
        self.test_ratio = 0.1
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self.split_data()

        self.model = self.build_model(args)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def build_model(self, args):
        if "bio" in args.dataset:
            num_tasks = len(self.dataset[0].go_target_downstream)
        elif "bbbp" == args.dataset or "bace" == args.dataset:
            num_tasks = 1

        model = GNNPred(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_tasks=num_tasks,
            JK=self.JK,
            dropout=args.dropout,
            graph_pooling=args.pooling,
            input_layer=self.opt.get("input_layer", None),
            edge_emb=self.opt.get("edge_emb", None),
            edge_encode=self.opt.get("edge_encode", None),
            num_atom_type=self.opt.get("num_atom_type", None),
            num_chirality_tag=self.opt.get("num_chirality_tag", None),
            concat=self.opt["concat"],
        )
        model.load_from_pretrained(args.checkpoint)
        return model

    def split_data(self):
        length = len(self.dataset)
        indices = np.arange(length)
        np.random.shuffle(indices)
        train_index = torch.LongTensor(indices[: int(length * self.train_ratio)])
        valid_index = torch.LongTensor(indices[int(length * self.train_ratio): -int(length * self.test_ratio)])
        test_index = torch.LongTensor(indices[-int(length * self.test_ratio):])

        datasets = [self.dataset[train_index], self.dataset[valid_index], self.dataset[test_index]]
        dataloaders = [
            DataLoaderFinetune(
                dataset=item,
                batch_size=self.batch_size * (1 + int(idx == 0) * 9),
                shuffle=(idx == 0),
                num_workers=self.num_workers
            )
            for idx, item in enumerate(datasets)
        ]
        return dataloaders

    def _train_step(self):
        self.model.train()
        for batch in self.train_loader:
            batch = batch.to(self.device)
            pred = self.model(batch, self.self_loop_index, self.self_loop_type)
            if "bio" in self.dataset_name:
                labels = batch.go_target_downstream.view(pred.shape).to(torch.float64)
                is_valid = labels != -1
            else:
                labels = batch.y.view(pred.shape).to(torch.float64)
                is_valid = labels ** 2 > 0
                labels = (labels + 1) / 2

            self.optimizer.zero_grad()
            loss = self.loss_fn(pred, labels)
            loss = torch.where(is_valid, loss, torch.zeros(loss.shape).to(loss.device).to(loss.dtype))
            loss = torch.sum(loss) / torch.sum(is_valid)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def _test_step(self, split="val"):
        self.model.eval()
        if split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        else:
            loader = self.train_loader

        y_pred = []
        y_labels = []
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch, self.self_loop_index, self.self_loop_type)
            y_pred.append(pred)
            if "bio" in self.dataset_name:
                target = batch.go_target_downstream.view(pred.shape)
            else:
                target =  batch.y.view(pred.shape)
            y_labels.append(target)
        y_pred = torch.cat(y_pred, dim=0)
        y_labels = torch.cat(y_labels, dim=0)

        loss = self.loss_fn(y_pred, y_labels.to(torch.float64))
        y_pred = y_pred.cpu().numpy()
        y_labels = y_labels.cpu().numpy()
        auc_scores = []
        for i in range(len(y_pred[1])):
            if "bio" in self.dataset_name:
                is_valid = y_labels[:,i] != -1
            else:
                is_valid = y_labels[:,i]**2 > 0
                y_labels[:,i] = (y_labels[:,i] + 1) / 2
            if (y_labels[is_valid, i] == 1).sum() > 0 and (y_labels[is_valid, i] == 0).sum() > 0:
                auc_scores.append(roc_auc_score(y_labels[:, i], y_pred[:, i]))
            else:
                # All zeros or all ones
                auc_scores.append(np.nan)
        auc_scores = np.array(auc_scores)
        return np.mean(auc_scores[np.where(~np.isnan(auc_scores))]), loss.item()

    def fit(self):
        best_loss = 100000.
        best_model = None
        patience = 0
        for epoch in range(self.max_epoch):
            self._train_step()
            val_auc, val_loss = self._test_step(split="val")
            print(f"#epoch {epoch}: val_loss: {val_loss}, val_auc: {val_auc}, best_loss: {best_loss}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(self.model)
                patience = 0
            else:
                patience += 1
                if patience > self.patience:
                    break
        self.model = best_model
        test_auc, test_loss = self._test_step(split="test")
        print(f"Test auc:{test_auc}, test loss: {test_loss}")
        return dict(Acc=test_auc)
