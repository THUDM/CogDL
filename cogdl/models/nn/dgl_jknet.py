import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from tqdm import tqdm

from cogdl.models.supervised_model import SupervisedHomogeneousNodeClassificationModel
from cogdl.trainers.supervised_model_trainer import SupervisedHomogeneousNodeClassificationTrainer
from .. import register_model


class GraphConvLayer(torch.nn.Module):
    """Graph convolution layer.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """

    AGGREGATIONS = {
        "sum": torch.sum,
        "mean": torch.mean,
        "max": torch.max,
    }

    def __init__(self, in_features, out_features, aggregation="sum"):
        super(GraphConvLayer, self).__init__()

        if aggregation not in self.AGGREGATIONS.keys():
            raise ValueError("'aggregation' argument has to be one of " "'sum', 'mean' or 'max'.")
        self.aggregate = lambda nodes: self.AGGREGATIONS[aggregation](nodes, dim=1)

        self.linear = torch.nn.Linear(in_features, out_features)
        self.self_loop_w = torch.nn.Linear(in_features, out_features)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, graph, x):
        graph.ndata["h"] = x
        graph.update_all(fn.copy_src(src="h", out="msg"), lambda nodes: {"h": self.aggregate(nodes.mailbox["msg"])})
        h = graph.ndata.pop("h")
        h = self.linear(h)
        return h + self.self_loop_w(x) + self.bias


class JKNetConcat(torch.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with concatenation.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the middle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """

    def __init__(self, in_features, out_features, n_layers=6, n_units=16, aggregation="sum"):
        super(JKNetConcat, self).__init__()
        self.n_layers = n_layers

        self.gconv0 = GraphConvLayer(in_features, n_units, aggregation)
        self.dropout0 = torch.nn.Dropout(0.5)
        for i in range(1, self.n_layers):
            setattr(self, "gconv{}".format(i), GraphConvLayer(n_units, n_units, aggregation))
            setattr(self, "dropout{}".format(i), torch.nn.Dropout(0.5))
        self.last_linear = torch.nn.Linear(n_layers * n_units, out_features)

    def forward(self, graph, x):
        layer_outputs = []
        for i in range(self.n_layers):
            dropout = getattr(self, "dropout{}".format(i))
            gconv = getattr(self, "gconv{}".format(i))
            x = dropout(F.relu(gconv(graph, x)))
            layer_outputs.append(x)

        h = torch.cat(layer_outputs, dim=1)
        return self.last_linear(h)


class JKNetMaxpool(torch.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with Maxpool.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the middle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """

    def __init__(self, in_features, out_features, n_layers=6, n_units=16, aggregation="sum"):
        super(JKNetMaxpool, self).__init__()
        self.n_layers = n_layers

        self.gconv0 = GraphConvLayer(in_features, n_units, aggregation)
        self.dropout0 = torch.nn.Dropout(0.5)
        for i in range(1, self.n_layers):
            setattr(self, "gconv{}".format(i), GraphConvLayer(n_units, n_units, aggregation))
            setattr(self, "dropout{}".format(i), torch.nn.Dropout(0.5))
        self.last_linear = torch.nn.Linear(n_units, out_features)

    def forward(self, graph, x):
        layer_outputs = []
        for i in range(self.n_layers):
            dropout = getattr(self, "dropout{}".format(i))
            gconv = getattr(self, "gconv{}".format(i))
            x = dropout(F.relu(gconv(graph, x)))
            layer_outputs.append(x)

        h = torch.stack(layer_outputs, dim=0)
        h = torch.max(h, dim=0)[0]
        return self.last_linear(h)


class JKNetTrainer(SupervisedHomogeneousNodeClassificationTrainer):
    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super(JKNetTrainer, self).__init__()
        self.graph = dgl.DGLGraph()
        self.args = args

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.loss(self.data).backward()
        self.optimizer.step()

    def _test_step(self, split="val", logits=None):
        self.model.eval()
        logits = logits if logits else self.model.predict(self.data)
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        loss = F.nll_loss(logits[mask], self.data.y[mask]).item()

        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc, loss

    def fit(self, model: SupervisedHomogeneousNodeClassificationModel, dataset):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        device = self.args.device_id[0] if not self.args.cpu else "cpu"
        data = dataset[0]
        data.apply(lambda x: x.to(device))
        self.max_epoch = self.args.max_epoch

        row, col = data.edge_index
        row, col = row.cpu().numpy(), col.cpu().numpy()
        num_edge = row.shape[0]
        num_node = data.x.to("cpu").shape[0]
        self.graph.add_nodes(num_node)
        for i in range(num_edge):
            src, dst = row[i], col[i]
            self.graph.add_edge(src, dst)
        self.graph = self.graph.to(device)
        model.set_graph(self.graph)

        self.data = data
        self.model = model.to(device)

        epoch_iter = tqdm(range(self.max_epoch))
        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        for epoch in epoch_iter:
            self._train_step()
            train_acc, _ = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
            if val_loss <= min_loss or val_acc >= max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_score = val_acc
                min_loss = np.min((min_loss, val_loss))
                max_score = np.max((max_score, val_acc))

        print(f"Best accurracy = {best_score}")

        test_acc, _ = self._test_step(split="test")
        print(f"Test accuracy = {test_acc}")
        return dict(Acc=test_acc)


@register_model("jknet")
class JKNet(SupervisedHomogeneousNodeClassificationModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--lr',
                            help='Learning rate',
                            type=float, default=0.005)
        parser.add_argument('--layer-aggregation',
                            help='The way to aggregate outputs of layers',
                            type=str, choices=('maxpool', 'concat'),
                            default='maxpool')
        parser.add_argument('--weight-decay',
                            help='Weight decay',
                            type=float, default=0.0005)
        parser.add_argument('--node-aggregation',
                            help='The way to aggregate neighbourhoods',
                            type=str, choices=('sum', 'mean', 'max'),
                            default='sum')
        parser.add_argument('--n-layers',
                            help='Number of convolution layers',
                            type=int, default=6)
        parser.add_argument('--n-units',
                            help='Size of middle layers.',
                            type=int, default=16)
        parser.add_argument('--in-features',
                            help='Input feature dimension, 1433 for cora',
                            type=int, default=1433)
        parser.add_argument('--out-features',
                            help='Output feature dimension, 7 for cora',
                            type=int, default=7)
        parser.add_argument('--max-epoch',
                            help='Epochs to train',
                            type=int, default=100)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.in_features,
            args.out_features,
            args.n_layers,
            args.n_units,
            args.node_aggregation,
            args.layer_aggregation,
        )

    def __init__(self, in_features, out_features, n_layers, n_units, node_aggregation, layer_aggregation):
        model_args = (in_features, out_features, n_layers, n_units, node_aggregation)
        super(JKNet, self).__init__()
        if layer_aggregation == "maxpool":
            self.model = JKNetMaxpool(*model_args)
        else:
            self.model = JKNetConcat(*model_args)

    def forward(self, graph, x):
        y = F.log_softmax(self.model(graph, x), dim=1)
        return y

    def predict(self, data):
        return self.forward(self.graph, data.x)

    def loss(self, data):
        return F.nll_loss(self.forward(self.graph, data.x)[data.train_mask], data.y[data.train_mask])

    def set_graph(self, graph):
        self.graph = graph

    @staticmethod
    def get_trainer(args):
        return JKNetTrainer
