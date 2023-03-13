import numpy as np
import jittor
from jittor import nn
from cogdl.layers import SELayer, GCNLayer, GATLayer
from cogdl.datasets.planetoid_data import CoraDataset, CiteSeerDataset
from cogdl.models import BaseModel
from cogdl.datasets.customized_data import NodeDataset, generate_random_graph


class DrGAT(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=8)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.6)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.nhead,
            args.dropout,
        )

    def __init__(self, num_features, num_classes, hidden_size, num_heads, dropout):
        super(DrGAT, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.conv1 = GATLayer(num_features, hidden_size, nhead=num_heads, attn_drop=dropout)
        self.conv2 = GATLayer(hidden_size * num_heads, num_classes, nhead=2, attn_drop=dropout)
        self.se1 = SELayer(num_features, se_channels=int(np.sqrt(num_features)))
        self.se2 = SELayer(hidden_size * num_heads, se_channels=int(np.sqrt(hidden_size * num_heads)))

    def execute(self, graph):
        x = graph.x
        x = nn.dropout(x, p=self.dropout, is_train=self.is_train)
        x = self.se1(x)
        x = nn.elu(self.conv1(graph, x))
        x = nn.dropout(x, p=self.dropout, is_train=self.is_train)
        x = self.se2(x)
        x = nn.elu(self.conv2(graph, x))
        return x


def train(model, dataset):
    graph = dataset[0]

    optimizer = nn.AdamW(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()

    train_mask = graph.train_mask
    test_mask = graph.test_mask
    val_mask = graph.val_mask
    labels = graph.y

    for epoch in range(500):
        model.train()
        output = model(graph)
        loss = loss_function(output[train_mask], labels[train_mask])
        optimizer.step(loss)

        model.eval()
        with jittor.no_grad():
            pred = model(graph)
            pred = pred.argmax(1)[0]
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        print(
            f"Epoch:{epoch}, loss:{loss:.3f},train_acc:{train_acc:.3f}, val_acc:{val_acc:.3f}, test_acc:{test_acc:.3f}"
        )


if __name__ == "__main__":
    dataset = CoraDataset()
    # data = generate_random_graph(num_nodes=100, num_edges=300, num_feats=30)
    # dataset = NodeDataset(data=data)
    dataset.data.add_remaining_self_loops()
    model = DrGAT(
        num_features=dataset.num_features, hidden_size=8, num_classes=dataset.num_classes, num_heads=8, dropout=0.6
    )

    train(model, dataset)
