import torch

from cogdl import experiment
from cogdl.data import Graph
from cogdl.datasets import NodeDataset, register_dataset


@register_dataset("mydataset")
class MyNodeClassificationDataset(NodeDataset):
    def __init__(self):
        self.path = "mydata.pt"
        super(MyNodeClassificationDataset, self).__init__(self.path)

    def process(self):
        num_nodes = 100
        num_edges = 300
        feat_dim = 30

        # load or generate your dataset
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, feat_dim)
        y = torch.randint(0, 2, (num_nodes,))

        # set train/val/test mask in node_classification task
        train_mask = torch.zeros(num_nodes).bool()
        train_mask[0 : int(0.3 * num_nodes)] = True
        val_mask = torch.zeros(num_nodes).bool()
        val_mask[int(0.3 * num_nodes) : int(0.7 * num_nodes)] = True
        test_mask = torch.zeros(num_nodes).bool()
        test_mask[int(0.7 * num_nodes) :] = True
        data = Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        torch.save(data, self.path)
        return data


if __name__ == "__main__":
    # Run with self-loaded dataset
    experiment(task="node_classification", dataset="mydataset", model="gcn")
    # Run with given datapaath
    experiment(task="node_classification", dataset="./mydata.pt", model="gcn")
