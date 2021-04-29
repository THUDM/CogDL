import torch
from cogdl.data import Graph
from cogdl.datasets import BaseDataset, register_dataset, build_dataset, build_dataset_from_name
from cogdl.utils import build_args_from_dict


@register_dataset("mydataset")
class MyNodeClassificationDataset(BaseDataset):
    def __init__(self):
        super(MyNodeClassificationDataset, self).__init__()
        self.data = self.process()

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
        torch.save(data, "mydata.pt")
        return data


def test_customized_dataset():
    dataset = build_dataset_from_name("mydataset")
    assert isinstance(dataset[0], Graph)
    assert dataset[0].x.shape[0] == 100


def test_build_dataset_from_path():
    args = build_args_from_dict({"dataset": "mydata.pt", "task": "node_classification"})
    dataset = build_dataset(args)
    assert dataset[0].x.shape[0] == 100


if __name__ == "__main__":
    test_customized_dataset()
