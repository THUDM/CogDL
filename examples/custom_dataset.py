from cogdl.data.data import Data
import torch

from cogdl.tasks import build_task
from cogdl.models import build_model
from cogdl.options import get_task_model_args


"""Define your data"""


class MyData(Data):
    def __init__(self):
        super(MyData, self).__init__()
        num_nodes = 100
        num_edges = 300
        feat_dim = 30
        # load or generate data
        self.edge_index = torch.randint(0, num_nodes, (2, num_edges))
        self.x = torch.randn(num_nodes, feat_dim)
        self.y = torch.randint(0, 2, (num_nodes,))

        # set train/val/test mask in node_classification task
        self.train_mask = torch.zeros(num_nodes).bool()
        self.train_mask[0 : int(0.3 * num_nodes)] = True
        self.val_mask = torch.zeros(num_nodes).bool()
        self.val_mask[int(0.3 * num_nodes) : int(0.7 * num_nodes)] = True
        self.test_mask = torch.zeros(num_nodes).bool()
        self.test_mask[int(0.7 * num_nodes) :] = True


"""Define your dataset"""


class MyNodeClassificationDataset(object):
    def __init__(self):
        self.data = MyData()
        self.num_classes = self.data.num_classes
        self.num_features = self.data.num_features

    def __getitem__(self, index):
        assert index == 0
        return self.data


def set_args(args):
    """Change default setttings"""
    cuda_available = torch.cuda.is_available()
    args.cpu = not cuda_available
    return args


def main_dataset():
    args = get_task_model_args(task="node_classification", model="gcn")
    # use customized dataset
    dataset = MyNodeClassificationDataset()
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    # use model in cogdl
    model = build_model(args)
    task = build_task(args, dataset, model)
    result = task.train()
    print(result)


if __name__ == "__main__":
    main_dataset()
