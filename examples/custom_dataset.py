from cogdl.data.data import Data
import torch

from cogdl.tasks import build_task
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict
from cogdl.data import Dataset


"""Define your data"""
class MyData(object):
    def __init__(self):
        num_nodes = 100
        num_edges = 300
        feat_dim = 30
        # load or generate data
        self.edge_index = torch.randint(0, num_nodes, (2, num_edges))
        self.x = torch.randn(num_nodes, feat_dim)
        self.y = torch.randint(0, 2, (num_nodes,))

        # set train/val/test mask in node_classification task
        self.train_mask = torch.zeros(num_nodes).bool()
        self.train_mask[0:int(0.3*num_nodes)] = True
        self.val_mask = torch.zeros(num_nodes).bool()
        self.val_mask[int(0.3*num_nodes):int(0.7*num_nodes)] = True
        self.test_mask = torch.zeros(num_nodes).bool()
        self.test_mask[int(0.7*num_nodes):] = True

    def apply(self, func):
        for name, value in vars(self).items():
            setattr(self, name, func(value))

    @property
    def num_features(self):
        return self.x.shape[1]
    
    @property
    def num_classes(self):
        return int(torch.max(self.y)) + 1


"""Define your dataset"""
class MyDataset(object):
    def __init__(self, datalist):
        self.datalist = datalist
        self.num_features = self.datalist[0].num_features
        self.num_classes = self.datalist[0].num_classes

    def __getitem__(self, index):
        assert index == 0
        return self.datalist[index]



def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {'hidden_size': 16,
                    'dropout': 0.5,
                    'patience': 100,
                    'max_epoch': 500,
                    'cpu': not cuda_available,
                    'lr': 0.01,
                    'weight_decay': 5e-4}
    return build_args_from_dict(default_dict)


def main_dataset():
    args = get_default_args()
    args.task = "node_classification"
    args.model = "gcn"
    # use customized dataset
    mydata = MyData()
    dataset = MyDataset([mydata])
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    # use model in cogdl
    model = build_model(args)
    task = build_task(args, dataset, model)
    result = task.train()
    print(result)


if __name__ == "__main__":
    main_dataset()