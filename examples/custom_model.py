import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.data import Data
from cogdl.datasets import build_dataset
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict
from cogdl.models.nn.gcn import GraphConvolution


"""Define your Model"""
class MyModel(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(MyModel, self).__init__()
        self.layer = GraphConvolution(in_features=in_feats, out_features=hidden_size)
        self.fc = nn.Linear(hidden_size, out_feats)
    
    def forward(self, x, edge_index):
        h = self.layer(x, edge_index)
        return F.log_softmax(self.fc(h))

    def loss(self, data):
        out = self.forward(data.x, data.edge_index)[data.train_mask]
        loss_n = F.nll_loss(out, data.y[data.train_mask])
        return loss_n
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)


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


def main_model():
    args = get_default_args()
    # Set the task
    args.task = "node_classification"
    args.dataset = "cora"
    # use dataset in cogdl
    dataset = build_dataset(args)
    hidden_size = args.hidden_size
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    # use customized model
    model = MyModel(num_features, hidden_size, num_classes)

    task = build_task(args, dataset, model)
    result = task.train()
    print(result)


if __name__ == "__main__":
    main_model()
