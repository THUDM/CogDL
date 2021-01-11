import torch.nn as nn
import torch.nn.functional as F

from cogdl.datasets import build_dataset
from cogdl.tasks import build_task
from cogdl.utils import get_example_args
from cogdl.models import BaseModel
from cogdl.models.nn.gcn import GraphConvolution


"""Define your Model"""


class MyModel(BaseModel):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(MyModel, self).__init__()
        self.layer = GraphConvolution(in_features=in_feats, out_features=hidden_size)
        self.fc = nn.Linear(hidden_size, out_feats)

    def forward(self, x, edge_index):
        h = self.layer(x, edge_index)
        return F.log_softmax(self.fc(h))

    def node_classification_loss(self, data):
        out = self.forward(data.x, data.edge_index)[data.train_mask]
        loss_n = F.nll_loss(out, data.y[data.train_mask])
        return loss_n

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


def set_args(args):
    """Set parameters for your model and task"""
    args.model = "my_model"
    args.dataset = "cora"
    args.hidden_size = 32
    return args


def main_model():
    # Set the task
    args = get_example_args(task="node_classification")
    args = set_args(args)
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
