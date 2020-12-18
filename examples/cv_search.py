import torch
import random
import numpy as np
from itertools import product

from cogdl.datasets import build_dataset
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_dataset(args):
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    return dataset, args


def build_default_args():
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.01,
        "weight_decay": 5e-4,
        "max_epoch": 500,
        "patience": 100,
        "cpu": cpu,
        "device_id": [0],
        "seed": [0, 1, 2],
        "dropout": 0.5,
        "hidden_size": 128,
        "num_layers": 2,
        "sample_size": [10, 10],
        "task": "node_classification",
        "model": "gcn",
        "dataset": "cora",
    }
    return build_args_from_dict(args)


def get_parameters():
    scope = {"epoch": [10, 20, 30], "lr": [0.01, 0.001, 0.0001]}
    keys = list(scope.keys())
    values = list(scope.values())
    combination = product(*values)
    return [{keys[i]: val[i] for i in range(len(keys))} for val in combination]


def train():
    args = build_default_args()
    dataset, args = get_dataset(args)

    combinations = get_parameters()
    for item in combinations:
        for key, val in item.items():
            setattr(args, key, val)
        task = build_task(args, dataset=dataset)
        res = task.train()
        print(item, res)


if __name__ == "__main__":
    train()
