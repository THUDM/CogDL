import torch
import random
import numpy as np

from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict


def get_default_args():
    cuda_available = torch.cuda.is_available()
    args = {
        "dataset": "cora",
        "trainer": ["self_auxiliary_task"],
        "model": "gcn",
        "hidden_size": 64,
        "dropout": 0.5,
        "patience": 2,
        "device_id": [0],
        "max_epoch": 3,
        "sampler": "none",
        "cpu": not cuda_available,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "missing_rate": -1,
        "task": "node_classification",
        "checkpoint": False,
        "label_mask": 0,
        "num_layers": 2,
    }
    args = build_args_from_dict(args)
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    return args


def test_edgemask():
    args = get_default_args()
    args.auxiliary_task = "edgemask"
    args.alpha = 1
    dataset = build_dataset(args)
    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_pairwise_distance():
    args = get_default_args()
    args.auxiliary_task = "pairwise-distance"
    args.alpha = 35
    dataset = build_dataset(args)
    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_distance_to_clusters():
    args = get_default_args()
    args.auxiliary_task = "distance2clusters"
    args.alpha = 3
    dataset = build_dataset(args)
    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_pairwise_attr_sim():
    args = get_default_args()
    args.auxiliary_task = "pairwise-attr-sim"
    args.alpha = 100
    dataset = build_dataset(args)
    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_distance_to_clustersPP():
    args = get_default_args()
    args.auxiliary_task = "distance2clusters++"
    args.alpha = 1
    dataset = build_dataset(args)
    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


if __name__ == "__main__":
    test_edgemask()
    test_pairwise_distance()
    test_distance_to_clusters()
    test_pairwise_attr_sim()
    test_distance_to_clustersPP()
