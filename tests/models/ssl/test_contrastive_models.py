import numpy as np
import torch

from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def get_default_args():
    default_dict = {
        "hidden_size": 16,
        "num_shuffle": 1,
        "cpu": True,
        "enhance": None,
        "save_dir": "./embedding",
        "task": "unsupervised_node_classification",
        "checkpoint": False,
        "load_emb_path": None,
        "training_percents": [0.1],
        "activation": "relu",
        "residual": False,
        "sampling": False,
        "sample_size": 20,
        "norm": None,
    }
    return build_args_from_dict(default_dict)


def get_unsupervised_nn_args():
    default_dict = {
        "hidden_size": 16,
        "num_layers": 2,
        "lr": 0.01,
        "dropout": 0.0,
        "patience": 1,
        "max_epoch": 1,
        "cpu": not torch.cuda.is_available(),
        "weight_decay": 5e-4,
        "num_shuffle": 2,
        "save_dir": "./embedding",
        "enhance": None,
        "device_id": [
            0,
        ],
        "task": "unsupervised_node_classification",
        "checkpoint": False,
        "load_emb_path": None,
        "sampling": False,
        "sample_size": 20,
        "training_percents": [0.1],
    }
    return build_args_from_dict(default_dict)


def build_nn_dataset(args):
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    return args, dataset


def test_unsupervised_graphsage():
    args = get_unsupervised_nn_args()
    args.negative_samples = 10
    args.walk_length = 5
    args.sample_size = [5, 5]
    args.patience = 20
    args.task = "unsupervised_node_classification"
    args.dataset = "cora"
    args.max_epochs = 2
    args.save_model = "graphsage.pt"
    args.model = "unsup_graphsage"
    args.trainer = "self_supervised"
    args, dataset = build_nn_dataset(args)
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0

    args.checkpoint = "graphsage.pt"
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0

    args.load_emb_path = args.save_dir + "/" + args.model + "_" + args.dataset + ".npy"
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_dgi():
    args = get_unsupervised_nn_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "cora"
    args.activation = "relu"
    args.sparse = True
    args.max_epochs = 2
    args.model = "dgi"
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_mvgrl():
    args = get_unsupervised_nn_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "cora"
    args.max_epochs = 2
    args.model = "mvgrl"
    args.sparse = False
    args.sample_size = 2000
    args.batch_size = 4
    args.alpha = 0.2
    args, dataset = build_nn_dataset(args)
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_grace():
    args = get_unsupervised_nn_args()
    args.model = "grace"
    args.num_layers = 2
    args.max_epoch = 2
    args.drop_feature_rates = [0.1, 0.2]
    args.drop_edge_rates = [0.2, 0.3]
    args.activation = "relu"
    args.proj_hidden_size = 32
    args.tau = 0.5
    args.dataset = "cora"
    args, dataset = build_nn_dataset(args)

    for bs in [-1, 512]:
        args.batch_size = bs
        task = build_task(args)
        ret = task.train()
        assert ret["Acc"] > 0


def test_gcc_usa_airport():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "usa-airport"
    args.model = "gcc"
    args.load_path = "./saved/gcc_pretrained.pth"
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


if __name__ == "__main__":
    # test_gcc_usa_airport()
    test_grace()
    test_mvgrl()
    test_unsupervised_graphsage()
    test_dgi()
