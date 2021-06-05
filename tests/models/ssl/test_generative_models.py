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
        "trainer": "self_auxiliary_task_joint",
        "model": "gcn",
        "hidden_size": 64,
        "label_mask": 0,
        "mask_ratio": 0.1,
        "dropout": 0.5,
        "patience": 2,
        "device_id": [0],
        "max_epoch": 3,
        "sampler": "none",
        "sampling": False,
        "cpu": not cuda_available,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "missing_rate": -1,
        "task": "node_classification",
        "checkpoint": False,
        "num_layers": 2,
        "activation": "relu",
        "dropedge_rate": 0,
        "agc_eval": False,
        "residual": False,
        "norm": None,
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


def test_edgemask_pt_ft():
    args = get_default_args()
    args.auxiliary_task = "edgemask"
    args.trainer = "self_auxiliary_task_pretrain"
    args.alpha = 1
    args.freeze = False
    dataset = build_dataset(args)
    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_edgemask_pt_ft_freeze():
    args = get_default_args()
    args.auxiliary_task = "edgemask"
    args.trainer = "self_auxiliary_task_pretrain"
    args.alpha = 1
    args.freeze = True
    dataset = build_dataset(args)
    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_attribute_mask():
    args = get_default_args()
    args.auxiliary_task = "attributemask"
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


def test_pairwise_distance_sampling():
    args = get_default_args()
    args.auxiliary_task = "pairwise-distance"
    args.alpha = 35
    args.sampling = True
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


def test_supergat():
    args = get_default_args()
    args.model = "supergat"
    args.trainer = None
    args.heads = 8
    args.attention_type = "mask_only"
    args.neg_sample_ratio = 0.5
    args.edge_sampling_ratio = 0.8
    args.val_interval = 1
    args.att_lambda = 10
    args.pretraining_noise_ratio = 0
    args.to_undirected_at_neg = False
    args.to_undirected = False
    args.out_heads = None
    args.total_pretraining_epoch = 0
    args.super_gat_criterion = None
    args.scaling_factor = None
    dataset = build_dataset(args)
    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


def test_m3s():
    args = get_default_args()
    args.model = "m3s"
    args.trainer = None
    args.approximate = True
    args.num_clusters = 50
    args.num_stages = 1
    args.epochs_per_stage = 3
    args.label_rate = 1
    args.num_new_labels = 2
    args.alpha = 1
    dataset = build_dataset(args)
    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
    assert 0 <= ret["Acc"] <= 1


if __name__ == "__main__":
    # test_supergat()
    test_m3s()
    test_edgemask()
    test_edgemask_pt_ft()
    test_edgemask_pt_ft_freeze()
    test_pairwise_distance()
    test_pairwise_distance_sampling()
    test_distance_to_clusters()
    test_pairwise_attr_sim()
    test_distance_to_clustersPP()
