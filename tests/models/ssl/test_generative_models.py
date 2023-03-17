import torch

from cogdl.experiments import train
from cogdl.options import get_default_args


cuda_available = torch.cuda.is_available()
default_dict = {
    "hidden_size": 64,
    "label_mask": 0,
    "mask_ratio": 0.1,
    "dropedge_rate": 0,
    "activation": "relu",
    "norm": None,
    "residual": False,
    "dropout": 0.5,
    "patience": 2,
    "device_id": [0],
    "epochs": 3,
    "sampler": "none",
    "sampling": False,
    "cpu": not cuda_available,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "checkpoint": False,
    "label_mask": 0,
    "num_layers": 2,
    "do_train": True,
    "do_eval": True,
    "save_dir": "./embedding",
    "load_dir": "./embedding",
    "eval_agc": False,
    "subgraph_sampling": False,
    "sample_size": 128,
    "actnn": False,
}


def get_default_args_generative(dataset, model, dw="node_classification_dw", mw="self_auxiliary_mw", **kwargs):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict.items():
        args.__setattr__(key, value)
    for key, value in kwargs.items():
        args.__setattr__(key, value)
    return args


def test_edgemask():
    args = get_default_args_generative("cora", "gcn", auxiliary_task="edge_mask")
    args.alpha = 1
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_attribute_mask():
    args = get_default_args_generative("cora", "gcn", auxiliary_task="attribute_mask")
    args.alpha = 1
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_pairwise_distance():
    args = get_default_args_generative("cora", "gcn", auxiliary_task="pairwise_distance")
    args.alpha = 35
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_pairwise_distance_sampling():
    args = get_default_args_generative("cora", "gcn", auxiliary_task="pairwise_distance")
    args.alpha = 35
    args.sampling = True
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_distance_to_clusters():
    args = get_default_args_generative("cora", "gcn", auxiliary_task="distance2clusters")
    args.alpha = 3
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_pairwise_attr_sim():
    args = get_default_args_generative("cora", "gcn", auxiliary_task="pairwise_attr_sim")
    args.alpha = 100
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_m3s():
    args = get_default_args_generative("cora", "m3s", dw="m3s_dw", mw="m3s_mw")
    args.approximate = True
    args.num_clusters = 50
    args.num_stages = 1
    args.epochs_per_stage = 3
    args.label_rate = 1
    args.num_new_labels = 2
    args.alpha = 1
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


if __name__ == "__main__":
    test_m3s()
    test_edgemask()
    test_pairwise_distance()
    test_pairwise_distance_sampling()
    test_distance_to_clusters()
    test_pairwise_attr_sim()
