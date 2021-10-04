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
    "max_epoch": 3,
    "sampler": "none",
    "sampling": False,
    "cpu": not cuda_available,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "missing_rate": -1,
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


def get_default_args_generative(dataset, model, dw="node_classification_dw", mw="self_auxiliary_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict.items():
        args.__setattr__(key, value)
    return args


def test_edgemask():
    args = get_default_args_generative("cora", "gcn", auxiliary_task="edgemask")
    args.alpha = 1
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_attribute_mask():
    args = get_default_args_generative("cora", "gcn", auxiliary_task="attribute_mask")
    args.alpha = 1
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_pairwise_distance():
    args = get_default_args_generative("cora", "gcn", auxiliary_task="pairwise-distance")
    args.alpha = 35
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


def test_pairwise_distance_sampling():
    args = get_default_args_generative("cora", "gcn", auxiliary_task="pairwise-distance")
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
    args = get_default_args_generative("cora", "gcn", auxiliary_task="pairwise-attr-sim")
    args.alpha = 100
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


# def test_supergat():
#     args = get_default_args()
#     args.model = "supergat"
#     args.trainer = None
#     args.heads = 8
#     args.attention_type = "mask_only"
#     args.neg_sample_ratio = 0.5
#     args.edge_sampling_ratio = 0.8
#     args.val_interval = 1
#     args.att_lambda = 10
#     args.pretraining_noise_ratio = 0
#     args.to_undirected_at_neg = False
#     args.to_undirected = False
#     args.out_heads = None
#     args.total_pretraining_epoch = 0
#     args.super_gat_criterion = None
#     args.scaling_factor = None
#     ret = train(args)
#     assert 0 <= ret["test_acc"] <= 1


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
    ret = train(args)
    assert 0 <= ret["test_acc"] <= 1


if __name__ == "__main__":
    # test_supergat()
    test_m3s()
    test_edgemask()
    test_pairwise_distance()
    test_pairwise_distance_sampling()
    test_distance_to_clusters()
    test_pairwise_attr_sim()
