import torch

from cogdl.experiments import train
from cogdl.options import get_default_args


default_dict = {
    "hidden_size": 16,
    "num_layers": 2,
    "lr": 0.01,
    "dropout": 0.0,
    "patience": 1,
    "epochs": 1,
    "cpu": not torch.cuda.is_available(),
    "weight_decay": 5e-4,
    "num_shuffle": 2,
    "save_dir": "./embedding",
    "enhance": None,
    "device_id": [0],
    "task": "unsupervised_node_classification",
    "checkpoint": False,
    "load_emb_path": None,
    "training_percents": [0.1],
    "subgraph_sampling": False,
    "sample_size": 128,
    "do_train": True,
    "do_eval": True,
    "eval_agc": False,
    "save_dir": "./embedding",
    "load_dir": "./embedding",
    "alpha": 1,
}


def get_default_args_for_unsup_nn(dataset, model, dw="node_classification_dw", mw="self_auxiliary_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict.items():
        args.__setattr__(key, value)
    return args


def test_unsupervised_graphsage():
    args = get_default_args_for_unsup_nn("cora", "unsup_graphsage", dw="unsup_graphsage_dw", mw="unsup_graphsage_mw")
    args.negative_samples = 10
    args.hidden_size = [128]
    args.walk_length = 5
    args.sample_size = [5, 5]
    args.patience = 20
    args.epochs = 2
    args.checkpoint_path = "graphsage.pt"
    ret = train(args)
    assert ret["micro-f1 0.1"] > 0


def test_dgi():
    args = get_default_args_for_unsup_nn("cora", "dgi", mw="dgi_mw")
    args.activation = "relu"
    args.sparse = True
    args.epochs = 2
    ret = train(args)
    assert ret["test_acc"] > 0


def test_mvgrl():
    args = get_default_args_for_unsup_nn("cora", "mvgrl", mw="mvgrl_mw")
    args.epochs = 2
    args.sparse = False
    args.sample_size = 200
    args.batch_size = 4
    args.alpha = 0.2
    ret = train(args)
    assert ret["test_acc"] > 0


def test_grace():
    args = get_default_args_for_unsup_nn("cora", "grace", mw="grace_mw")
    args.num_layers = 2
    args.epochs = 2
    args.drop_feature_rates = [0.1, 0.2]
    args.drop_edge_rates = [0.2, 0.3]
    args.activation = "relu"
    args.proj_hidden_size = 32
    args.tau = 0.5

    for bs in [-1, 512]:
        args.batch_size = bs
        ret = train(args)
        assert ret["test_acc"] > 0


if __name__ == "__main__":
    test_unsupervised_graphsage()
    # test_grace()
    # test_mvgrl()
    # test_dgi()
