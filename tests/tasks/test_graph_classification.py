import torch
from cogdl.options import get_default_args
from cogdl.experiments import train

cuda_available = torch.cuda.is_available()
default_dict = {
    "task": "graph_classification",
    "hidden_size": 32,
    "dropout": 0.5,
    "patience": 1,
    "epochs": 2,
    "cpu": not cuda_available,
    "lr": 0.001,
    "kfold": False,
    "seed": [0],
    "weight_decay": 5e-4,
    "gamma": 0.5,
    "train_ratio": 0.7,
    "test_ratio": 0.1,
    "device_id": [0 if cuda_available else "cpu"],
    "sampler": "none",
    "degree_node_features": False,
    "checkpoint": False,
    "residual": False,
    "activation": "relu",
    "norm": None,
}


def get_default_args_graph_clf(dataset, model, dw="graph_classification_dw", mw="graph_classification_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict.items():
        args.__setattr__(key, value)
    return args


def add_diffpool_args(args):
    args.num_layers = 2
    args.num_pooling_layers = 1
    args.no_link_pred = False
    args.pooling_ratio = 0.15
    args.embedding_dim = 64
    args.hidden_size = 64
    args.batch_size = 20
    args.dropout = 0.1
    return args


def add_gin_args(args):
    args.epsilon = 0.0
    args.hidden_size = 32
    args.num_layers = 5
    args.num_mlp_layers = 2
    args.train_epsilon = True
    args.pooling = "sum"
    args.batch_size = 128
    return args


def add_sortpool_args(args):
    args.hidden_size = 64
    args.batch_size = 20
    args.num_layers = 2
    args.out_channels = 32
    args.k = 30
    args.kernel_size = 5
    return args


def add_patchy_san_args(args):
    args.hidden_size = 64
    args.batch_size = 20
    args.sample = 10
    args.stride = 1
    args.neighbor = 10
    args.iteration = 2
    args.train_ratio = 0.7
    args.test_ratio = 0.1
    return args


def test_gin_mutag():
    args = get_default_args_graph_clf(dataset="mutag", model="gin")
    args = add_gin_args(args)
    args.batch_size = 20
    for kfold in [True, False]:
        args.kfold = kfold
        args.seed = 0
        ret = train(args)
        assert ret["test_acc"] > 0


def test_gin_imdb_binary():
    args = get_default_args_graph_clf(dataset="imdb-b", model="gin")
    args = add_gin_args(args)
    args.degree_node_features = True
    ret = train(args)
    assert ret["test_acc"] > 0


def test_gin_proteins():
    args = get_default_args_graph_clf(dataset="imdb-b", model="gin")
    args = add_gin_args(args)
    ret = train(args)
    assert ret["test_acc"] > 0


def test_diffpool_imdb_binary():
    args = get_default_args_graph_clf(dataset="imdb-b", model="diffpool")
    args = add_diffpool_args(args)
    args.batch_size = 100
    args.train_ratio = 0.6
    args.test_ratio = 0.2
    ret = train(args)
    assert ret["test_acc"] > 0


def test_sortpool_mutag():
    args = get_default_args_graph_clf(dataset="mutag", model="sortpool")
    args = add_sortpool_args(args)
    args.batch_size = 20
    ret = train(args)
    assert ret["test_acc"] > 0


def test_patchy_san_mutag():
    args = get_default_args_graph_clf(dataset="mutag", model="patchy_san", dw="patchy_san_dw")
    args = add_patchy_san_args(args)
    args.batch_size = 5
    ret = train(args)
    assert ret["test_acc"] > 0


if __name__ == "__main__":

    test_gin_imdb_binary()
    test_gin_mutag()
    test_gin_proteins()

    test_sortpool_mutag()
    test_diffpool_imdb_binary()
    test_patchy_san_mutag()
