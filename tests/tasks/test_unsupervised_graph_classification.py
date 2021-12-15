import torch

from cogdl.options import get_default_args
from cogdl.experiments import train


cuda_available = torch.cuda.is_available()
default_dict = {
    "task": "unsupervised_graph_classification",
    "gamma": 0.5,
    "device_id": [0 if cuda_available else "cpu"],
    "num_shuffle": 1,
    "save_dir": ".",
    "dropout": 0.5,
    "patience": 1,
    "epochs": 2,
    "cpu": not cuda_available,
    "lr": 0.001,
    "weight_decay": 5e-4,
    "checkpoint": False,
    "activation": "relu",
    "residual": False,
    "norm": None,
}


def accuracy_check(x):
    for _, value in x.items():
        assert value >= 0


def get_default_args_graph_clf(dataset, model, dw="graph_embedding_dw", mw="graph_embedding_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict.items():
        args.__setattr__(key, value)
    return args


def add_infograp_args(args):
    args.hidden_size = 16
    args.batch_size = 32
    args.target = 0
    args.train_num = 100
    args.num_layers = 2
    args.sup = False
    args.epochs = 2
    args.nn = True
    args.lr = 0.0001
    args.train_ratio = 0.7
    args.test_ratio = 0.2
    args.model = "infograph"
    args.degree_node_features = False
    return args


def add_graph2vec_args(args):
    args.hidden_size = 16
    args.window_size = 0
    args.min_count = 5
    args.dm = 0
    args.sampling = 0.0001
    args.iteration = 2
    args.epochs = 4
    args.nn = False
    args.lr = 0.001
    args.model = "graph2vec"
    args.degree_node_features = False
    return args


def add_dgk_args(args):
    args.hidden_size = 16
    args.window_size = 2
    args.min_count = 5
    args.sampling = 0.0001
    args.iteration = 2
    args.epochs = 4
    args.nn = False
    args.alpha = 0.01
    args.model = "dgk"
    args.degree_node_features = False
    return args


def test_infograph_imdb_binary():
    args = get_default_args_graph_clf(dataset="imdb-b", model="infograph", dw="infograph_dw", mw="infograph_mw")
    args = add_infograp_args(args)
    args.degree_node_features = True
    ret = train(args)
    accuracy_check(ret)


def test_graph2vec_mutag():
    args = get_default_args_graph_clf(dataset="mutag", model="graph2vec")
    args = add_graph2vec_args(args)
    ret = train(args)
    accuracy_check(ret)


def test_graph2vec_proteins():
    args = get_default_args_graph_clf(dataset="proteins", model="graph2vec")
    args = add_graph2vec_args(args)
    ret = train(args)
    accuracy_check(ret)


def test_dgk_mutag():
    args = get_default_args_graph_clf(dataset="mutag", model="dgk")
    args = add_dgk_args(args)
    ret = train(args)
    accuracy_check(ret)


def test_dgk_proteins():
    args = get_default_args_graph_clf(dataset="proteins", model="dgk")
    args = add_dgk_args(args)
    ret = train(args)
    accuracy_check(ret)


if __name__ == "__main__":
    test_graph2vec_mutag()
    test_graph2vec_proteins()

    test_infograph_imdb_binary()

    test_dgk_mutag()
    test_dgk_proteins()
