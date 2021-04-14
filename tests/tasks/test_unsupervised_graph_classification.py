import torch

from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict


def accuracy_check(x):
    for _, value in x.items():
        assert value > 0


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "task": "unsupervised_graph_classification",
        "gamma": 0.5,
        "device_id": [0 if cuda_available else "cpu"],
        "num_shuffle": 1,
        "save_dir": ".",
        "dropout": 0.5,
        "patience": 1,
        "epoch": 2,
        "cpu": not cuda_available,
        "lr": 0.001,
        "weight_decay": 5e-4,
        "checkpoint": False,
        "activation": "relu",
        "residual": False,
        "norm": None,
    }
    return build_args_from_dict(default_dict)


def add_infograp_args(args):
    args.hidden_size = 64
    args.batch_size = 20
    args.target = 0
    args.train_num = 5000
    args.num_layers = 3
    args.sup = False
    args.epoch = 3
    args.nn = True
    args.lr = 0.0001
    args.train_ratio = 0.7
    args.test_ratio = 0.1
    args.model = "infograph"
    args.degree_feature = False
    return args


def add_graph2vec_args(args):
    args.hidden_size = 128
    args.window_size = 0
    args.min_count = 5
    args.dm = 0
    args.sampling = 0.0001
    args.iteration = 2
    args.epoch = 4
    args.nn = False
    args.lr = 0.001
    args.model = "graph2vec"
    args.degree_feature = False
    return args


def add_dgk_args(args):
    args.hidden_size = 128
    args.window_size = 2
    args.min_count = 5
    args.sampling = 0.0001
    args.iteration = 2
    args.epoch = 4
    args.nn = False
    args.alpha = 0.01
    args.model = "dgk"
    args.degree_feature = False
    return args


def test_infograph_proteins():
    args = get_default_args()
    args = add_infograp_args(args)
    args.dataset = "proteins"
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_infograph_imdb_binary():
    args = get_default_args()
    args = add_infograp_args(args)
    args.dataset = "imdb-b"
    args.degree_feature = True
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_infograph_mutag():
    args = get_default_args()
    args = add_infograp_args(args)
    args.dataset = "mutag"
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_graph2vec_mutag():
    args = get_default_args()
    args = add_graph2vec_args(args)
    args.dataset = "mutag"
    print(args)
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_graph2vec_proteins():
    args = get_default_args()
    args = add_graph2vec_args(args)
    args.dataset = "proteins"
    print(args)
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_dgk_mutag():
    args = get_default_args()
    args = add_dgk_args(args)
    args.dataset = "mutag"
    print(args)
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_dgk_proteins():
    args = get_default_args()
    args = add_dgk_args(args)
    args.dataset = "proteins"
    print(args)
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


if __name__ == "__main__":
    test_graph2vec_mutag()
    test_graph2vec_proteins()

    test_infograph_mutag()
    test_infograph_imdb_binary()
    test_infograph_proteins()

    test_dgk_mutag()
    test_dgk_proteins()
