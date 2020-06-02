import argparse

from cogdl import options
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict


def build_args_from_dict(dic):
    args = ArgClass()
    for key, value in dic.items():
        args.__setattr__(key, value)
    return args


def accuracy_check(x):
    for _, value in x.items():
        assert value > 0


def get_default_args():
    default_dict = {'task': 'unsupervised_graph_classification',
                    'gamma': 0.5,
                    'device': 'cpu',
                    'num_shuffle': 1,
                    'save_dir': '.',
                    'dropout': 0.5,
                    'patience': 1,
                    'epochs': 2,
                    'cpu': True,
                    'lr': 0.001,
                    'weight_decay': 5e-4}
    return build_args_from_dict(default_dict)

def add_infograp_args(args):
    args.hidden_size = 64
    args.batch_size = 20
    args.target = 0
    args.train_num = 5000
    args.num_layers = 3
    args.unsup = True
    args.epochs = 30
    args.nn = True
    args.lr = 0.0001
    args.train_ratio = 0.7
    args.test_ratio = 0.1
    args.model = 'infograph'
    args.degree_feature = False
    return args


def add_graph2vec_args(args):
    args.hidden_size = 128
    args.window = 0
    args.min_count = 5
    args.dm = 0
    args.sampling = 0.0001
    args.iteration = 2
    args.epochs = 40
    args.nn = False
    args.lr = 0.001
    args.model = 'graph2vec'
    args.degree_feature = False
    return args


def test_infograph_proteins():
    args = get_default_args()
    args = add_infograp_args(args)
    args.dataset = 'proteins'
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_infograph_collab():
    args = get_default_args()
    args = add_infograp_args(args)
    args.dataset = 'collab'
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_infograph_lmdb_binary():
    args = get_default_args()
    args = add_infograp_args(args)
    args.dataset = 'lmdb-b'
    args.degree_feature = True
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_infograph_mutag():
    args = get_default_args()
    args = add_infograp_args(args)
    args.dataset = 'mutag'
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_graph2vec_mutag():
    args = get_default_args()
    args = add_graph2vec_args(args)
    args.dataset = 'mutag'
    print(args)
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


def test_graph2vec_proteins():
    args = get_default_args()
    args = add_graph2vec_args(args)
    args.dataset = 'proteins'
    print(args)
    task = build_task(args)
    ret = task.train()
    accuracy_check(ret)


if __name__ == "__main__":
    test_graph2vec_mutag()
    test_graph2vec_proteins()

    test_infograph_mutag()
    test_infograph_lmdb_binary()
    test_infograph_proteins()
