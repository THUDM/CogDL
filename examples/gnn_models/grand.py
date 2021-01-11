import torch

from cogdl import experiment
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def default_parameter():
    args = {
        "max_epoch": 1000,
        "seed": [100],
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func

    return register_func_name


@register_func("cora")
def cora_config(args):
    args.order = 8
    args.sample = 4
    args.lam = 1.0
    args.tem = 0.5
    args.alpha = 0.5
    args.patience = 200
    args.input_dropout = 0.5
    args.hidden_dropout = 0.5
    return args


@register_func("citeseer")
def citeseer_config(args):
    args.order = 2
    args.sample = 2
    args.lam = 0.7
    args.tem = 0.3
    args.alpha = 0.5
    args.input_dropout = 0.0
    args.hidden_dropout = 0.2
    args.patience = 200
    return args


@register_func("pubmed")
def pubmed_config(args):
    args.order = 5
    args.sample = 4
    args.lam = 1.0
    args.tem = 0.2
    args.alpha = 0.5
    args.lr = 0.2
    args.bn = True
    args.input_dropout = 0.6
    args.hidden_dropout = 0.8
    args["hidden_dropout"] = 0.8
    return args


def run(dataset_name):
    args = default_parameter()
    args = DATASET_REGISTRY[dataset_name](args).__dict__
    results = experiment(task="node_classification", dataset=dataset_name, model="grand", **args)
    return results


if __name__ == "__main__":
    datasets = ["cora", "citeseer", "pubmed"]
    for x in datasets:
        run(x)
