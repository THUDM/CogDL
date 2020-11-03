import random
import numpy as np

import torch

from utils import print_result, set_random_seed, get_dataset
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def build_default_args_for_node_classification(dataset):
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.01,
        "weight_decay": 5e-4,
        "max_epoch": 1000,
        "max_epochs": 1000,
        "patience": 100,
        "cpu": cpu,
        "device_id": [0],
        "seed": [42],

        "dropout": 0.5,
        "hidden_size": 256,
        "num_layers": 32,
        "lmbda": 0.5,
        "wd1": 0.001,
        "wd2": 5e-4,
        "alpha": 0.1,

        "task": "node_classification",
        "model": "gcnii",
        "dataset": dataset
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func
    return register_func_name


@register_func("cora")
def cora_config(args):
    args.num_layers = 64
    args.hidden_size = 64
    args.dropout = 0.6
    return args


@register_func("citeseer")
def citeseer_config(args):
    args.num_layers = 32
    args.hidden_size = 256
    args.lr = 0.001
    args.patience = 200
    args.max_epoch = 2000
    args.lmbda = 0.6
    args.dropout = 0.7
    return args


@register_func("pubmed")
def pubmed_config(args):
    args.num_layers = 16
    args.hidden_size = 256
    args.lmbda = 0.4
    args.dropout = 0.5
    args.wd1 = 5e-4
    return args


def run(dataset_name):
    args = build_default_args_for_node_classification(dataset_name)
    args = DATASET_REGISTRY[dataset_name](args)
    dataset, args = get_dataset(args)
    results = []
    for seed in args.seed:
        set_random_seed(seed)
        task = build_task(args, dataset=dataset)
        result = task.train()
        results.append(result)
    return results


if __name__ == "__main__":
    # datasets = ["cora", "citeseer", "pubmed"]
    datasets = ["citeseer"]
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "gcnii")
