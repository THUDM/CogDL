import random
import numpy as np

import torch

from utils import print_result, set_random_seed, get_dataset
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def build_default_args_for_node_classification(dataset):
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.001,
        "weight_decay": 0,
        "max_epoch": 1000,
        "max_epochs": 1000,
        "patience": 20,
        "cpu": cpu,
        "device_id": [0],
        "seed": [0],
        "num_shuffle": 5,

        "drop_edge_rates": [0.1, 0.2],
        "drop_feature_rates": [0.2, 0.3],
        "hidden_size": 128,
        "num_layers": 2,
        "proj_hidden_size": 128,
        "tau": 0.4,
        "activation": "relu",
        "sampler": "none",

        "task": "unsupervised_node_classification",
        "model": "grace",
        "dataset": dataset,

        "save_dir": "./saved",
        "enhance": None,
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func
    return register_func_name


@register_func("cora")
def cora_config(args):
    args.lr = 0.0005
    args.weight_decay = 0.00001
    args.tau = 0.4
    args.drop_feature_rates = [0.3, 0.4]
    args.drop_edge_rates = [0.2, 0.4]
    args.max_epoch = 200
    args.hidden_size = 128
    args.proj_hidden_size = 128
    return args


@register_func("citeseer")
def citeseer_config(args):
    args.hidden_size = 256
    args.proj_hidden_size = 256
    args.drop_feature_rates = [0.3, 0.2]
    args.drop_edge_rates = [0.2, 0.0]
    args.lr = 0.001
    args._weight_decay = 0.00001
    args.tau = 0.9
    args.activation = "prelu"
    return args


@register_func("pubmed")
def pubmed_config(args):
    args.hidden_size = 256
    args.proj_hidden_size = 256
    args.drop_edge_rates = [0.4, 0.1]
    args.drop_feature_rates = [0.0, 0.2]
    args.tau = 0.7
    args.lr = 0.001
    args.weight_decay = 0.00001
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
    datasets = ["cora", "citeseer", "pubmed"]
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "grace")
