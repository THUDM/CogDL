import random
import numpy as np

import torch

from utils import print_result, set_random_seed, get_dataset, get_extra_args
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def build_default_args_for_node_classification(dataset):
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.01,
        "weight_decay": 5e-4,
        "max_epoch": 500,
        "patience": 100,
        "cpu": cpu,
        "device_id": [0],
        "seed": [
            0,
        ],
        "dropout": 0.5,
        "hidden_size": [
            128,
        ],
        "num_layers": 2,
        "sample_size": [10, 10],
        "task": "node_classification",
        "model": "graphsage",
        "dataset": dataset,
    }
    args = get_extra_args(args)
    return build_args_from_dict(args)


def build_default_args_for_unsupervised_node_classification(dataset):
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.001,
        "weight_decay": 0,
        "max_epoch": 500,
        "max_epochs": 3000,
        "patience": 100,
        "cpu": cpu,
        "device_id": 0,
        "seed": [
            0,
        ],
        "num_shuffle": 5,
        "save_dir": ".",
        "enhance": None,
        "negative_samples": 30,
        "dropout": 0.5,
        "hidden_size": 128,
        "num_layers": 2,
        "sample_size": [10, 10],
        "walk_length": 10,
        "task": "unsupervised_node_classification",
        "model": "unsup_graphsage",
        "dataset": dataset,
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func

    return register_func_name


@register_func("cora")
def cora_config(args):
    return args


@register_func("citeseer")
def citeseer_config(args):
    return args


@register_func("pubmed")
def pubmed_config(args):
    return args


def run(dataset_name):
    unsup = False  # unsupervised or supervised node classification
    if unsup:
        args = build_default_args_for_unsupervised_node_classification(dataset_name)
    else:
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
    print_result(results, datasets, "graphsage")
