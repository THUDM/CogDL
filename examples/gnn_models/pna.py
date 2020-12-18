import random
import numpy as np
import torch
from utils import print_result, set_random_seed, get_dataset
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}

def build_default_args_for_graph_classification(dataset):
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.001,
        "weight_decay": 5e-4,
        "max_epoch": 500,
        "patience": 50,
        "cpu": cpu,
        "device_id": [0],
        "seed": [0],

        "train_ratio": 0.7,
        "test_ratio": 0.1,
        "batch_size": 128,
        "kfold": False,
        "degree_feature": False,

        "hidden_size": 60,
        "avg_deg": 1,
        "layer": 4,
        "pre_layers": 1,
        "towers": 5,
        "post_layers": 1,
        "edge_dim": None,
        "aggregators": ['mean', 'min', 'max', 'std'],
        "scalers": ['identity', 'amplification', 'attenuation'], 
        "divide_input": False,

        "task": "graph_classification",
        "model": "pyg_pna",
        "dataset": dataset
    }
    return build_args_from_dict(args)

def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func
    return register_func_name

@register_func("proteins")
def proteins_config(args):
    return args

def run(dataset_name):
    args = build_default_args_for_graph_classification(dataset_name)
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
    datasets = ["proteins"]
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "pyg_pna")