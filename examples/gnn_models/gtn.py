import torch
from cogdl.datasets import build_dataset
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict, print_result, set_random_seed

DATASET_REGISTRY = {}


def build_default_args_for_heterogeneous_node_classification(dataset):
    cpu = not torch.cuda.is_available()
    args = {
        "hidden_size": 128,
        "patience": 100,
        "max_epoch": 500,
        "cpu": cpu,
        "device_id": [0],
        "lr": 0.005,
        "weight_decay": 0.001,
        "seed": [0, 1, 2],

        "num_layers": 2,
        "num_channels": 2,

        "task": "heterogeneous_node_classification",
        "model": "gtn",
        "dataset": dataset
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func
    return register_func_name


@register_func("gtn-dblp")
def dblp_config(args):
    return args


@register_func("gtn-acm")
def acm_config(args):
    return args


@register_func("gtn-imdb")
def imdb_config(args):
    return args


def run(dataset_name):
    args = build_default_args_for_heterogeneous_node_classification(dataset_name)
    args = DATASET_REGISTRY[dataset_name](args)
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_edge = dataset.num_edge
    args.num_nodes = dataset.num_nodes
    results = []
    for seed in args.seed:
        set_random_seed(seed)
        task = build_task(args, dataset=dataset)
        result = task.train()
        results.append(result)
    return results


if __name__ == "__main__":
    datasets = ["gtn-dblp", "gtn-acm", "gtn-imdb"]
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "gtn")
