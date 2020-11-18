import torch
from cogdl.datasets import build_dataset
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict, print_result, set_random_seed

DATASET_REGISTRY = {}


def build_default_args_for_multiplex_link_prediction(dataset):
    cpu = not torch.cuda.is_available()
    args = {
        "hidden_size": 200,
        "cpu": cpu,
        "eval_type": "all",
        "seed": [0, 1, 2],

        "walk_length": 10,
        "walk_num": 10,
        "window_size": 5,
        "worker": 10,
        "epoch": 20,
        "batch_size": 256,
        "edge_dim": 10,
        "att_dim": 20,
        "negative_samples": 5,
        "neighbor_samples": 10,
        "schema": None,

        "task": "multiplex_link_prediction",
        "model": "gatne",
        "dataset": dataset
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func
    return register_func_name


@register_func("amazon")
def dblp_config(args):
    return args


@register_func("youtube")
def acm_config(args):
    return args


@register_func("twitter")
def imdb_config(args):
    args.eval_type = "1"
    return args


def run(dataset_name):
    args = build_default_args_for_multiplex_link_prediction(dataset_name)
    args = DATASET_REGISTRY[dataset_name](args)
    dataset = build_dataset(args)
    results = []
    for seed in args.seed:
        set_random_seed(seed)
        task = build_task(args, dataset=dataset)
        result = task.train()
        results.append(result)
    return results


if __name__ == "__main__":
    datasets = ["amazon", "youtube", "twitter"]
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "gatne")
