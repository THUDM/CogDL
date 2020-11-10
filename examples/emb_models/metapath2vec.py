from cogdl.datasets import build_dataset
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict, print_result, set_random_seed

DATASET_REGISTRY = {}


def build_default_args_for_multiplex_node_classification(dataset):
    args = {
        "hidden_size": 128,
        "cpu": True,
        "enhance": None,
        "save_dir": ".",
        "seed": [0, 1, 2],

        "walk_length": 80,
        "walk_num": 40,
        "window_size": 5,
        "worker": 10,
        "iteration": 10,
        "schema": "No",

        "task": "multiplex_node_classification",
        "model": "metapath2vec",
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
    args = build_default_args_for_multiplex_node_classification(dataset_name)
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
    datasets = ["gtn-dblp", "gtn-acm", "gtn-imdb"]
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "metapath2vec")
