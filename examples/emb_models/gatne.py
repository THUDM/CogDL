from cogdl import experiment
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def default_parameter():
    args = {
        "hidden_size": 200,
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
    args = default_parameter()
    args = DATASET_REGISTRY[dataset_name](args).__dict__
    results = experiment(task="multiplex_link_prediction", dataset=dataset_name, model="gatne", **args)
    return results


if __name__ == "__main__":
    datasets = ["amazon", "youtube", "twitter"]
    for x in datasets:
        run(x)
