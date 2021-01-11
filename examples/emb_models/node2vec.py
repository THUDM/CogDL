from cogdl import experiment
from cogdl.utils import build_args_from_dict, print_result, set_random_seed, get_extra_args

DATASET_REGISTRY = {}


def default_parameter():
    args = {
        "hidden_size": 128,
        "num_shuffle": 5,
        "cpu": True,
        "seed": [0, 1, 2],
        "walk_length": 80,
        "walk_num": 40,
        "window_size": 5,
        "worker": 10,
        "iteration": 10,
        "p": 1.0,
        "q": 1.0,
    }
    args = get_extra_args(args)
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func

    return register_func_name


@register_func("ppi")
def ppi_config(args):
    return args


@register_func("blogcatalog")
def blog_config(args):
    return args


@register_func("wikipedia")
def wiki_config(args):
    return args


def run(dataset_name):
    args = default_parameter()
    args = DATASET_REGISTRY[dataset_name](args).__dict__
    results = experiment(task="unsupervised_node_classification", dataset=dataset_name, model="node2vec", **args)
    return results


if __name__ == "__main__":
    datasets = ["ppi", "blogcatalog", "wikipedia"]
    for x in datasets:
        run(x)
