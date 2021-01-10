from cogdl import experiment
from cogdl.utils import build_args_from_dict, print_result, set_random_seed, get_extra_args

DATASET_REGISTRY = {}


def default_parameter():
    args = {
        "hidden_size": 128,
        "num_shuffle": 5,
        "cpu": True,
        "seed": [0, 1, 2],
        "lr": 0.001,
        "max_epoch": 500,
        "hidden_size1": 1000,
        "hidden_size2": 128,
        "noise": 0.2,
        "alpha": 0.1,
        "step": 10,
    }
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
    results = experiment(task="unsupervised_node_classification", dataset=dataset_name, model="dngr", **args)
    return results


if __name__ == "__main__":
    datasets = ["ppi", "blogcatalog", "wikipedia"]
    for x in datasets:
        run(x)
