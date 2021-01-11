from cogdl import experiment
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def default_parameters_sup():
    args = {
        "seed": [0],
    }
    return build_args_from_dict(args)


def default_parameter_unsup():
    args = {
        "lr": 0.001,
        "weight_decay": 0,
        "max_epoch": 3000,
        "seed": [0],
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
    unsup = False
    if unsup:
        task = "unsupervised_node_classification"
        model = "unsup_graphsage"
        args = default_parameter_unsup()
    else:
        task = "node_classification"
        model = "graphsage"
        args = default_parameters_sup()
    args = DATASET_REGISTRY[dataset_name](args).__dict__
    results = experiment(task=task, dataset=dataset_name, model=model, **args)
    return results


if __name__ == "__main__":
    datasets = ["cora", "citeseer", "pubmed"]
    for x in datasets:
        run(x)
