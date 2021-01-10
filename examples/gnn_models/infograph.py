from cogdl import experiment
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def default_parameter():
    args = {
        "lr": 0.0001,
        "weight_decay": 5e-4,
        "seed": [0],
        "sup": False,
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func

    return register_func_name


@register_func("mutag")
def mutag_config(args):
    args.num_layers = 1
    args.seed = 0
    args.epoch = 20
    return args


@register_func("imdb-b")
def imdb_b_config(args):
    args.degree_feature = True
    return args


@register_func("imdb-m")
def imdb_m_config(args):
    args.degree_feature = True
    return args


@register_func("proteins")
def proteins_config(args):
    return args


@register_func("collab")
def collab_config(args):
    args.degree_feature = True
    return args


def run(dataset_name):
    args = default_parameter()
    args = DATASET_REGISTRY[dataset_name](args).__dict__
    results = experiment(task="graph_classification", dataset=dataset_name, model="infograph", **args)
    return results


if __name__ == "__main__":
    datasets = ["mutag", "imdb-b", "imdb-m", "proteins", "collab"]
    for x in datasets:
        run(x)
