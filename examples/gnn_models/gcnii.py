from cogdl import experiment
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def default_parameter():
    args = {
        "max_epoch": 1000,
        "seed": [42],
        "dropout": 0.5,
        "wd1": 0.001,
        "wd2": 5e-4,
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func

    return register_func_name


@register_func("cora")
def cora_config(args):
    args.num_layers = 64
    args.hidden_size = 64
    args.dropout = 0.6
    return args


@register_func("citeseer")
def citeseer_config(args):
    args.num_layers = 32
    args.hidden_size = 256
    args.lr = 0.001
    args.patience = 200
    args.max_epoch = 2000
    args.lmbda = 0.6
    args.dropout = 0.7
    return args


@register_func("pubmed")
def pubmed_config(args):
    args.num_layers = 16
    args.hidden_size = 256
    args.lmbda = 0.4
    args.dropout = 0.5
    args.wd1 = 5e-4
    return args


def run(dataset_name):
    args = default_parameter()
    args = DATASET_REGISTRY[dataset_name](args).__dict__
    results = experiment(task="node_classification", dataset=dataset_name, model="gcnii", **args)
    return results


if __name__ == "__main__":
    # datasets = ["cora", "citeseer", "pubmed"]
    datasets = ["citeseer"]
    for x in datasets:
        run(x)
