from cogdl import experiment
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def default_parameter():
    args = {
        "weight_decay": 0,
        "max_epoch": 1000,
        "patience": 20,
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
    args.lr = 0.0005
    args.weight_decay = 0.00001
    args.tau = 0.4
    args.drop_feature_rates = [0.3, 0.4]
    args.drop_edge_rates = [0.2, 0.4]
    args.max_epoch = 200
    args.hidden_size = 128
    args.proj_hidden_size = 128
    return args


@register_func("citeseer")
def citeseer_config(args):
    args.hidden_size = 256
    args.proj_hidden_size = 256
    args.drop_feature_rates = [0.3, 0.2]
    args.drop_edge_rates = [0.2, 0.0]
    args.lr = 0.001
    args._weight_decay = 0.00001
    args.tau = 0.9
    args.activation = "prelu"
    return args


@register_func("pubmed")
def pubmed_config(args):
    args.hidden_size = 256
    args.proj_hidden_size = 256
    args.drop_edge_rates = [0.4, 0.1]
    args.drop_feature_rates = [0.0, 0.2]
    args.tau = 0.7
    args.lr = 0.001
    args.weight_decay = 0.00001
    return args


def run(dataset_name):
    args = default_parameter()
    args = DATASET_REGISTRY[dataset_name](args).__dict__
    results = experiment(task="unsupervised_node_classification", dataset=dataset_name, model="grace", **args)
    return results


if __name__ == "__main__":
    datasets = ["cora", "citeseer", "pubmed"]
    for x in datasets:
        run(x)
