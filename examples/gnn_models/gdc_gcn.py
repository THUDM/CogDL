from cogdl import experiment
from cogdl.utils import build_args_from_dict


def default_parameter():
    default_dict = {
        "hidden_size": 16,
        "gdc_type": "ppr",
    }
    return build_args_from_dict(default_dict)


def run(dataset_name):
    args = default_parameter().__dict__
    results = experiment(task="node_classification", dataset=dataset_name, model="gdc_gcn", **args)
    return results


if __name__ == "__main__":
    datasets = ["cora", "citeseer", "pubmed"]
    for x in datasets:
        run(x)
