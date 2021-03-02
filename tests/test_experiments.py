from collections import namedtuple

from cogdl.experiments import check_task_dataset_model_match, experiment, gen_variants, train, set_best_config
from cogdl.options import get_default_args

import metis


def test_set_best_config():
    args = get_default_args(task="node_classification", dataset="citeseer", model="gat")
    args.model = args.model[0]
    args.dataset = args.dataset[0]
    args = set_best_config(args)

    assert args.lr == 0.005
    assert args.max_epoch == 1000
    assert args.weight_decay == 0.001


def test_train():
    args = get_default_args(task="node_classification", dataset="cora", model="gcn", max_epoch=10, cpu=True)
    args.dataset = args.dataset[0]
    args.model = args.model[0]
    args.seed = args.seed[0]
    result = train(args)

    assert "Acc" in result
    assert result["Acc"] > 0


def test_gen_variants():
    variants = list(gen_variants(dataset=["cora"], model=["gcn", "gat"], seed=[1, 2]))

    assert len(variants) == 4


def test_check_task_dataset_model_match():
    variants = list(gen_variants(dataset=["cora"], model=["gcn", "gat"], seed=[1, 2]))
    variants.append(namedtuple("Variant", ["dataset", "model", "seed"])(dataset="cora", model="deepwalk", seed=1))
    variants = check_task_dataset_model_match("node_classification", variants)

    assert len(variants) == 4


def test_experiment():
    results = experiment(
        task="node_classification", dataset="cora", model="gcn", hidden_size=32, max_epoch=10, cpu=True
    )

    assert ("cora", "gcn") in results
    assert results[("cora", "gcn")][0]["Acc"] > 0


def test_auto_experiment():
    def func_search_example(trial):
        return {
            "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
            "dropout": trial.suggest_uniform("dropout", 0.5, 0.9),
        }

    results = experiment(
        task="node_classification",
        dataset="cora",
        model="gcn",
        seed=[1, 2],
        n_trials=2,
        max_epoch=10,
        func_search=func_search_example,
        cpu=True,
    )

    assert ("cora", "gcn") in results
    assert results[("cora", "gcn")][0]["Acc"] > 0


if __name__ == "__main__":
    test_set_best_config()
    test_train()
    test_gen_variants()
    test_check_task_dataset_model_match()
    test_experiment()
    test_auto_experiment()
