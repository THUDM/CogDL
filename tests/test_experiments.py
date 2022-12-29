from cogdl.experiments import experiment, gen_variants, train, set_best_config
from cogdl.options import get_default_args
from examples.simple_trafficPre.example import experiment as traffic_experiment
from cogdl.utils import makedirs
import numpy as np
import os

def test_set_best_config():
    args = get_default_args(task="node_classification", dataset="citeseer", model="gat")
    args.model = args.model[0]
    args.dataset = args.dataset[0]
    args = set_best_config(args)

    assert args.lr == 0.005
    assert args.epochs == 1000
    assert args.weight_decay == 0.001


def test_train():
    args = get_default_args(dataset="cora", model="gcn", epochs=10, cpu=True)
    args.dataset = args.dataset[0]
    args.model = args.model[0]
    args.seed = args.seed[0]
    result = train(args)

    assert "test_acc" in result
    assert result["test_acc"] > 0


def test_gen_variants():
    variants = list(gen_variants(dataset=["cora"], model=["gcn", "gat"], seed=[1, 2]))

    assert len(variants) == 4


def test_experiment():
    results = experiment(task="node_classification", dataset="cora", model="gcn", hidden_size=32, epochs=10, cpu=True)

    assert ("cora", "gcn") in results
    assert results[("cora", "gcn")][0]["test_acc"] > 0


def test_auto_experiment():
    def search_space_example(trial):
        return {
            "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
            "dropout": trial.suggest_uniform("dropout", 0.5, 0.9),
        }

    results = experiment(
        dataset="cora",
        model="gcn",
        seed=[1, 2],
        n_trials=2,
        epochs=10,
        search_space=search_space_example,
        cpu=True,
    )

    assert ("cora", "gcn") in results
    assert results[("cora", "gcn")][0]["test_acc"] > 0


def test_autognn_experiment():
    results = experiment(
        dataset="cora",
        model="autognn",
        seed=[1],
        n_trials=10,
        epochs=2,
        cpu=True,
    )
    assert ("cora", "autognn") in results
    assert results[("cora", "autognn")][0]["test_acc"] > 0


def test_stgcn_experiment():   
    root_path = os.path.dirname(os.path.abspath(__file__))
    test_path = root_path + "/test_stgcn/"

    if not os.path.exists(test_path):
        makedirs(test_path)

    kwargs = {"epochs":1,
              "kernel_size":3,
              "n_his":20,
              "n_pred":1,
              "channel_size_list":np.array([[ 1, 4, 4],[4, 4, 4],[4, 4, 4]]),
              "num_layers":3,
              "num_nodes":288,
              "train_prop": 0.1,
              "val_prop": 0.1,
              "test_prop": 0.1,
              "pred_length":288,}

    results =traffic_experiment(
        dataset="pems-stgcn",
        model="stgcn",
        resume_training=False,
        cpu=True,
        **kwargs
    )
    assert ("pems-stgcn", "stgcn") in results
    assert results[("pems-stgcn", "stgcn")][0]["test__metric"] > 0

if __name__ == "__main__":
    test_set_best_config()
    test_train()
    test_gen_variants()
    test_experiment()
    test_auto_experiment()
    test_autognn_experiment()
    test_stgcn_experiment()
