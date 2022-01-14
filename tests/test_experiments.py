from cogdl.experiments import experiment, gen_variants, train, set_best_config
from cogdl.options import get_default_args


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

    assert "(cora, gcn)" in results[0]
    assert results[0]["(cora, gcn)"][0]["test_acc"] > 0


def test_autognn_experiment():
    results = experiment(
        dataset="cora",
        model="autognn",
        seed=[1],
        n_trials=2,
        epochs=2,
        cpu=True,
    )
    
    assert "(cora, autognn)" in results[0]
    assert results[0]["(cora, autognn)"][0]["test_acc"] > 0


if __name__ == "__main__":
    test_set_best_config()
    test_train()
    test_gen_variants()
    test_experiment()
    test_auto_experiment()
    test_autognn_experiment()
