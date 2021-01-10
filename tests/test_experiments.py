from collections import namedtuple

from cogdl.exp import check_task_dataset_model_match, experiment, gen_variants, train
from cogdl.options import get_default_args


def test_train():
    args = get_default_args(task="node_classification", dataset="cora", model="gcn", cpu=True)
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
        task="node_classification", dataset="cora", model="gcn", hidden_size=32, max_epoch=200, cpu=True
    )

    assert ("cora", "gcn") in results
    assert results[("cora", "gcn")][0]["Acc"] > 0


if __name__ == "__main__":
    test_train()
    test_gen_variants()
    test_check_task_dataset_model_match()
    test_experiment()
