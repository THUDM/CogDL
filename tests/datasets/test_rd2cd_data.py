from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_rd2cd_github():
    args = build_args_from_dict({"dataset": "Github"})
    assert args.dataset == "Github"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.num_nodes == 37700
    assert dataset.num_features == 4005


def test_rd2cd_elliptic():
    args = build_args_from_dict({"dataset": "Elliptic"})
    assert args.dataset == "Elliptic"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.num_nodes == 203769
    assert dataset.num_features == 164


def test_rd2cd_clothing():
    args = build_args_from_dict({"dataset": "Clothing"})
    assert args.dataset == "Clothing"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.num_nodes == 24919
    assert dataset.num_features == 9034


def test_rd2cd_electronics():
    args = build_args_from_dict({"dataset": "Electronics"})
    assert args.dataset == "Electronics"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.num_nodes == 42318
    assert dataset.num_features == 8669


def test_rd2cd_dblp():
    args = build_args_from_dict({"dataset": "Dblp"})
    assert args.dataset == "Dblp"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.num_nodes == 40672
    assert dataset.num_features == 7202


if __name__ == "__main__":
    test_rd2cd_github()
    test_rd2cd_elliptic()
    test_rd2cd_clothing()
    test_rd2cd_electronics()
    test_rd2cd_dblp()
