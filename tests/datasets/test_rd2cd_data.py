from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_rd2cd_github():
    args = build_args_from_dict({"dataset": "Github"})
    assert args.dataset == "Github"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.num_nodes == 37700
    assert dataset.num_features == 4005


if __name__ == "__main__":
    test_rd2cd_github()
