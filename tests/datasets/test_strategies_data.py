from cogdl.data import Graph
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_strategies_dataset():
    args = build_args_from_dict({"dataset": "bbbp"})
    data = build_dataset(args)
    assert isinstance(data[0], Graph)


if __name__ == "__main__":
    test_strategies_dataset()
