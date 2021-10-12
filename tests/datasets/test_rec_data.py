from cogdl.data import Graph
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_rec_dataset():
    args = build_args_from_dict({"dataset": "yelp2018"})
    data = build_dataset(args)
    assert isinstance(data[0], Graph)


if __name__ == "__main__":
    test_rec_dataset()
