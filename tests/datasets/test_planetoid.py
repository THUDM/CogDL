from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_citeseer():
    args = build_args_from_dict({"dataset": "cora"})
    data = build_dataset(args)
    assert data.data.num_nodes == 3372
    assert data.data.num_edges == 9216
    assert data.num_features == 3702
    assert data.num_classes == 6


if __name__ == "__main__":
    test_citeseer()
