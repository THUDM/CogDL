from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_citeseer():
    args = build_args_from_dict({"dataset": "citeseer"})
    data = build_dataset(args)
    assert data.data.num_nodes == 3327
    assert data.num_features == 3703
    assert data.num_classes == 6


if __name__ == "__main__":
    test_citeseer()
