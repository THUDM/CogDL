from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_dblp_ne():
    args = build_args_from_dict({"dataset": "dblp-ne"})
    data = build_dataset(args)
    assert data.data.num_nodes == 51264
    assert data.data.num_edges == 255936
    assert data.num_classes == 60


if __name__ == "__main__":
    test_dblp_ne()
