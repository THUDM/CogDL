from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_gcc_kdd_icdm():
    args = build_args_from_dict({"dataset": "kdd_icdm"})
    assert args.dataset == "kdd_icdm"
    dataset = build_dataset(args)
    data = dataset.data
    assert data[0].edge_index[0].shape[0] == 17316
    assert data[1].edge_index[0].shape[0] == 10846


if __name__ == "__main__":
    test_gcc_kdd_icdm()
