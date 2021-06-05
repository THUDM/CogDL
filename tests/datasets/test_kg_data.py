from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_fb13():
    args = build_args_from_dict({"dataset": "fb13"})
    assert args.dataset == "fb13"
    dataset = build_dataset(args)
    data = dataset.data
    assert len(data.edge_index) == 2
    assert data.edge_index[0].shape[0] == 345873
    assert data.edge_attr.shape[0] == 345873


def test_fb15k():
    args = build_args_from_dict({"dataset": "fb15k"})
    assert args.dataset == "fb15k"
    dataset = build_dataset(args)
    data = dataset.data
    assert len(data.edge_index) == 2
    assert data.edge_index[0].shape[0] == 592213
    assert data.edge_attr.shape[0] == 592213


def test_fb15k237():
    args = build_args_from_dict({"dataset": "fb15k237"})
    assert args.dataset == "fb15k237"
    dataset = build_dataset(args)
    data = dataset.data
    assert len(data.edge_index) == 2
    assert data.edge_index[0].shape[0] == 310116
    assert data.edge_attr.shape[0] == 310116


if __name__ == "__main__":
    test_fb13()
    test_fb15k()
    test_fb15k237()
