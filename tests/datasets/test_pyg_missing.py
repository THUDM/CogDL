from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_cora_missing():
    args = build_args_from_dict({'dataset': 'cora-missing-0'})
    assert args.dataset == 'cora-missing-0'
    cora = build_dataset(args)
    assert cora.data.num_nodes == 2708
    assert cora.data.num_edges == 10556

    args = build_args_from_dict({'dataset': 'cora-missing-20'})
    assert args.dataset == 'cora-missing-20'
    cora = build_dataset(args)
    assert cora.data.num_nodes == 2708
    assert cora.data.num_edges == 10556

    args = build_args_from_dict({'dataset': 'cora-missing-40'})
    assert args.dataset == 'cora-missing-40'
    cora = build_dataset(args)
    assert cora.data.num_nodes == 2708
    assert cora.data.num_edges == 10556

    args = build_args_from_dict({'dataset': 'cora-missing-60'})
    assert args.dataset == 'cora-missing-60'
    cora = build_dataset(args)
    assert cora.data.num_nodes == 2708
    assert cora.data.num_edges == 10556

    args = build_args_from_dict({'dataset': 'cora-missing-80'})
    assert args.dataset == 'cora-missing-80'
    cora = build_dataset(args)
    assert cora.data.num_nodes == 2708
    assert cora.data.num_edges == 10556

    args = build_args_from_dict({'dataset': 'cora-missing-100'})
    assert args.dataset == 'cora-missing-100'
    cora = build_dataset(args)
    assert cora.data.num_nodes == 2708
    assert cora.data.num_edges == 10556


if __name__ == "__main__":
    test_cora_missing()
