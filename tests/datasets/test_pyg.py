from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict

def test_cora():
    args = build_args_from_dict({'dataset': 'cora'})
    assert args.dataset == 'cora'
    cora = build_dataset(args)
    assert cora.data.num_nodes == 2708
    assert cora.data.num_edges == 10556

if __name__ == "__main__":
    test_cora()