from cogdl.datasets import build_dataset

# HACK
class tmp:
    pass


def test_cora():
    args = tmp()
    args.dataset = "cora"
    cora = build_dataset(args)
    assert cora.data.num_nodes == 2708
    assert cora.data.num_edges == 10556
