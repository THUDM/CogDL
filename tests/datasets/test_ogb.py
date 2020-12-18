from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_ogbn_arxiv():
    args = build_args_from_dict({"dataset": "ogbn-arxiv"})
    assert args.dataset == "ogbn-arxiv"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.num_nodes == 169343
    assert data.num_edges == 2315598


def test_ogbg_molhiv():
    args = build_args_from_dict({"dataset": "ogbg-molhiv"})
    assert args.dataset == "ogbg-molhiv"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.edge_index.shape[1] == 2259376
    assert data.x.shape[0] == 1049163
    assert data.y.shape[0] == 41127


if __name__ == "__main__":
    test_ogbn_arxiv()
    test_ogbg_molhiv()
