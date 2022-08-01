from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_ogbn_arxiv():
    args = build_args_from_dict({"dataset": "ogbn-arxiv"})
    assert args.dataset == "ogbn-arxiv"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.num_nodes == 169343


def test_ogbg_molhiv():
    args = build_args_from_dict({"dataset": "ogbg-molhiv"})
    assert args.dataset == "ogbg-molhiv"
    dataset = build_dataset(args)
    assert dataset.all_edges == 2259376
    assert dataset.all_nodes == 1049163
    assert len(dataset.data) == 41127

def test_ogbl_ddi():
    args = build_args_from_dict({"dataset": "ogbl-ddi"})
    assert args.dataset == "ogbl-ddi"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.num_nodes == 4267

def test_ogbl_collab():
    args = build_args_from_dict({"dataset": "ogbl-collab"})
    assert args.dataset == "ogbl-collab"
    dataset = build_dataset(args)
    data = dataset.data
    assert data.num_nodes == 235868
    
if __name__ == "__main__":
    test_ogbn_arxiv()
    test_ogbg_molhiv()
    test_ogbl_ddi()
    test_ogbl_collab()