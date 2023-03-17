from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def test_chameleon():
    args = build_args_from_dict({"dataset": "chameleon"})
    data = build_dataset(args)
    assert data.num_nodes == 2277
    assert data.num_features == 2325
    assert data.num_classes == 5


def test_cornell():
    args = build_args_from_dict({"dataset": "cornell"})
    data = build_dataset(args)
    assert data.num_nodes == 183
    assert data.num_features == 1703
    assert data.num_classes == 5


def test_film():
    args = build_args_from_dict({"dataset": "film"})
    data = build_dataset(args)
    assert data.num_nodes == 7600
    assert data.num_features == 932
    assert data.num_classes == 5


def test_squirrel():
    args = build_args_from_dict({"dataset": "squirrel"})
    data = build_dataset(args)
    assert data.num_nodes == 5201
    assert data.num_features == 2089
    assert data.num_classes == 5


def test_texas():
    args = build_args_from_dict({"dataset": "texas"})
    data = build_dataset(args)
    assert data.num_nodes == 183
    assert data.num_features == 1703
    assert data.num_classes == 5


def test_wisconsin():
    args = build_args_from_dict({"dataset": "wisconsin"})
    data = build_dataset(args)
    assert data.num_nodes == 251
    assert data.num_features == 1703
    assert data.num_classes == 5


def test_citeseer_geom():
    args = build_args_from_dict({"dataset": "citeseer_geom"})
    data = build_dataset(args)
    assert data.data.num_nodes == 3327
    assert data.num_features == 3703
    assert data.num_classes == 6


if __name__ == "__main__":
    test_chameleon()
    test_cornell()
    test_film()
    test_squirrel()
    test_texas()
    test_wisconsin()
    test_citeseer_geom()
