from argparse import ArgumentParser
from typing import Dict, List
from unittest import mock
from unittest.mock import call, patch

import networkx as nx
import numpy as np
import torch
from cogdl.data import Graph
from cogdl.models.emb.deepwalk import DeepWalk


class Word2VecFake:
    def __init__(self, data: Dict[str, List[float]]) -> None:
        self.wv = data


embed_1 = [-0.1, 0.3, 0.5, 0.7]
embed_2 = [0.2, 0.4, 0.6, -0.8]
embed_3 = [0.3, 0.2, 0.1, -0.1]


def creator(walks, vector_size, window, min_count, sg, workers, epochs):
    return Word2VecFake({"0": embed_1, "1": embed_2, "2": embed_3})


class Args:
    hidden_size: int
    walk_length: int
    walk_num: int
    window_size: int
    worker: int
    iteration: int


def get_args():
    args = Args()
    args.hidden_size = 4
    args.walk_length = 5
    args.walk_num = 3
    args.window_size = 2
    args.worker = 777
    args.iteration = 10
    return args


def test_adds_correct_args():
    deep_walk_args = ["walk-length", "walk-num", "window-size", "worker", "iteration"]
    deep_walk_calls = [call(f"--{x}", type=int, default=mock.ANY, help=mock.ANY) for x in deep_walk_args]

    parser = ArgumentParser()
    with patch.object(parser, "add_argument", return_value=None) as mocked_method:
        DeepWalk.add_args(parser)
        mocked_method.assert_has_calls(deep_walk_calls)


def test_correctly_builds():
    args = get_args()
    model = DeepWalk.build_model_from_args(args)
    assert model.dimension == args.hidden_size
    assert model.walk_length == args.walk_length
    assert model.walk_num == args.walk_num
    assert model.window_size == args.window_size
    assert model.worker == args.worker
    assert model.iteration == args.iteration


def test_will_return_computed_embeddings_for_simple_fully_connected_graph():
    args = get_args()
    model: DeepWalk = DeepWalk.build_model_from_args(args)
    graph = Graph(edge_index=(torch.LongTensor([0]), torch.LongTensor([1])))
    trained = model(graph, creator)
    assert len(trained) == 2
    np.testing.assert_array_equal(trained[0], embed_1)
    np.testing.assert_array_equal(trained[1], embed_2)


def test_will_return_computed_embeddings_for_simple_graph():
    args = get_args()
    model: DeepWalk = DeepWalk.build_model_from_args(args)
    graph = Graph(edge_index=(torch.LongTensor([0, 1]), torch.LongTensor([1, 2])))
    trained = model(graph, creator)
    assert len(trained) == 3
    np.testing.assert_array_equal(trained[0], embed_1)
    np.testing.assert_array_equal(trained[1], embed_2)
    np.testing.assert_array_equal(trained[2], embed_3)


def test_will_pass_correct_number_of_walks():
    args = get_args()
    args.walk_num = 2
    model: DeepWalk = DeepWalk.build_model_from_args(args)
    graph = Graph(edge_index=(torch.LongTensor([0, 1]), torch.LongTensor([1, 2])))
    captured_walks_no = []

    def creator_mocked(walks, vector_size, window, min_count, sg, workers, epochs):
        captured_walks_no.append(len(walks))
        return creator(walks, vector_size, window, min_count, sg, workers, epochs)

    model(graph, creator_mocked)
    assert captured_walks_no[0] == args.walk_num * graph.num_nodes


if __name__ == "__main__":
    test_adds_correct_args()
    test_correctly_builds()
    test_will_return_computed_embeddings_for_simple_fully_connected_graph()
    test_will_return_computed_embeddings_for_simple_graph()
    test_will_pass_correct_number_of_walks()
