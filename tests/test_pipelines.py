import numpy as np
from cogdl import pipeline


def test_dataset_stats():
    stats = pipeline("dataset-stats")
    outputs = stats("cora")
    outputs = outputs[0]

    assert len(outputs) == 6
    assert tuple(outputs) == ("cora", 2708, 10184, 1433, 7, 140)


def test_dataset_visual():
    visual = pipeline("dataset-visual")
    outputs = visual("cora", seed=0, depth=3)

    assert len(outputs) == 72


def test_oagbert():
    oagbert = pipeline("oagbert", model="oagbert-test", load_weights=False)
    outputs = oagbert("CogDL is developed by KEG, Tsinghua.")

    assert len(outputs) == 2
    assert tuple(outputs[0].shape) == (1, 14, 32)
    assert tuple(outputs[1].shape) == (1, 32)


def test_gen_emb():
    generator = pipeline("generate-emb", model="prone")

    edge_index = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]])
    outputs = generator(edge_index)
    assert tuple(outputs.shape) == (4, 4)

    edge_weight = np.array([0.1, 0.3, 1.0, 0.8, 0.5])
    outputs = generator(edge_index, edge_weight)
    assert tuple(outputs.shape) == (4, 4)

    generator = pipeline("generate-emb", model="dgi", num_features=8, hidden_size=10)
    outputs = generator(edge_index, x=np.random.randn(4, 8))
    assert tuple(outputs.shape) == (4, 10)


if __name__ == "__main__":
    test_dataset_stats()
    test_dataset_visual()
    test_oagbert()
    test_gen_emb()
