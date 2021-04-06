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


if __name__ == "__main__":
    test_dataset_stats()
    test_dataset_visual()
    test_oagbert()
