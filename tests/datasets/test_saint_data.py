import os
import json
import numpy as np
import scipy.sparse as sp
from cogdl.datasets.saint_data import SAINTDataset


def build_test_data():
    os.makedirs("data", exist_ok=True)
    os.makedirs("./data/test_saint", exist_ok=True)
    os.makedirs("./data/test_saint/raw", exist_ok=True)
    prefix = "./data/test_saint/raw"

    def join(x):
        return os.path.join(prefix, x)

    num_nodes = 100
    train_edge = np.random.randint(0, num_nodes, (2, 200))
    all_edge = np.random.randint(0, num_nodes, (2, 200))
    all_edge = np.concatenate([train_edge, all_edge], axis=1)
    adj_train = sp.csr_matrix((np.ones(200), (train_edge[0], train_edge[1])), shape=(num_nodes, num_nodes))
    adj_full = sp.csr_matrix((np.ones(400), (all_edge[0], all_edge[1])), shape=(num_nodes, num_nodes))
    sp.save_npz(join("adj_train.npz"), adj_train)
    sp.save_npz(join("adj_full.npz"), adj_full)

    feats = np.random.rand(100, 10)
    class_map = [(str(i), np.random.randint(0, 1, 10).astype(np.float).tolist()) for i in range(num_nodes)]
    class_map = dict(class_map)
    roles = {"tr": list(range(40)), "va": list(range(40, 80)), "te": list(range(80, 100))}
    np.save(join("feats.npy"), feats)
    with open(join("class_map.json"), "w") as f:
        json.dump(class_map, f)
    with open(join("role.json"), "w") as f:
        json.dump(roles, f)


def test_saint_data():
    build_test_data()
    dataset = "test_saint"
    path = os.path.join("./data/", dataset)
    dataset = SAINTDataset(path, dataset)
    assert dataset.num_features == 10
    assert dataset.num_classes == 10


if __name__ == "__main__":
    test_saint_data()
