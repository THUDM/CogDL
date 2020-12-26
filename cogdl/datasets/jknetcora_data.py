from . import register_dataset
from cogdl.data import Data, Dataset
import os
import pickle
from torch_geometric.data import download_url
import os.path as osp
import torch
import numpy as np

@register_dataset("dgljknet-cora")
class DGICoraDataset(Dataset):
    # the original url of cora dataset
    url = "https://github.com/mori97/JKNet-dgl/raw/master/datasets/cora"

    def __init__(self, args=None):
        dataset = "dgljknet-cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        print(path)
        if not osp.exists(path):
            os.makedirs(path)
        super(DGICoraDataset, self).__init__(path)
        with open(self.processed_paths[0], "rb") as fin:
            load_data = pickle.load(fin)
        self.num_nodes = load_data["node_num"]

        data = Data()
        data.x = load_data["xs"]
        data.y = load_data["ys"]

        train_size = int(self.num_nodes * 0.8)
        train_mask = np.zeros((self.num_nodes,), dtype=bool)
        train_idx = np.random.choice(np.arange(self.num_nodes), size=train_size, replace=False)
        train_mask[train_idx] = True
        test_mask = np.ones((self.num_nodes,), dtype=bool)
        test_mask[train_idx] = False
        val_mask = test_mask

        edges = load_data["edges"]
        edges = np.array(edges, dtype=int).transpose((1, 0))

        data.edge_index = torch.from_numpy(edges)
        data.train_mask = torch.from_numpy(train_mask)
        data.test_mask = torch.from_numpy(test_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.x = torch.Tensor(data.x)
        data.y = torch.LongTensor(data.y)

        self.data = data
        self.num_classes = torch.max(self.data.y).item() + 1

    @property
    def raw_file_names(self):
        return ["cora.cites", "cora.content"]

    @property
    def processed_file_names(self):
        return ["cora.pkl"]

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        for name in self.raw_file_names:
            if not os.path.exists(os.path.join(self.raw_dir, name)):
                download_url("{}/{}".format(self.url, name), self.raw_dir)

    def process(self):
        class2index = {}
        paper2index = {}
        xs = []
        ts = []
        content_filename = osp.join(self.raw_dir, self.raw_file_names[1])
        cites_filename = osp.join(self.raw_dir, self.raw_file_names[0])

        with open(content_filename, "r") as f:
            for line in f:
                words = line.strip().split("\t")
                paper_id = words[0]
                word_attributes = list(map(float, words[1:-1]))
                class_label = words[-1]

                if paper_id not in paper2index:
                    paper2index[paper_id] = len(paper2index)
                if class_label not in class2index:
                    class2index[class_label] = len(class2index)

                xs.append(word_attributes)
                ts.append(class2index[class_label])

        node_num = len(xs)
        edges = []

        with open(cites_filename, "r") as f:
            for line in f:
                words = line.strip().split("\t")
                try:
                    src = paper2index[words[0]]
                    dst = paper2index[words[1]]
                    edges.append([src, dst])
                except KeyError:
                    continue

        xs = np.array(xs)
        ys = np.array(ts)

        data = {"xs": xs, "ys": ys, "node_num": node_num, "edges": edges}
        with open(self.processed_paths[0], "wb") as fout:
            pickle.dump(data, fout)
