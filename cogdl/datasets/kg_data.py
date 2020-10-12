from cogdl.datasets import register_dataset
import os
import os.path as osp
import sys
import torch

from cogdl.data import Data, Dataset, download_url


def read_triplet_data(folder):
    filenames = ["train2id.txt", "valid2id.txt", "test2id.txt"]
    count = 0
    edge_index = []
    edge_attr = []
    count_list = []
    for filename in filenames:
        with open(osp.join(folder, filename), "r") as f:
            num = int(f.readline().strip())
            for line in f:
                items = line.strip().split()
                edge_index.append([int(items[0]), int(items[1])])
                edge_attr.append(int(items[2]))
                count += 1
            count_list.append(count)


    edge_index = torch.LongTensor(edge_index).t()
    edge_attr = torch.LongTensor(edge_attr)
    data = Data()
    data.edge_index = edge_index
    data.edge_attr = edge_attr

    def generate_mask(start, end):
        mask = torch.BoolTensor(count)
        mask[:] = False
        mask[start:end] = True
        return mask
    data.train_mask = generate_mask(0, count_list[0])
    data.val_mask = generate_mask(count_list[0], count_list[1])
    data.test_mask = generate_mask(count_list[1], count_list[2])
    return data


class KnowledgeGraphDataset(Dataset):
    url = "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks"

    def __init__(self, root, name):
        self.name = name
        super(KnowledgeGraphDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ["train2id.txt", "valid2id.txt", "test2id.txt"]
        return names

    @property
    def processed_file_names(self):
        return ["data.pt"]
    
    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        for name in self.raw_file_names:
            download_url("{}/{}/{}".format(self.url, self.name, name), self.raw_dir)
        
    def process(self):
        data = read_triplet_data(self.raw_dir)
        torch.save(data, self.processed_paths[0])


@register_dataset("fb13")
class FB13Datset(KnowledgeGraphDataset):
    def __init__(self):
        dataset = "FB13"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(FB13Datset, self).__init__(path, dataset)


@register_dataset("fb15k")
class FB15kDatset(KnowledgeGraphDataset):
    def __init__(self):
        dataset = "FB15K"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(FB15kDatset, self).__init__(path, dataset)


@register_dataset("fb15k237")
class FB15k237Datset(KnowledgeGraphDataset):
    def __init__(self):
        dataset = "FB15K237"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(FB15k237Datset, self).__init__(path, dataset)


@register_dataset("wn18")
class WN18Datset(KnowledgeGraphDataset):
    def __init__(self):
        dataset = "WN18"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(WN18Datset, self).__init__(path, dataset)


@register_dataset("wn18rr")
class WN18RRDataset(KnowledgeGraphDataset):
    def __init__(self):
        dataset = "WN18RR"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(WN18RRDataset, self).__init__(path, dataset)
