import os.path as osp

import torch
from cogdl.data import Graph, Dataset
from cogdl.utils import download_url


def read_triplet_data(folder):
    filenames = ["train2id.txt", "valid2id.txt", "test2id.txt"]
    count = 0
    edge_index = []
    edge_attr = []
    count_list = []
    triples = []
    num_entities = 0
    num_relations = 0
    entity_dic = {}
    relation_dic = {}
    for filename in filenames:
        with open(osp.join(folder, filename), "r") as f:
            _ = int(f.readline().strip())
            if "train" in filename:
                train_start_idx = len(triples)
            elif "valid" in filename:
                valid_start_idx = len(triples)
            elif "test" in filename:
                test_start_idx = len(triples)
            for line in f:
                items = line.strip().split()
                edge_index.append([int(items[0]), int(items[1])])
                edge_attr.append(int(items[2]))
                triples.append((int(items[0]), int(items[2]), int(items[1])))
                if items[0] not in entity_dic:
                    entity_dic[items[0]] = num_entities
                    num_entities += 1
                if items[1] not in entity_dic:
                    entity_dic[items[1]] = num_entities
                    num_entities += 1
                if items[2] not in relation_dic:
                    relation_dic[items[2]] = num_relations
                    num_relations += 1
                count += 1
            count_list.append(count)

    edge_index = torch.LongTensor(edge_index).t()
    edge_attr = torch.LongTensor(edge_attr)
    data = Graph()
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
    return data, triples, train_start_idx, valid_start_idx, test_start_idx, num_entities, num_relations


class KnowledgeGraphDataset(Dataset):
    # url = "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks"
    url = "https://cloud.tsinghua.edu.cn/d/b567292338f2488699b7/files/?p=%2F{}%2F{}&dl=1"

    def __init__(self, root, name):
        self.name = name
        super(KnowledgeGraphDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])
        triple_config = torch.load(self.processed_paths[1])
        self.triples = triple_config["triples"]
        self._train_start_index = triple_config["train_start_index"]
        self._valid_start_index = triple_config["valid_start_index"]
        self._test_start_index = triple_config["test_start_index"]
        self._num_entities = triple_config["num_entities"]
        self._num_relations = triple_config["num_relations"]

    @property
    def raw_file_names(self):
        names = ["train2id.txt", "valid2id.txt", "test2id.txt"]
        return names

    @property
    def processed_file_names(self):
        return ["data.pt", "triple_config.pt"]

    @property
    def train_start_idx(self):
        return self._train_start_index

    @property
    def valid_start_idx(self):
        return self._valid_start_index

    @property
    def test_start_idx(self):
        return self._test_start_index

    @property
    def num_entities(self):
        return self._num_entities

    @property
    def num_relations(self):
        return self._num_relations

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        for name in self.raw_file_names:
            # download_url("{}/{}/{}".format(self.url, self.name, name), self.raw_dir)
            download_url(self.url.format(self.name, name), self.raw_dir, name=name)

    def process(self):
        (
            data,
            triples,
            train_start_index,
            valid_start_index,
            test_start_index,
            num_entities,
            num_relations,
        ) = read_triplet_data(self.raw_dir)
        torch.save(data, self.processed_paths[0])
        triple_config = {
            "triples": triples,
            "train_start_index": train_start_index,
            "valid_start_index": valid_start_index,
            "test_start_index": test_start_index,
            "num_entities": num_entities,
            "num_relations": num_relations,
        }
        torch.save(triple_config, self.processed_paths[1])


class FB13Datset(KnowledgeGraphDataset):
    def __init__(self, data_path="data"):
        dataset = "FB13"
        path = osp.join(data_path, dataset)
        super(FB13Datset, self).__init__(path, dataset)


class FB15kDatset(KnowledgeGraphDataset):
    def __init__(self, data_path="data"):
        dataset = "FB15K"
        path = osp.join(data_path, dataset)
        super(FB15kDatset, self).__init__(path, dataset)


class FB15k237Datset(KnowledgeGraphDataset):
    def __init__(self, data_path="data"):
        dataset = "FB15K237"
        path = osp.join(data_path, dataset)
        super(FB15k237Datset, self).__init__(path, dataset)


class WN18Datset(KnowledgeGraphDataset):
    def __init__(self, data_path="data"):
        dataset = "WN18"
        path = osp.join(data_path, dataset)
        super(WN18Datset, self).__init__(path, dataset)


class WN18RRDataset(KnowledgeGraphDataset):
    def __init__(self, data_path="data"):
        dataset = "WN18RR"
        path = osp.join(data_path, dataset)
        super(WN18RRDataset, self).__init__(path, dataset)


class FB13SDatset(KnowledgeGraphDataset):
    def __init__(self, data_path="data"):
        dataset = "FB13S"
        path = osp.join(data_path, dataset)
        super(FB13SDatset, self).__init__(path, dataset)
