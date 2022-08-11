import os.path as osp

import numpy as np

import torch
from cogdl.data import Graph, Dataset
from cogdl.utils import download_url


class BidirectionalOneShotIterator(object):


    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        """
        Transform a PyTorch Dataloader into python iterator
        """
        while True:
            for data in dataloader:
                yield data


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == "head-batch":
            tmp = [
                (0, rand_head) if (rand_head, relation, tail) not in self.triple_set else (-1, head)
                for rand_head in range(self.nentity)
            ]
            tmp[head] = (0, head)
        elif self.mode == "tail-batch":
            tmp = [
                (0, rand_tail) if (head, relation, rand_tail) not in self.triple_set else (-1, tail)
                for rand_tail in range(self.nentity)
            ]
            tmp[tail] = (0, tail)
        else:
            raise ValueError("negative batch mode %s not supported" % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == "head-batch":
                mask = np.in1d(negative_sample, self.true_head[(relation, tail)], assume_unique=True, invert=True)
            elif self.mode == "tail-batch":
                mask = np.in1d(negative_sample, self.true_tail[(head, relation)], assume_unique=True, invert=True)
            else:
                raise ValueError("Training batch mode %s not supported" % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[: self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        """
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        """
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        """
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        """

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail




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
            if "train" in filename:
                train_start_idx = len(triples)
            elif "valid" in filename:
                valid_start_idx = len(triples)
            elif "test" in filename:
                test_start_idx = len(triples)
            for line in f:
                items = line.strip().split()
                if len(items) != 3:
                    continue  
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
    url = "https://cloud.tsinghua.edu.cn/d/d1c733373b014efab986/files/?p=%2F{}%2F{}&dl=1"

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
