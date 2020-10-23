import json
import os
import os.path as osp
from itertools import product
import subprocess

import numpy as np
import scipy.io
import torch

from cogdl.data import Data, Dataset, download_url
from cogdl.data.makedirs import makedirs

from . import register_dataset

def files_exist(files):
    return all([osp.exists(f) for f in files])

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
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
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

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
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

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

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
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
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


class KnowledgeGraph(Dataset):
    def __init__(self, root, name, url):
        self.name = name
        self.url = url
        self.order_sop = False
        self.negative_sample_size = 5

        super(KnowledgeGraph, self).__init__(root)
    
    def download(self):
        download_url(self.url, self.raw_dir)
        compressed = osp.join(self.raw_dir, self.name + ".tar.gz")
        if osp.exists(compressed):
            cmds = ["tar", "xvf", compressed, "-C", self.raw_dir]
            subprocess.run(cmds)
            rm_cmd = ["rm", compressed]
            subprocess.run(rm_cmd)
        self.linking_or_rename()

    @property
    def raw_file_names(self):
        names = ["train.txt","valid.txt","test.txt"]
        return names 
    
    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def train_start_idx(self):
        return self._train_start_idx
    
    @property
    def valid_start_idx(self):
        return self._valid_start_idx
    
    @property
    def test_start_idx(self):
        return self._test_start_idx
    
    @property
    def num_entities(self):
        return self._num_entities

    @property
    def num_train_entities(self):
        return self._num_train_entities

    @property 
    def num_relations(self):
        return self._num_relations

    @property
    def tuples(self):
        return self._tuple_dict
    
    @property
    def objects_dict(self):
        return self._o_rel_dict
    
    @property 
    def subjects_dict(self):
        return self._s_rel_dict

    @property
    def train_objects_dict(self):
        return self._o_train_dict
    
    @property 
    def train_subjects_dict(self):
        return self._s_train_dict

    def get(self, idx):
        return self.tuple[idx]

    def linking_or_rename(self):
        raise NotImplementedError

    def _process(self):
        if files_exist(self.processed_paths) and False:
            print("detect existing processed data")
            datas = torch.load(self.processed_paths[0])
            self.entities = datas["entities"]
            self.relations = datas["relations"]
            self.tuple = datas["tuples"]
            self._test_start_idx = datas["test_start_idx"]
            self._train_start_idx = datas["train_start_idx"]
            self._valid_start_idx = datas["valid_start_idx"]
            self._num_train_entities = datas["number_train_entities"]
            self._num_entities = len(self.entities)
            self._num_relations = len(self.relations)
            self._tuple_dict = {}
            # to record which subjects/objects can exist given object/subject and predication
            self._s_rel_dict = {}
            self._o_rel_dict = {}
            self._o_train_dict = {}
            self._s_train_dict = {}

            for idx, tup in enumerate(self.tuple):
                self._tuple_dict[tup[0],tup[1],tup[2]] = 1
                if (tup[0], tup[1]) not in self._o_rel_dict:
                    self._o_rel_dict[tup[0], tup[1]] = []
                self._o_rel_dict[tup[0], tup[1]].append(tup[2])
                if (tup[1], tup[2]) not in self._s_rel_dict:
                    self._s_rel_dict[tup[1], tup[2]] = []
                self._s_rel_dict[tup[1], tup[2]].append(tup[0])
                if idx < self._valid_start_idx:
                    if (tup[0], tup[1]) not in self._o_train_dict:
                        self._o_train_dict[tup[0], tup[1]] = []
                    self._o_train_dict[tup[0], tup[1]].append(tup[2])
                    if (tup[1], tup[2]) not in self._s_train_dict:
                        self._s_train_dict[tup[1], tup[2]] = []
                    self._s_train_dict[tup[1], tup[2]].append(tup[0])
             
            print("load processed data successfully")
            return 

        print("Processing...")

        #makedirs(self.processed_dir)
        self.process()

        print("Done!")
    
    def process(self):
        split_files = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}
        if self.order_sop:
            S, P, O = 0, 2, 1
        else:
            S, P, O = 0, 1, 2
        raw = {}
        self.entities = {}
        self.relations = {}
        ent_id = 0
        rel_id = 0
        self.tuple = []
        self._num_train_entities = 0
        for split, filename in split_files.items():
            with open(osp.join(self.raw_dir, filename), "r") as f:
                raw[split] = list(map(lambda s: s.strip().split("\t"), f.readlines()))
                
                if "train" in split:
                    self._train_start_idx = len(self.tuple)
                elif "valid" in split:
                    self._valid_start_idx = len(self.tuple)
                elif "test" in split:
                    self._test_start_idx = len(self.tuple)
                
                for t in raw[split]:
                    if t[S] not in self.entities:
                        self.entities[t[S]] = ent_id
                        ent_id += 1
                    if t[P] not in self.relations:
                        self.relations[t[P]] = rel_id
                        rel_id += 1
                    if t[O] not in self.entities:
                        self.entities[t[O]] = ent_id
                        ent_id += 1
                    
                    self.tuple.append((self.entities[t[S]], self.relations[t[P]], self.entities[t[O]]))
                
                if "train" in split:
                    self._num_train_entities = ent_id

                print(
                    f"Found {len(raw[split])} triples in {split} split "
                    f"(file: {filename})."
                )
        
        print(f"{len(self.relations)} distinct relations")
        print(f"{len(self.entities)} distinct entities")
        self._num_entities = len(self.entities)
        self._num_relations = len(self.relations)
        self._tuple_dict = {}
        # to record which subjects/objects can exist given object/subject and predication
        self._s_rel_dict = {}
        self._o_rel_dict = {}
        self._o_train_dict = {}
        self._s_train_dict = {}

        for idx, tup in enumerate(self.tuple):
            self._tuple_dict[tup[0],tup[1],tup[2]] = 1
            if (tup[0], tup[1]) not in self._o_rel_dict:
                self._o_rel_dict[tup[0], tup[1]] = []
            self._o_rel_dict[tup[0], tup[1]].append(tup[2])
            if (tup[1], tup[2]) not in self._s_rel_dict:
                self._s_rel_dict[tup[1], tup[2]] = []
            self._s_rel_dict[tup[1], tup[2]].append(tup[0])
            if idx < self._valid_start_idx:
                if (tup[0], tup[1]) not in self._o_train_dict:
                    self._o_train_dict[tup[0], tup[1]] = []
                self._o_train_dict[tup[0], tup[1]].append(tup[2])
                if (tup[1], tup[2]) not in self._s_train_dict:
                    self._s_train_dict[tup[1], tup[2]] = []
                self._s_train_dict[tup[1], tup[2]].append(tup[0])

        datas = {"entities":self.entities, "relations":self.relations, "tuples":self.tuple,
                "train_start_idx":self._train_start_idx, "valid_start_idx":self._valid_start_idx, "test_start_idx":self._test_start_idx, "number_train_entities":self._num_train_entities}

        torch.save(datas, self.processed_paths[0])


@register_dataset("triple-fb15k")
class FB15KDataset(KnowledgeGraph):
    
    def __init__(self):
        self.name = "fb15k"
        self.url = "http://web.informatik.uni-mannheim.de/pi1/kge-datasets/fb15k.tar.gz"
        super(FB15KDataset, self).__init__("data/fb15k", self.name, self.url)
    
    def linking_or_rename(self):
        raw_test_file, raw_train_file, raw_valid_file = "freebase_mtr100_mte100-test.txt", "freebase_mtr100_mte100-train.txt", "freebase_mtr100_mte100-valid.txt"
        raw_test_file = osp.join(self.raw_dir, self.name, raw_test_file)
        raw_train_file = osp.join(self.raw_dir, self.name, raw_train_file)
        raw_valid_file = osp.join(self.raw_dir, self.name, raw_valid_file)
        rename_test_cmds = ["mv", raw_test_file, osp.join(self.raw_dir, "test.txt")]
        rename_train_cmds = ["mv", raw_train_file, osp.join(self.raw_dir, "train.txt")]
        rename_valid_cmds = ["mv", raw_valid_file, osp.join(self.raw_dir, "valid.txt")]
        subprocess.run(rename_test_cmds)
        subprocess.run(rename_train_cmds)
        subprocess.run(rename_valid_cmds)

@register_dataset("triple-fb15k-237")
class FB15K237Dataset(KnowledgeGraph):
    
    def __init__(self):
        self.name = "fb15k-237"
        self.url = "http://web.informatik.uni-mannheim.de/pi1/kge-datasets/fb15k-237.tar.gz"
        super(FB15K237Dataset, self).__init__("data/fb15k-237", self.name, self.url)
    
    def linking_or_rename(self):
        raw_test_file, raw_train_file, raw_valid_file = "test.txt", "train.txt", "valid.txt"
        raw_test_file = osp.join(self.raw_dir, self.name, raw_test_file)
        raw_train_file = osp.join(self.raw_dir, self.name, raw_train_file)
        raw_valid_file = osp.join(self.raw_dir, self.name, raw_valid_file)
        rename_test_cmds = ["mv", raw_test_file, osp.join(self.raw_dir, "test.txt")]
        rename_train_cmds = ["mv", raw_train_file, osp.join(self.raw_dir, "train.txt")]
        rename_valid_cmds = ["mv", raw_valid_file, osp.join(self.raw_dir, "valid.txt")]
        subprocess.run(rename_test_cmds)
        subprocess.run(rename_train_cmds)
        subprocess.run(rename_valid_cmds)

@register_dataset("triple-wn18")
class WN18Dataset(KnowledgeGraph):
    
    def __init__(self):
        self.name = "wn18"
        self.url = "http://web.informatik.uni-mannheim.de/pi1/kge-datasets/wn18.tar.gz"
        super(WN18Dataset, self).__init__("data/wn18", self.name, self.url)
    
    def linking_or_rename(self):
        raw_test_file, raw_train_file, raw_valid_file = "wordnet-mlj12-test.txt", "wordnet-mlj12-train.txt", "wordnet-mlj12-valid.txt"
        raw_test_file = osp.join(self.raw_dir, self.name, raw_test_file)
        raw_train_file = osp.join(self.raw_dir, self.name, raw_train_file)
        raw_valid_file = osp.join(self.raw_dir, self.name, raw_valid_file)
        rename_test_cmds = ["mv", raw_test_file, osp.join(self.raw_dir, "test.txt")]
        rename_train_cmds = ["mv", raw_train_file, osp.join(self.raw_dir, "train.txt")]
        rename_valid_cmds = ["mv", raw_valid_file, osp.join(self.raw_dir, "valid.txt")]
        subprocess.run(rename_test_cmds)
        subprocess.run(rename_train_cmds)
        subprocess.run(rename_valid_cmds)

@register_dataset("triple-wn18rr")
class WNRRDataset(KnowledgeGraph):
    
    def __init__(self):
        self.name = "wnrr"
        self.url = "http://web.informatik.uni-mannheim.de/pi1/kge-datasets/wnrr.tar.gz"
        super(WNRRDataset, self).__init__("data/wnrr", self.name, self.url)
    
    def linking_or_rename(self):
        raw_test_file, raw_train_file, raw_valid_file = "test.txt", "train.txt", "valid.txt"
        raw_test_file = osp.join(self.raw_dir, self.name, raw_test_file)
        raw_train_file = osp.join(self.raw_dir, self.name, raw_train_file)
        raw_valid_file = osp.join(self.raw_dir, self.name, raw_valid_file)
        rename_test_cmds = ["mv", raw_test_file, osp.join(self.raw_dir, "test.txt")]
        rename_train_cmds = ["mv", raw_train_file, osp.join(self.raw_dir, "train.txt")]
        rename_valid_cmds = ["mv", raw_valid_file, osp.join(self.raw_dir, "valid.txt")]
        subprocess.run(rename_test_cmds)
        subprocess.run(rename_train_cmds)
        subprocess.run(rename_valid_cmds)

@register_dataset("triple-yago3-10")
class YAGO310Dataset(KnowledgeGraph):
    
    def __init__(self):
        self.name = "yago3-10"
        self.url = "http://web.informatik.uni-mannheim.de/pi1/kge-datasets/yago3-10.tar.gz"
        super(YAGO310Dataset, self).__init__("data/yago3-10", self.name, self.url)
    
    def linking_or_rename(self):
        raw_test_file, raw_train_file, raw_valid_file = "test.txt", "train.txt", "valid.txt"
        raw_test_file = osp.join(self.raw_dir, self.name, raw_test_file)
        raw_train_file = osp.join(self.raw_dir, self.name, raw_train_file)
        raw_valid_file = osp.join(self.raw_dir, self.name, raw_valid_file)
        rename_test_cmds = ["mv", raw_test_file, osp.join(self.raw_dir, "test.txt")]
        rename_train_cmds = ["mv", raw_train_file, osp.join(self.raw_dir, "train.txt")]
        rename_valid_cmds = ["mv", raw_valid_file, osp.join(self.raw_dir, "valid.txt")]
        subprocess.run(rename_test_cmds)
        subprocess.run(rename_train_cmds)
        subprocess.run(rename_valid_cmds)

if __name__=="__main__":
    fb15k = FB15KDataset()
    wn18 = WN18Dataset()
    wnrr = WNRRDataset()
    yago3_10 = YAGO310Dataset()