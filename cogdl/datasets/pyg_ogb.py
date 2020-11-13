import os.path as osp
from tqdm import tqdm

import torch
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_sparse import coalesce
import numpy as np

from cogdl.data import Data, Dataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset

from . import register_dataset
class OGBNDataset(PygNodePropPredDataset):
    def __init__(self, root, name):
        super(OGBNDataset, self).__init__(name, root)
        
        self.data.num_nodes = self.data.num_nodes[0]
        #split
        split_index = self.get_idx_split()
        self.data['train_mask'] = torch.zeros(self.data.num_nodes, dtype = torch.bool)
        self.data['test_mask'] = torch.zeros(self.data.num_nodes, dtype = torch.bool)
        self.data['val_mask'] = torch.zeros(self.data.num_nodes, dtype = torch.bool)
        self.data['train_mask'][split_index['train']] = True
        self.data['test_mask'][split_index['test']] = True
        self.data['val_mask'][split_index['valid']] = True

        self.data.y = self.data.y.squeeze()
    
    def get(self, idx):
        assert idx == 0
        return self.data

@register_dataset("ogbn-arxiv")
class OGBArxivDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-arxiv"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            PygNodePropPredDataset(dataset, path)
        super(OGBArxivDataset, self).__init__(path, dataset)

        #to_symmetric
        rev_edge_index = self.data.edge_index[[1, 0]]
        edge_index = torch.cat([self.data.edge_index, rev_edge_index], dim = 1).to(dtype=torch.int64)
        self.data.edge_index, self.data.edge_attr = coalesce(edge_index, None, self.data.num_nodes, self.data.num_nodes)

@register_dataset("ogbn-products")
class OGBProductsDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-products"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            PygNodePropPredDataset(dataset, path)
        super(OGBArxivDataset, self).__init__(path, dataset)

@register_dataset("ogbn-proteins")
class OGBProductsDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-proteins"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            PygNodePropPredDataset(dataset, path)
        super(OGBArxivDataset, self).__init__(path, dataset)

@register_dataset("ogbn-mag")
class OGBProductsDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-mag"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            PygNodePropPredDataset(dataset, path)
        super(OGBArxivDataset, self).__init__(path, dataset)

@register_dataset("ogbn-papers100M")
class OGBPapers100MDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-papers100M"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            PygNodePropPredDataset(dataset, path)
        super(OGBArxivDataset, self).__init__(path, dataset)

class OGBGDataset(PygGraphPropPredDataset):
    def __init__(self, root, name):
        super(OGBGDataset, self).__init__(name, root)
        self.name = name

    def get_loader(self, args):
        split_index = self.get_idx_split()
        dataset = PygGraphPropPredDataset(self.name, osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", self.name))
        train_loader = DataLoader(dataset[split_index["train"]], batch_size = args.batch_size, shuffle = True)
        valid_loader = DataLoader(dataset[split_index["valid"]], batch_size = args.batch_size, shuffle = False)
        test_loader = DataLoader(dataset[split_index["test"]], batch_size = args.batch_size, shuffle = False)
        return train_loader, valid_loader, test_loader

    def get(self, idx):
        return self.data
    
@register_dataset("ogbg-molbace")
class OGBMolbaceDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-molbace"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            PygGraphPropPredDataset(dataset, path)
        super(OGBMolbaceDataset, self).__init__(path, dataset)

@register_dataset("ogbg-molhiv")
class OGBMolhivDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-molhiv"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            PygGraphPropPredDataset(dataset, path)
        super(OGBMolhivDataset, self).__init__(path, dataset)

@register_dataset("ogbg-molpcba")
class OGBMolpcbaDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-molpcba"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            PygGraphPropPredDataset(dataset, path)
        super(OGBMolpcbaDataset, self).__init__(path, dataset)

@register_dataset("ogbg-ppa")
class OGBPpaDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-ppa"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            PygGraphPropPredDataset(dataset, path)
        super(OGBPpaDataset, self).__init__(path, dataset)

@register_dataset("ogbg-code")
class OGBCodeDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-code"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            PygGraphPropPredDataset(dataset, path)
        super(OGBCodeDataset, self).__init__(path, dataset)
