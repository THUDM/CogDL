
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from itertools import chain
from cogdl.data import Graph
from cogdl.utils.graph_utils import to_undirected, remove_self_loops

import utils
import json


def process_npz(path):
    with np.load(path) as f:
        x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']), f['attr_shape']).todense()
        x = torch.from_numpy(x).to(torch.float)
        x[x > 0] = 1

        adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                            f['adj_shape']).tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))

        y = torch.from_numpy(f['labels']).to(torch.long)

        return Graph(x=x, edge_index=edge_index, y=y)


def process_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        
        x = torch.tensor(data['features'], dtype=torch.float)
        y = torch.tensor(data['labels'], dtype=torch.long)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))

        train_mask = torch.tensor(data['train_masks'], dtype=torch.bool)
        train_mask = train_mask.t().contiguous()

        val_mask = torch.tensor(data['val_masks'], dtype=torch.bool)
        val_mask = val_mask.t().contiguous()

        test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)

        stopping_mask = torch.tensor(data['stopping_masks'], dtype=torch.bool)
        stopping_mask = stopping_mask.t().contiguous()

        return Graph(
            x=x, 
            y=y, 
            edge_index=edge_index, 
            train_mask=train_mask,
            val_mask=val_mask, 
            test_mask=test_mask,
            stopping_mask=stopping_mask
        )


def normalize_feature(data):
    feature = data.x
    feature = feature - feature.min()
    data.x = feature / feature.sum(dim=-1, keepdim=True).clamp_(min=1.)


def get_data(dataset):
    dataset_filepath = {
        "photo": "./data/Photo/raw/amazon_electronics_photo.npz",
        "computers": "./data/Computers/raw/amazon_electronics_computers.npz",
        "cs": "./data/CS/raw/ms_academic_cs.npz",
        "physics": "./data/Physics/raw/ms_academic_phy.npz",
        "WikiCS": "./data/WikiCS/raw/data.json"
    }
    assert dataset in dataset_filepath
    filepath = dataset_filepath[dataset]
    if dataset in ['WikiCS']:
        data = process_json(filepath)
        normalize_feature(data)
        std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
        data.x = (data.x - mean) / std
        data.edge_index = to_undirected(data.edge_index)
    else:
        data = process_npz(filepath)
        normalize_feature(data)

    data.add_remaining_self_loops()
    data.sym_norm()
        
    data = utils.create_masks(data=data)
    return data
 
