import sys
import os.path as osp
from itertools import repeat
import networkx as nx
import numpy as np

import torch
from torch_sparse import coalesce
from cogdl.data import Data



def read_edgelist_label_data(folder, prefix):
    graph_path = osp.join(folder, '{}.ungraph'.format(prefix))
    cmty_path = osp.join(folder, '{}.cmty'.format(prefix))

    G = nx.read_edgelist(graph_path, nodetype=int, create_using=nx.Graph())
    num_node = G.number_of_nodes()
    print('edge number: ', num_node)
    with open(graph_path) as f:
        context = f.readlines()
        print('edge number: ', len(context))
        edge_index = np.zeros((2, len(context)))
        for i, line in enumerate(context):
            edge_index[:, i] = list(map(int, line.strip().split('\t')))
    edge_index = torch.from_numpy(edge_index).to(torch.int)

    with open(cmty_path) as f:
        context = f.readlines()
        print('class number: ', len(context))
        label = np.zeros((num_node, len(context)))

        for i, line in enumerate(context):
            line = map(int, line.strip().split('\t'))
            for node in line:
                label[node, i] = 1

    y = torch.from_numpy(label).to(torch.float)
    data = Data(x=None, edge_index=edge_index, y=y)

    return data