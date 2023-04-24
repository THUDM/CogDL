import logging
from typing import Counter
from xml.sax.handler import feature_string_interning
import numpy as np
from collections import namedtuple
import scipy.sparse as sp
from scipy.special import iv
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

import torch
import torch.nn.functional as F

from cogdl.utils import to_undirected
from cogdl.datasets import build_dataset_from_path
from sklearn.preprocessing import StandardScaler

def load_small_dataset(data_dir, dataset_name):
    dataset = build_dataset_from_path(data_dir, dataset=dataset_name)
    if dataset_name == "ogbn-arxiv":
        graph = dataset[0]
        feat = graph.x
        feat = scale_feats(feat)
        graph.x = feat
    else:
        graph = dataset[0]
        graph.add_remaining_self_loops()

    num_features = graph.x.shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)

def preprocess(graph):
    feat = graph.x
    edge_index = graph.edge_index
    edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)
    graph.edge_index = edge_index
    graph.x = feat

    graph.add_remaining_self_loops()
    return graph


def scale_feats(x):
    logging.info("### scaling features ###")
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
