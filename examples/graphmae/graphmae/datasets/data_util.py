
from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

from cogdl.datasets import build_dataset_from_name
from cogdl.utils import to_undirected



def preprocess(graph):
    feat = graph.x
    edge_index = graph.edge_index
    edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)
    graph.edge_index = edge_index
    graph.x = feat

    graph.add_remaining_self_loops()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    dataset = build_dataset_from_name(dataset_name)
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

def load_inductive_dataset(dataset_name):
    dataset = build_dataset_from_name(dataset_name)
    g = dataset[0]
    g.add_remaining_self_loops()

    feat = g.x
    num_features = feat.shape[1]
    
    train_g_nodes = g.nodes()[g.train_mask]
    train_g = g.subgraph(train_g_nodes)

    if dataset_name == "ppi-large":
        val_g_nodes = g.nodes()[g.val_mask]
        val_g = g.subgraph(val_g_nodes)

        test_g_nodes = g.nodes()[g.test_mask]
        test_g = g.subgraph(test_g_nodes)

        train_dataloader = [train_g]
        valid_dataloader = [val_g]
        test_dataloader = [test_g]
        eval_train_dataloader = [train_g]
        num_classes = train_g.y.shape[1]
    else:
        g.eval()
        train_dataloader = [train_g]
        valid_dataloader = test_dataloader = [g]
        eval_train_dataloader = [train_g]

        g.x = scale_feats(feat)
        num_classes = g.y.max().item() + 1
          
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes


def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.lower()
    dataset = build_dataset_from_name(dataset_name)
    graph = dataset[0]

    if not hasattr(graph, "x") or graph.x is None:
        print("Using degree as node features")
        feature_dim = 0
        degrees = []
        for g in dataset:
            feature_dim = max(feature_dim, g.degrees().max().item())
            degrees.extend(g.degrees().tolist())
        MAX_DEGREES = 400

        oversize = 0
        for d, n in Counter(degrees).items():
            if d > MAX_DEGREES:
                oversize += n
        # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
        feature_dim = min(feature_dim, MAX_DEGREES)

        feature_dim += 1
        for g in dataset:
            degrees = g.degrees()
            degrees[degrees > MAX_DEGREES] = MAX_DEGREES
            
            feat = F.one_hot(degrees.long(), num_classes=int(feature_dim)).float()
            g.x = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.x.shape[1]

    labels = torch.tensor([g.y for g in dataset])
    
    num_classes = torch.max(labels).item() + 1
    for g in dataset:
        g.add_remaining_self_loops()
    dataset = [(g, g.y) for g in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (int(feature_dim), int(num_classes))
