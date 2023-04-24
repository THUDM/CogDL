import os

import numpy as np
import torch

import cogdl
from cogdl.datasets import build_dataset_from_path
from cogdl.data import Graph

from .data_proc import preprocess, scale_feats
from utils import mask_edge

import logging
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def load_dataset(data_dir, dataset_name):
    dataset = build_dataset_from_path(data_dir, dataset=dataset_name)
    if dataset_name.startswith("ogbn"):
        graph = dataset[0]
        logging.info("--- to undirected graph ---")
        graph = preprocess(graph)
        feat = graph.x
        feat = scale_feats(feat)
        graph.x = feat
        graph.add_remaining_self_loops()

    # num_features = graph.x.shape[1]
    # num_classes = dataset.num_classes
    # return graph, (num_features, num_classes)
    # feats, graph, labels, split_idx
    train_idx = graph.train_mask.nonzero().squeeze(1)
    val_idx = graph.val_mask.nonzero().squeeze(1)
    test_idx = graph.test_mask.nonzero().squeeze(1)
    split_idx = {"train": train_idx, "valid": val_idx, "test": test_idx}
    return graph.x, graph, graph.y, split_idx


class LinearProbingDataLoader(DataLoader):
    def __init__(self, idx, feats, labels=None, **kwargs):
        self.labels = labels
        self.feats = feats

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=idx, **kwargs)

    def __collate_fn__(self, batch_idx):
        feats = self.feats[batch_idx]
        label = self.labels[batch_idx]

        return feats, label
    
class OnlineLCLoader(DataLoader):
    def __init__(self, root_nodes, graph, feats, labels=None, drop_edge_rate=0, **kwargs):
        self.graph = graph
        self.labels = labels
        self._drop_edge_rate = drop_edge_rate
        self.ego_graph_nodes = root_nodes
        self.feats = feats

        dataset = np.arange(len(root_nodes))
        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset, **kwargs)

    def drop_edge(self, g):
        if self._drop_edge_rate <= 0:
            return g, g
        n_node = g.num_nodes
        g = g.remove_self_loops()
        mask_index1 = mask_edge(g, self._drop_edge_rate)
        mask_index2 = mask_edge(g, self._drop_edge_rate)
        src = g.edge_index[0]
        dst = g.edge_index[1]
        nsrc1 = src[mask_index1]
        ndst1 = dst[mask_index1]
        nsrc2 = src[mask_index2]
        ndst2 = dst[mask_index2]
        g1 = Graph(edge_index=(nsrc1, ndst1), num_nodes=n_node)
        g1.add_remaining_self_loops()
        g2 = Graph(edge_index=(nsrc2, ndst2), num_nodes=n_node)
        g2.add_remaining_self_loops()

        return g1, g2

    def __collate_fn__(self, batch_idx):
        ego_nodes = [self.ego_graph_nodes[i] for i in batch_idx]
        subgs = [self.graph.subgraph(ego_nodes[i]) for i in range(len(ego_nodes))]
        
        sg = cogdl.data.batch_graphs(subgs)

        nodes = torch.from_numpy(np.concatenate(ego_nodes)).long()
        num_nodes = [x.shape[0] for x in ego_nodes]
        cum_num_nodes = np.cumsum([0] + num_nodes)[:-1]

        if self._drop_edge_rate > 0:
            drop_g1, drop_g2 = self.drop_edge(sg)

        sg = sg.add_remaining_self_loops()
        sg.x = self.feats[nodes]
        targets = torch.from_numpy(cum_num_nodes)

        if self.labels != None:
            label = self.labels[batch_idx]
        else:
            label = None
        
        if self._drop_edge_rate > 0:
            return sg, targets, label, nodes, drop_g1, drop_g2
        else:
            return sg, targets, label, nodes


def setup_training_data(dataset_name, data_dir, ego_graphs_file_path):
    feats, graph, labels, split_idx = load_dataset(data_dir, dataset_name)

    train_lbls = labels[split_idx["train"]]
    val_lbls = labels[split_idx["valid"]]
    test_lbls = labels[split_idx["test"]]

    labels = torch.cat([train_lbls, val_lbls, test_lbls])
    
    if not os.path.exists(ego_graphs_file_path):
        raise FileNotFoundError(f"{ego_graphs_file_path} doesn't exist")
    else:
        nodes = torch.load(ego_graphs_file_path)

    return feats, graph, labels, split_idx, nodes


def setup_training_dataloder(loader_type, training_nodes, graph, feats, batch_size, drop_edge_rate=0, pretrain_clustergcn=False, cluster_iter_data=None):
    num_workers = 8

    print(" -------- drop edge rate: {} --------".format(drop_edge_rate))
    dataloader = OnlineLCLoader(training_nodes, graph, feats=feats, drop_edge_rate=drop_edge_rate, batch_size=batch_size, shuffle=True, drop_last=False, persistent_workers=True, num_workers=num_workers)
    return dataloader


def setup_eval_dataloder(loader_type, graph, feats, ego_graph_nodes=None, batch_size=128, shuffle=False):
    num_workers = 8
    if loader_type == "lc":
        assert ego_graph_nodes is not None

    dataloader = OnlineLCLoader(ego_graph_nodes, graph, feats, batch_size=batch_size, shuffle=shuffle, drop_last=False, persistent_workers=True, num_workers=num_workers)
    return dataloader


def setup_finetune_dataloder(loader_type, graph, feats, ego_graph_nodes, labels, batch_size, shuffle=False):
    num_workers = 8
    dataloader = OnlineLCLoader(ego_graph_nodes, graph, feats, labels=labels, feats=feats, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers, persistent_workers=True)
    
    return dataloader
