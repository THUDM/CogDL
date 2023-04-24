import argparse
from collections import namedtuple
import multiprocessing
import os

import numpy as np
from localgraphclustering import *
from scipy.sparse import csr_matrix
from ogb.nodeproppred import DglNodePropPredDataset
import torch
import logging

import dgl
from dgl.data import load_data


def my_sweep_cut(g, node):
    vol_sum = 0.0
    in_edge = 0.0
    conds = np.zeros_like(node, dtype=np.float32)
    for i in range(len(node)):
        idx = node[i]
        vol_sum += g.d[idx]
        denominator = min(vol_sum, g.vol_G - vol_sum)
        if denominator == 0.0:
            denominator = 1.0
        in_edge += 2*sum([g.adjacency_matrix[idx,prev] for prev in node[:i+1]])
        cut = vol_sum - in_edge
        conds[i] = cut/denominator
    return conds


def calc_local_clustering(args):
    i, log_steps, num_iter, ego_size, method = args
    if i % log_steps == 0:
        print(i)
    node, ppr = approximate_PageRank(graphlocal, [i], iterations=num_iter, method=method, normalize=False)
    d_inv = graphlocal.dn[node]
    d_inv[d_inv > 1.0] = 1.0
    ppr_d_inv = ppr * d_inv
    output = list(zip(node, ppr_d_inv))[:ego_size]
    node, ppr_d_inv = zip(*sorted(output, key=lambda x: x[1], reverse=True))
    assert node[0] == i
    node = np.array(node, dtype=np.int32)
    conds = my_sweep_cut(graphlocal, node)
    return node, conds


def step1_local_clustering(data, name, idx_split, ego_size=128, num_iter=1000, log_steps=10000, num_workers=16, method='acl', save_dir=None):
    if save_dir is None:
        save_path = f"{name}-lc-ego-graphs-{ego_size}.pt"
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{name}-lc-ego-graphs-{ego_size}.pt")

    N = data.num_nodes()
    edge_index = data.edges()
    edge_index = (edge_index[0].numpy(), edge_index[1].numpy())
    adj = csr_matrix((np.ones(edge_index[0].shape[0]), edge_index), shape=(N, N))

    global graphlocal
    graphlocal = GraphLocal.from_sparse_adjacency(adj)
    print('graphlocal generated')

    train_idx = idx_split["train"].cpu().numpy()
    valid_idx = idx_split["valid"].cpu().numpy()
    test_idx = idx_split["test"].cpu().numpy()

    with multiprocessing.Pool(num_workers) as pool:
        ego_graphs_train, conds_train = zip(*pool.imap(calc_local_clustering, [(i, log_steps, num_iter, ego_size, method) for i in train_idx], chunksize=512))

    with multiprocessing.Pool(num_workers) as pool:
        ego_graphs_valid, conds_valid = zip(*pool.imap(calc_local_clustering, [(i, log_steps, num_iter, ego_size, method) for i in valid_idx], chunksize=512))

    with multiprocessing.Pool(num_workers) as pool:
        ego_graphs_test, conds_test = zip(*pool.imap(calc_local_clustering, [(i, log_steps, num_iter, ego_size, method) for i in test_idx], chunksize=512))

    ego_graphs = []
    conds = []
    ego_graphs.extend(ego_graphs_train)
    ego_graphs.extend(ego_graphs_valid)
    ego_graphs.extend(ego_graphs_test)
    conds.extend(conds_train)
    conds.extend(conds_valid)
    conds.extend(conds_test)

    ego_graphs = [ego_graphs_train, ego_graphs_valid, ego_graphs_test]

    torch.save(ego_graphs, save_path)


def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    # src, dst = graph.all_edges()
    # graph.add_edges(dst, src)
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    # graph.create_formats_()
    return graph


def load_dataset(data_dir, dataset_name):
    if dataset_name.startswith("ogbn"):
        dataset = DglNodePropPredDataset(dataset_name, root=os.path.join(data_dir, "dataset"))
        graph, label = dataset[0]

        if "year" in graph.ndata:
            del graph.ndata["year"]
        if not graph.is_multigraph:
            graph = preprocess(graph)
        # graph = graph.remove_self_loop().add_self_loop()

        split_idx = dataset.get_idx_split()
        label = label.view(-1)
            
    elif dataset_name == "mag_scholar_f":
        edge_index = np.load(os.path.join(data_dir, dataset_name, "edge_index_f.npy"))
        print(len(edge_index[0]))
        graph = dgl.DGLGraph((edge_index[0], edge_index[1]))
        print(graph)
        num_nodes = graph.num_nodes()
        assert num_nodes == 12403930
        split_idx = torch.load(os.path.join(data_dir, dataset_name, "split_idx_f.pt"))
    else:
        raise NotImplementedError

    return graph, split_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LCGNN (Preprocessing)')
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="lc_ego_graphs")
    parser.add_argument('--ego_size', type=int, default=256)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--log_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default='acl')
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)

    graph, split_idx = load_dataset(args.data_dir, args.dataset)
    step1_local_clustering(graph, args.dataset, split_idx, args.ego_size, args.num_iter, args.log_steps, args.num_workers, args.method, args.save_dir)
