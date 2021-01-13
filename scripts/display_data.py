import os.path as osp
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from cogdl import options
from cogdl.datasets import build_dataset_from_name
from grave import plot_network, use_attributes
from tabulate import tabulate


def plot_graph(args):
    if not isinstance(args.dataset, list):
        args.dataset = [args.dataset]

    for name in args.dataset:
        dataset = build_dataset_from_name(name)
        data = dataset[0]

        depth = args.depth
        pic_file = osp.join(args.save_dir, f"display_{name}.png")

        col_names = [
            "Dataset",
            "#nodes",
            "#edges",
            "#features",
            "#classes",
            "#labeled data",
        ]
        tab_data = [
            [
                name,
                data.x.shape[0],
                data.edge_index.shape[1],
                data.x.shape[1],
                len(set(data.y.numpy())),
                sum(data.train_mask.numpy()),
            ]
        ]
        print(tabulate(tab_data, headers=col_names, tablefmt="psql"))

        G = nx.Graph()
        G.add_edges_from([tuple(data.edge_index[:, i].numpy()) for i in range(data.edge_index.shape[1])])

        s = random.choice(list(G.nodes()))
        q = [s]
        node_set = set([s])
        node_index = {s: 0}
        max_index = 1
        for _ in range(depth):
            nq = []
            for x in q:
                for key in G[x].keys():
                    if key not in node_set:
                        nq.append(key)
                        node_set.add(key)
                        node_index[key] = node_index[x] + 1
            if len(nq) > 0:
                max_index += 1
            q = nq

        cmap = cm.rainbow(np.linspace(0.0, 1.0, max_index))

        for node, index in node_index.items():
            G.nodes[node]["color"] = cmap[index]
            G.nodes[node]["size"] = (max_index - index) * 50

        fig, ax = plt.subplots()
        plot_network(G.subgraph(list(node_set)), node_style=use_attributes())
        plt.savefig(pic_file)
        print(f"Sampled ego network saved to {pic_file} .")


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0, help='random seed')
    parser.add_argument('--depth', '-d', type=int, default=3, help='neighborhood depth')
    parser.add_argument('--name', '-n', type=str, default='Cora', help='dataset name')
    parser.add_argument('--file', '-f', type=str, default='graph.jpg', help='saved file name')
    args = parser.parse_args()
    """
    parser = options.get_display_data_parser()
    args = parser.parse_args()

    if isinstance(args.seed, list):
        args.seed = args.seed[0]

    random.seed(args.seed)
    np.random.seed(args.seed)

    plot_graph(args)
