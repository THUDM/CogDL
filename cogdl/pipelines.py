import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from grave import plot_network, use_attributes
from tabulate import tabulate

from cogdl import oagbert
from cogdl.datasets import build_dataset_from_name


class Pipeline(object):
    def __init__(self, app: str, **kwargs):
        self.app = app
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        raise NotImplementedError


class DatasetPipeline(Pipeline):
    def __init__(self, app: str, **kwargs):
        super(DatasetPipeline, self).__init__(app, **kwargs)

    def __call__(self, dataset, **kwargs):
        if isinstance(dataset, str):
            dataset = [dataset]

        return self._call(dataset, **kwargs)


class DatasetStatsPipeline(DatasetPipeline):
    def __init__(self, app: str, **kwargs):
        super(DatasetStatsPipeline, self).__init__(app, **kwargs)

    def _call(self, dataset=[], **kwargs):
        if isinstance(dataset, str):
            dataset = [dataset]
        tab_data = []
        col_names = [
            "Dataset",
            "#nodes",
            "#edges",
            "#features",
            "#classes",
            "#labeled data",
        ]
        for name in dataset:
            dataset = build_dataset_from_name(name)
            data = dataset[0]

            tab_data.append(
                [
                    name,
                    data.x.shape[0],
                    data.edge_index.shape[1],
                    data.x.shape[1],
                    len(set(data.y.numpy())),
                    sum(data.train_mask.numpy()),
                ]
            )
        print(tabulate(tab_data, headers=col_names, tablefmt="psql"))

        return tab_data


class DatasetVisualPipeline(DatasetPipeline):
    def __init__(self, app: str, **kwargs):
        super(DatasetVisualPipeline, self).__init__(app, **kwargs)

    def _call(self, dataset="cora", seed=-1, depth=3, **kwargs):
        if isinstance(dataset, list):
            dataset = dataset[0]
        name = dataset
        dataset = build_dataset_from_name(name)
        data = dataset[0]

        G = nx.Graph()
        G.add_edges_from([tuple(data.edge_index[:, i].numpy()) for i in range(data.edge_index.shape[1])])

        if seed == -1:
            seed = random.choice(list(G.nodes()))
        q = [seed]
        node_set = set([seed])
        node_index = {seed: 0}
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

        pic_file = f"{name}.png"
        plt.subplots()
        plot_network(G.subgraph(list(node_set)), node_style=use_attributes())
        plt.savefig(pic_file)
        print(f"Sampled ego network saved to {pic_file}")

        return q


class OAGBertInferencePipepline(Pipeline):
    def __init__(self, app: str, model: str, **kwargs):
        super(OAGBertInferencePipepline, self).__init__(app, model=model, **kwargs)

        load_weights = kwargs["load_weights"] if "load_weights" in kwargs else True
        self.tokenizer, self.bert_model = oagbert(model, load_weights=load_weights)

    def __call__(self, sequence, **kwargs):
        tokens = self.tokenizer(sequence, return_tensors="pt", padding=True)
        outputs = self.bert_model(**tokens)

        return outputs


SUPPORTED_APPS = {
    "dataset-stats": {"impl": DatasetStatsPipeline, "default": {"dataset": "cora"}},
    "dataset-visual": {"impl": DatasetVisualPipeline, "default": {"dataset": "cora"}},
    "oagbert": {"impl": OAGBertInferencePipepline, "default": {"model": "oagbert-v1"}},
}


def check_app(app: str):
    if app in SUPPORTED_APPS:
        targeted_app = SUPPORTED_APPS[app]
        return targeted_app

    raise KeyError("Unknown app {}, available apps are {}".format(app, list(SUPPORTED_APPS.keys())))


def pipeline(app: str, **kwargs) -> Pipeline:
    targeted_app = check_app(app)
    task_class = targeted_app["impl"]
    default_args = targeted_app["default"]
    default_args.update(kwargs)

    return task_class(app=app, **default_args)
