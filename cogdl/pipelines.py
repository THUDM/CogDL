import os
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.lib.arraysetops import isin
import torch
import yaml
from grave import plot_network, use_attributes
from tabulate import tabulate

from cogdl import oagbert
from cogdl.data import Graph
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset_from_name, NodeDataset
from cogdl.models import build_model
from cogdl.options import get_default_args
from cogdl.datasets.rec_data import build_recommendation_data


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
                    data.edge_index[0].shape[0],
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
        edge_index = torch.stack(data.edge_index)
        G.add_edges_from([tuple(edge_index[:, i].numpy()) for i in range(edge_index.shape[1])])

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


class GenerateEmbeddingPipeline(Pipeline):
    def __init__(self, app: str, model: str, **kwargs):
        super(GenerateEmbeddingPipeline, self).__init__(app, model=model, **kwargs)

        match_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "match.yml")
        with open(match_path, "r", encoding="utf8") as f:
            match = yaml.load(f, Loader=yaml.FullLoader)
        objective = match.get("unsupervised_node_classification", None)
        for pair_dict in objective:
            if "blogcatalog" in pair_dict["dataset"]:
                emb_models = pair_dict["model"]
            elif "cora" in pair_dict["dataset"]:
                gnn_models = pair_dict["model"]

        if model in emb_models:
            self.method_type = "emb"
            args = get_default_args(
                task="unsupervised_node_classification", dataset="blogcatalog", model=model, **kwargs
            )
        elif model in gnn_models:
            self.method_type = "gnn"
            args = get_default_args(task="unsupervised_node_classification", dataset="cora", model=model, **kwargs)
        else:
            print("Please choose a model from ", emb_models, "or", gnn_models)
            exit(0)

        self.data_path = kwargs.get("data_path", "tmp_data.pt")
        self.num_features = kwargs.get("num_features", None)
        if self.num_features is not None:
            args.num_features = self.num_features
        elif self.method_type == "gnn":
            print("Please provide num_features for gnn model!")
            exit(0)

        args.model = args.model[0]
        self.model = build_model(args)

        self.trainer = self.model.get_trainer(args)
        if self.trainer is not None:
            self.trainer = self.trainer(args)

    def __call__(self, edge_index, x=None, edge_weight=None):
        if self.method_type == "emb":
            G = nx.Graph()
            if edge_weight is not None:
                if isinstance(edge_index, np.ndarray):
                    edges = np.concatenate([edge_index, np.expand_dims(edge_weight, -1)], -1)
                elif isinstance(edge_index, torch.Tensor):
                    edges = torch.cat([edge_index, edge_weight.unsqueeze(-1)], -1)
                else:
                    print("Please provide edges via np.ndarray or torch.Tensor.")
                    return
                G.add_weighted_edges_from(edges.tolist())
            else:
                if not isinstance(edge_index, np.ndarray) and not isinstance(edge_index, torch.Tensor):
                    print("Please provide edges via np.ndarray or torch.Tensor.")
                    return
                G.add_edges_from(edge_index.tolist())

            embeddings = self.model.train(G)
        elif self.method_type == "gnn":
            num_nodes = edge_index.max().item() + 1
            if x is None:
                print("No input node features, use random features instead.")
                np.random.randn(num_nodes, self.num_features)
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if isinstance(edge_index, np.ndarray):
                edge_index = torch.from_numpy(edge_index)
            edge_index = (edge_index[:, 0], edge_index[:, 1])
            data = Graph(x=x, edge_index=edge_index)
            torch.save(data, self.data_path)
            dataset = NodeDataset(path=self.data_path, scale_feat=False)
            embeddings = self.trainer.fit(self.model, dataset, evaluate=False)
            embeddings = embeddings.detach().cpu().numpy()

        return embeddings


class RecommendationPipepline(Pipeline):
    def __init__(self, app: str, model: str, **kwargs):
        super(RecommendationPipepline, self).__init__(app, model=model, **kwargs)

        if "data" in kwargs:
            data = kwargs["data"]
            val_data = test_data = data[-100:, :]
            data = build_recommendation_data("custom", data, val_data, test_data)
            self.data_path = kwargs.get("data_path", "tmp_data.pt")
            self.batch_size = kwargs.get("batch_size", 128)
            torch.save(data, self.data_path)
            self.dataset = NodeDataset(path=self.data_path, scale_feat=False)
        elif "dataset" in kwargs:
            dataset = kwargs.pop("dataset")
            self.dataset = build_dataset_from_name(dataset)
        else:
            print("Please provide recommendation data!")
            exit(0)

        self.batch_size = kwargs.get("batch_size", 2048)
        self.n_items = self.dataset[0].n_params["n_items"]

        args = get_default_args(task="recommendation", dataset="ali", model=model, **kwargs)
        args.model = args.model[0]

        task = build_task(args, dataset=self.dataset)
        task.train()

        self.model = task.model
        self.model.eval()

        self.user_emb, self.item_emb = self.model.generate()

    def __call__(self, user_batch, **kwargs):
        user_batch = np.array(user_batch)
        user_batch = torch.from_numpy(user_batch).to(self.model.device)
        u_g_embeddings = self.user_emb[user_batch]

        # batch-item test
        n_item_batchs = self.n_items // self.batch_size + 1
        rate_batch = np.zeros(shape=(len(user_batch), self.n_items))

        i_count = 0
        for i_batch_id in range(n_item_batchs):
            i_start = i_batch_id * self.batch_size
            i_end = min((i_batch_id + 1) * self.batch_size, self.n_items)

            item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(self.model.device)
            i_g_embddings = self.item_emb[item_batch]

            i_rate_batch = self.model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

            rate_batch[:, i_start:i_end] = i_rate_batch
            i_count += i_rate_batch.shape[1]

        topk = kwargs.get("topk", 10)
        results = {}
        for i in range(len(user_batch)):
            rate = list(zip(range(self.n_items), rate_batch[i]))
            rate.sort(key=lambda x: x[1], reverse=True)
            results[user_batch[i].item()] = [rate[j] for j in range(min(topk, len(rate)))]

        return results


SUPPORTED_APPS = {
    "dataset-stats": {"impl": DatasetStatsPipeline, "default": {"dataset": "cora"}},
    "dataset-visual": {"impl": DatasetVisualPipeline, "default": {"dataset": "cora"}},
    "oagbert": {"impl": OAGBertInferencePipepline, "default": {"model": "oagbert-v1"}},
    "generate-emb": {"impl": GenerateEmbeddingPipeline, "default": {"model": "prone"}},
    "recommendation": {"impl": RecommendationPipepline, "default": {"model": "lightgcn"}},
}


def check_app(app: str):
    if app in SUPPORTED_APPS:
        targeted_app = SUPPORTED_APPS[app]
        return targeted_app

    raise KeyError("Unknown app {}, available apps are {}".format(app, list(SUPPORTED_APPS.keys())))


def pipeline(app: str, **kwargs) -> Pipeline:
    targeted_app = check_app(app)
    task_class = targeted_app["impl"]
    default_args = targeted_app["default"].copy()
    default_args.update(kwargs)

    return task_class(app=app, **default_args)
