import os
import numpy as np
import torch

from ogb.nodeproppred import NodePropPredDataset
from ogb.nodeproppred import Evaluator as NodeEvaluator
from ogb.graphproppred import GraphPropPredDataset

from . import register_dataset
from cogdl.data import Dataset, Graph, DataLoader
from cogdl.utils import cross_entropy_loss, accuracy, remove_self_loops


def coalesce(row, col, edge_attr=None):
    row = torch.tensor(row)
    col = torch.tensor(col)
    if edge_attr is not None:
        edge_attr = torch.tensor(edge_attr)
    num = col.shape[0] + 1
    idx = torch.full((num,), -1, dtype=torch.float)
    idx[1:] = row * num + col
    mask = idx[1:] > idx[:-1]

    if mask.all():
        return row, col, edge_attr
    row = row[mask]
    col = col[mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    return row, col, edge_attr


class OGBNDataset(Dataset):
    def __init__(self, root, name):
        super(OGBNDataset, self).__init__(root)
        dataset = NodePropPredDataset(name, root)
        graph, y = dataset[0]
        x = torch.tensor(graph["node_feat"])
        y = torch.tensor(y.squeeze())
        row, col, edge_attr = coalesce(graph["edge_index"][0], graph["edge_index"][1], graph["edge_feat"])
        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        self.data = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        self.data.num_nodes = graph["num_nodes"]
        assert self.data.num_nodes == self.data.x.shape[0]

        # split
        split_index = dataset.get_idx_split()
        self.data.train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.test_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.train_mask[split_index["train"]] = True
        self.data.test_mask[split_index["test"]] = True
        self.data.val_mask[split_index["valid"]] = True

        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def get_loss_fn(self):
        return cross_entropy_loss

    def get_evaluator(self):
        return accuracy

    def _download(self):
        pass

    def _process(self):
        pass


@register_dataset("ogbn-arxiv")
class OGBArxivDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-arxiv"
        path = "data"
        super(OGBArxivDataset, self).__init__(path, dataset)

    def get_evaluator(self):
        evaluator = NodeEvaluator(name="ogbn-arxiv")

        def wrap(y_pred, y_true):
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            return evaluator.eval(input_dict)

        return wrap


@register_dataset("ogbn-products")
class OGBProductsDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-products"
        path = "data"
        super(OGBProductsDataset, self).__init__(path, dataset)


@register_dataset("ogbn-proteins")
class OGBProteinsDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-proteins"
        path = "data"
        super(OGBProteinsDataset, self).__init__(path, dataset)


@register_dataset("ogbn-papers100M")
class OGBPapers100MDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-papers100M"
        path = "data"
        super(OGBPapers100MDataset, self).__init__(path, dataset)


@register_dataset("ogbn-mag")
class MAGDataset(Dataset):
    def __init__(self):
        self.name = "ogbn-mag"
        name = "_".join(self.name.split("-"))
        self.root = "./data/" + name
        super(MAGDataset, self).__init__(self.root)
        data = torch.load(self.processed_paths[0])
        (self.data, self.node_type_dict, self.edge_type_dict, self.num_nodes_dict) = data
        self.paper_feat = torch.as_tensor(np.load(self.processed_paths[1]))
        self.other_feat = torch.as_tensor(np.load(self.processed_paths[2]))
        if self.other_feat.shape[0] == self.data.num_nodes:
            self.other_feat = self.other_feat[self.paper_feat.shape[0] :]

    def __len__(self):
        return 1

    def get(self, idx):
        assert idx == 0
        return self.data

    def _download(self):
        pass

    def process(self):
        dataset = NodePropPredDataset(name=self.name, root="./data")
        node_type_dict = {"paper": 0, "author": 1, "field_of_study": 2, "institution": 3}
        edge_type_dict = {"cites": 0, "affiliated_with": 1, "writes": 2, "has_topic": 3}
        num_nodes_dict = dataset[0][0]["num_nodes_dict"]
        num_nodes = torch.as_tensor(
            [0]
            + [
                num_nodes_dict["paper"],
                num_nodes_dict["author"],
                num_nodes_dict["field_of_study"],
                num_nodes_dict["institution"],
            ]
        )
        cum_num_nodes = torch.cumsum(num_nodes, dim=-1)
        node_types = torch.repeat_interleave(torch.arange(0, 4), num_nodes[1:])

        edge_index_dict = dataset[0][0]["edge_index_dict"]

        edge_index = [None] * len(edge_type_dict)
        edge_attr = [None] * len(edge_type_dict)

        i = 0
        for k, v in edge_index_dict.items():
            head, edge_type, tail = k
            head_offset = cum_num_nodes[node_type_dict[head]].item()
            tail_offset = cum_num_nodes[node_type_dict[tail]].item()
            src = v[0] + head_offset
            tgt = v[1] + tail_offset
            edge_tps = np.full(src.shape, edge_type_dict[edge_type])

            _src = np.concatenate([src, tgt])
            _tgt = np.concatenate([tgt, src])
            if edge_type == "cites":
                re_tps = np.full(src.shape, edge_type_dict[edge_type])
            else:
                re_tps = np.full(src.shape, len(edge_type_dict))
                edge_type_dict[edge_type + "_re"] = len(edge_type_dict)
            edge_index[i] = np.vstack([_src, _tgt])
            edge_tps = np.concatenate([edge_tps, re_tps])
            edge_attr[i] = edge_tps
            i += 1
        edge_index = np.concatenate(edge_index, axis=-1)
        edge_index = torch.from_numpy(edge_index)
        edge_attr = torch.from_numpy(np.concatenate(edge_attr))

        assert edge_index.shape[1] == edge_attr.shape[0]

        split_index = dataset.get_idx_split()
        train_index = torch.from_numpy(split_index["train"]["paper"])
        val_index = torch.from_numpy(split_index["valid"]["paper"])
        test_index = torch.from_numpy(split_index["test"]["paper"])
        y = torch.as_tensor(dataset[0][1]["paper"]).view(-1)

        paper_feat = dataset[0][0]["node_feat_dict"]["paper"]
        data = Graph(
            y=y,
            edge_index=edge_index,
            edge_types=edge_attr,
            train_mask=train_index,
            val_mask=val_index,
            test_mask=test_index,
            node_types=node_types,
        )
        # self.save_edges(data)
        torch.save((data, node_type_dict, edge_type_dict, num_nodes_dict), self.processed_paths[0])
        np.save(self.processed_paths[1], paper_feat)

    def get_evaluator(self):
        evaluator = NodeEvaluator(name="ogbn-mag")

        def wrap(y_pred, y_true):
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            return evaluator.eval(input_dict)

        return wrap

    @property
    def processed_file_names(self):
        return ["data.pt", "paper_feat.npy", "other_feat.npy"]

    @property
    def num_node_types(self):
        return len(self.node_type_dict)

    @property
    def num_edge_types(self):
        return len(self.edge_type_dict)

    @property
    def num_papers(self):
        return self.num_nodes_dict["paper"]

    @property
    def num_authors(self):
        return self.num_nodes_dict["author"]

    @property
    def num_institutions(self):
        return self.num_nodes_dict["institution"]

    @property
    def num_field_of_study(self):
        return self.num_nodes_dict["field_of_study"]

    def save_edges(self, data):
        edge_index = data.edge_index.numpy().transpose()
        edge_types = data.edge_types.numpy()
        os.makedirs("./ogbn_mag_kg", exist_ok=True)
        with open("./ogbn_mag_kg/train.txt", "w") as f:
            for i in range(edge_index.shape[0]):
                edge = edge_index[i]
                tp = edge_types[i]
                f.write(f"{edge[0]}\t{edge[1]}\t{tp}\n")

        with open("./ogbn_mag_kg/valid.txt", "w") as f:
            val_num = np.random.randint(0, edge_index.shape[0], (10000,))
            for i in val_num:
                edge = edge_index[i]
                tp = edge_types[i]
                f.write(f"{edge[0]}\t{edge[1]}\t{tp}\n")

        with open("./ogbn_mag_kg/test.txt", "w") as f:
            val_num = np.random.randint(0, edge_index.shape[0], (20000,))
            for i in val_num:
                edge = edge_index[i]
                tp = edge_types[i]
                f.write(f"{edge[0]}\t{edge[1]}\t{tp}\n")

        with open("./ogbn_mag_kg/entities.dict", "w") as f:
            for i in range(np.max(edge_index)):
                f.write(f"{i}\t{i}\n")
        with open("./ogbn_mag_kg/relations.dict", "w") as f:
            for i in range(np.max(edge_types)):
                f.write(f"{i}\t{i}\n")


class OGBGDataset(Dataset):
    def __init__(self, root, name):
        super(OGBGDataset, self).__init__(root)
        self.name = name
        self.dataset = GraphPropPredDataset(self.name, root)

        self.graphs = []
        self.all_nodes = 0
        self.all_edges = 0
        for i in range(len(self.dataset.graphs)):
            graph, label = self.dataset[i]
            data = Graph(
                x=torch.tensor(graph["node_feat"], dtype=torch.float),
                edge_index=torch.tensor(graph["edge_index"]),
                edge_attr=None if "edge_feat" not in graph else torch.tensor(graph["edge_feat"], dtype=torch.float),
                y=torch.tensor(label),
            )
            data.num_nodes = graph["num_nodes"]
            self.graphs.append(data)

            self.all_nodes += graph["num_nodes"]
            self.all_edges += graph["edge_index"].shape[1]

        self.transform = None

    def get_loader(self, args):
        split_index = self.dataset.get_idx_split()
        train_loader = DataLoader(self.get_subset(split_index["train"]), batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(self.get_subset(split_index["valid"]), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(self.get_subset(split_index["test"]), batch_size=args.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader

    def get_subset(self, subset):
        datalist = []
        for idx in subset:
            datalist.append(self.graphs[idx])
        return datalist

    def get(self, idx):
        return self.graphs[idx]

    def _download(self):
        pass

    def _process(self):
        pass

    @property
    def num_classes(self):
        return int(self.dataset.num_classes)


@register_dataset("ogbg-molbace")
class OGBMolbaceDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-molbace"
        path = "data"
        super(OGBMolbaceDataset, self).__init__(path, dataset)


@register_dataset("ogbg-molhiv")
class OGBMolhivDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-molhiv"
        path = "data"
        super(OGBMolhivDataset, self).__init__(path, dataset)


@register_dataset("ogbg-molpcba")
class OGBMolpcbaDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-molpcba"
        path = "data"
        super(OGBMolpcbaDataset, self).__init__(path, dataset)


@register_dataset("ogbg-ppa")
class OGBPpaDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-ppa"
        path = "data"
        super(OGBPpaDataset, self).__init__(path, dataset)


@register_dataset("ogbg-code")
class OGBCodeDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-code"
        path = "data"
        super(OGBCodeDataset, self).__init__(path, dataset)
