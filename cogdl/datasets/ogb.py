import os
import numpy as np
import torch

from ogb.nodeproppred import NodePropPredDataset
from ogb.nodeproppred import Evaluator as NodeEvaluator
from ogb.graphproppred import GraphPropPredDataset

from . import register_dataset
from cogdl.data import Dataset, Graph, DataLoader
from cogdl.utils import cross_entropy_loss, accuracy, remove_self_loops, coalesce, bce_with_logits_loss
from torch_geometric.utils import to_undirected


class OGBNDataset(Dataset):
    def __init__(self, root, name, transform=None):
        name = name.replace("-", "_")
        self.name = name
        root = os.path.join(root, name)
        super(OGBNDataset, self).__init__(root)
        self.transform = None
        self.data = torch.load(self.processed_paths[0])

    def get(self, idx):
        assert idx == 0
        return self.data

    def get_loss_fn(self):
        return cross_entropy_loss

    def get_evaluator(self):
        return accuracy

    def _download(self):
        pass

    @property
    def processed_file_names(self):
        return "data_cogdl.pt"

    def process(self):
        name = self.name.replace("_", "-")
        dataset = NodePropPredDataset(name, self.root)
        graph, y = dataset[0]
        x = torch.tensor(graph["node_feat"]) if graph["node_feat"] is not None else None
        y = torch.tensor(y.squeeze())
        row, col = graph["edge_index"][0], graph["edge_index"][1]
        row = torch.from_numpy(row)
        col = torch.from_numpy(col)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = torch.as_tensor(graph["edge_feat"]) if graph["edge_feat"] is not None else graph["edge_feat"]
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])

        row, col, _ = coalesce(row, col)
        edge_index = torch.stack([row, col], dim=0)

        data = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.num_nodes = graph["num_nodes"]

        # split
        split_index = dataset.get_idx_split()
        data.train_mask = torch.full((data.num_nodes,), False, dtype=torch.bool)
        data.val_mask = torch.full((data.num_nodes,), False, dtype=torch.bool)
        data.test_mask = torch.full((data.num_nodes,), False, dtype=torch.bool)

        data.train_mask[split_index["train"]] = True
        data.test_mask[split_index["test"]] = True
        data.val_mask[split_index["valid"]] = True

        torch.save(data, self.processed_paths[0])
        return data


@register_dataset("ogbn-arxiv")
class OGBArxivDataset(OGBNDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbn-arxiv"
        super(OGBArxivDataset, self).__init__(data_path, dataset)

    def get_evaluator(self):
        evaluator = NodeEvaluator(name="ogbn-arxiv")

        def wrap(y_pred, y_true):
            y_pred = y_pred.argmax(dim=-1, keepdim=True)
            y_true = y_true.view(-1, 1)
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            return evaluator.eval(input_dict)["acc"]

        return wrap


@register_dataset("ogbn-products")
class OGBProductsDataset(OGBNDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbn-products"
        super(OGBProductsDataset, self).__init__(data_path, dataset)


@register_dataset("ogbn-proteins")
class OGBProteinsDataset(OGBNDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbn-proteins"
        super(OGBProteinsDataset, self).__init__(data_path, dataset)

    @property
    def edge_attr_size(self):
        return [
            self.data.edge_attr.shape[1],
        ]

    def get_loss_fn(self):
        return bce_with_logits_loss

    def get_evaluator(self):
        evaluator = NodeEvaluator(name="ogbn-proteins")

        def wrap(y_pred, y_true):
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            return evaluator.eval(input_dict)["rocauc"]

        return wrap

    def process(self):
        name = self.name.replace("_", "-")
        dataset = NodePropPredDataset(name, self.root)
        graph, y = dataset[0]
        y = torch.tensor(y.squeeze())
        row, col = graph["edge_index"][0], graph["edge_index"][1]
        row = torch.from_numpy(row)
        col = torch.from_numpy(col)
        edge_attr = torch.as_tensor(graph["edge_feat"]) if "edge_feat" in graph else None

        data = Graph(x=None, edge_index=(row, col), edge_attr=edge_attr, y=y)
        data.num_nodes = graph["num_nodes"]

        # split
        split_index = dataset.get_idx_split()
        data.train_mask = torch.full((data.num_nodes,), False, dtype=torch.bool)
        data.val_mask = torch.full((data.num_nodes,), False, dtype=torch.bool)
        data.test_mask = torch.full((data.num_nodes,), False, dtype=torch.bool)

        data.train_mask[split_index["train"]] = True
        data.test_mask[split_index["test"]] = True
        data.val_mask[split_index["valid"]] = True

        edge_attr = data.edge_attr
        deg = data.degrees()
        dst, _ = data.edge_index
        dst = dst.view(-1, 1).expand(dst.shape[0], edge_attr.shape[1])
        x = torch.zeros((data.num_nodes, edge_attr.shape[1]), dtype=torch.float32)
        x = x.scatter_add_(dim=0, index=dst, src=edge_attr)
        deg = torch.clamp(deg, min=1)
        x = x / deg.view(-1, 1)
        data.x = x

        node_species = torch.as_tensor(graph["node_species"])
        n_species, new_index = torch.unique(node_species, return_inverse=True)
        one_hot_x = torch.nn.functional.one_hot(new_index, num_classes=torch.max(new_index).int().item())
        data.species = node_species
        data.x = torch.cat([data.x, one_hot_x], dim=1)
        torch.save(data, self.processed_paths[0])
        return data


@register_dataset("ogbn-papers100M")
class OGBPapers100MDataset(OGBNDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbn-papers100M"
        super(OGBPapers100MDataset, self).__init__(data_path, dataset)


@register_dataset("ogbn-mag")
class MAGDataset(Dataset):
    def __init__(self, data_path="data"):
        self.name = "ogbn-mag"
        name = "_".join(self.name.split("-"))
        self.root = f"./{data_path}/{name}"
        super(MAGDataset, self).__init__(self.root)
        data = torch.load(self.processed_paths[0])
        (self.data, self.node_type_dict, self.edge_type_dict, self.num_nodes_dict) = data
        self.paper_feat = torch.as_tensor(np.load(self.processed_paths[1]))

        self.other_feat = torch.as_tensor(np.load(self.processed_paths[2]))
        if self.other_feat.shape[0] == self.data.num_nodes:
            self.other_feat = self.other_feat[self.paper_feat.shape[0] :]

        # -- plus --
        # dataset = NodePropPredDataset(name=self.name, root="./data")
        # edge_index_dict = dataset[0][0]["edge_index_dict"]
        #
        # r, c = edge_index_dict[("author", "affiliated_with", "institution")]
        # edge_index_dict[("institution", "to", "author")] = torch.as_tensor([c, r], dtype=torch.long)
        #
        # r, c = edge_index_dict[("author", "writes", "paper")]
        # edge_index_dict[("paper", "to", "author")] = torch.as_tensor([c, r], dtype=torch.long)
        #
        # r, c = edge_index_dict[("paper", "has_topic", "field_of_study")]
        # edge_index_dict[("field_of_study", "to", "paper")] = torch.as_tensor([c, r], dtype=torch.long)
        #
        # edge_index = to_undirected(torch.as_tensor(edge_index_dict[("paper", "cites", "paper")], dtype=torch.long))
        # edge_index_dict[("paper", "cites", "paper")] = edge_index
        #
        # for k, v in edge_index_dict.items():
        #     edge_index_dict[k] = torch.as_tensor(v)
        # self.edge_index_dict = edge_index_dict

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
        edge_type_dict = {
            ("paper", "cites", "paper"): 0,
            ("author", "affiliated_with", "institution"): 1,
            ("author", "writes", "paper"): 2,
            ("paper", "has_topic", "field_of_study"): 3,
        }
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
            edge_tps = np.full(src.shape, edge_type_dict[k])

            if edge_type == "cites":
                _edges = torch.as_tensor([src, tgt])
                _src, _tgt = to_undirected(_edges).numpy()
                edge_tps = np.full(_src.shape, edge_type_dict[k])
                edge_idx = np.vstack([_src, _tgt])
            else:
                _src = np.concatenate([src, tgt])
                _tgt = np.concatenate([tgt, src])
                re_tps = np.full(src.shape, len(edge_type_dict))

                re_k = (tail, "to", head)
                edge_type_dict[re_k] = len(edge_type_dict)
                edge_tps = np.concatenate([edge_tps, re_tps])
                edge_idx = np.vstack([_src, _tgt])

            edge_index[i] = edge_idx
            edge_attr[i] = edge_tps
            assert edge_index[i].shape[1] == edge_attr[i].shape[0]
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
            y_pred = y_pred.argmax(dim=-1, keepdim=True)
            y_true = y_true.view(-1, 1)
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            return evaluator.eval(input_dict)["acc"]

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
    def __init__(self, data_path="data"):
        dataset = "ogbg-molbace"
        super(OGBMolbaceDataset, self).__init__(data_path, dataset)


@register_dataset("ogbg-molhiv")
class OGBMolhivDataset(OGBGDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-molhiv"
        super(OGBMolhivDataset, self).__init__(data_path, dataset)


@register_dataset("ogbg-molpcba")
class OGBMolpcbaDataset(OGBGDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-molpcba"
        super(OGBMolpcbaDataset, self).__init__(data_path, dataset)


@register_dataset("ogbg-ppa")
class OGBPpaDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-ppa"
        path = "data"
        super(OGBPpaDataset, self).__init__(path, dataset)


@register_dataset("ogbg-code")
class OGBCodeDataset(OGBGDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-code"
        super(OGBCodeDataset, self).__init__(data_path, dataset)
