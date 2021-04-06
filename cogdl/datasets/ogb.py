import torch

from ogb.nodeproppred import NodePropPredDataset
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


@register_dataset("ogbn-mag")
class OGBMAGDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-mag"
        path = "data"
        super(OGBMAGDataset, self).__init__(path, dataset)


@register_dataset("ogbn-papers100M")
class OGBPapers100MDataset(OGBNDataset):
    def __init__(self):
        dataset = "ogbn-papers100M"
        path = "data"
        super(OGBPapers100MDataset, self).__init__(path, dataset)


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
