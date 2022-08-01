import os
import torch

from ogb.nodeproppred import NodePropPredDataset
from ogb.nodeproppred import Evaluator as NodeEvaluator
from ogb.graphproppred import GraphPropPredDataset
from ogb.linkproppred import LinkPropPredDataset

from cogdl.data import Dataset, Graph, DataLoader
from cogdl.utils import CrossEntropyLoss, Accuracy, remove_self_loops, coalesce, BCEWithLogitsLoss


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
        return CrossEntropyLoss()

    def get_evaluator(self):
        return Accuracy()

    def _download(self):
        pass

    @property
    def processed_file_names(self):
        return "data_cogdl.pt"

    def process(self):
        name = self.name.replace("_", "-")
        dataset = NodePropPredDataset(name, self.root)
        graph, y = dataset[0]
        x = torch.tensor(graph["node_feat"]).contiguous() if graph["node_feat"] is not None else None
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


class OGBArxivDataset(OGBNDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbn-arxiv"
        super(OGBArxivDataset, self).__init__(data_path, dataset)


class OGBProductsDataset(OGBNDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbn-products"
        super(OGBProductsDataset, self).__init__(data_path, dataset)


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
        return BCEWithLogitsLoss()

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


class OGBPapers100MDataset(OGBNDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbn-papers100M"
        super(OGBPapers100MDataset, self).__init__(data_path, dataset)


class OGBGDataset(Dataset):
    def __init__(self, root, name):
        super(OGBGDataset, self).__init__(root)
        self.name = name
        self.dataset = GraphPropPredDataset(self.name, root)

        self.data = []
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
            self.data.append(data)

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
            datalist.append(self.data[idx])
        return datalist

    def get(self, idx):
        return self.data[idx]

    def _download(self):
        pass

    def _process(self):
        pass

    @property
    def num_classes(self):
        return int(self.dataset.num_classes)


class OGBMolbaceDataset(OGBGDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-molbace"
        super(OGBMolbaceDataset, self).__init__(data_path, dataset)


class OGBMolhivDataset(OGBGDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-molhiv"
        super(OGBMolhivDataset, self).__init__(data_path, dataset)


class OGBMolpcbaDataset(OGBGDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-molpcba"
        super(OGBMolpcbaDataset, self).__init__(data_path, dataset)


class OGBPpaDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-ppa"
        path = "data"
        super(OGBPpaDataset, self).__init__(path, dataset)


class OGBCodeDataset(OGBGDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-code"
        super(OGBCodeDataset, self).__init__(data_path, dataset)


#This part is for ogbl datasets

class OGBLDataset(Dataset):
    def __init__(self, root, name):
        """
           - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
        """        
        
        self.name = name
        
        dataset = LinkPropPredDataset(name, root)
        graph= dataset[0]
        x = torch.tensor(graph["node_feat"]).contiguous() if graph["node_feat"] is not None else None
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

        self.data = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr, y=None)
        self.data.num_nodes = graph["num_nodes"]
        
    def get(self, idx):
        assert idx == 0
        return self.data

    def get_loss_fn(self):
        return CrossEntropyLoss()

    def get_evaluator(self):
        return Accuracy()

    def _download(self):
        pass
    
    @property
    def processed_file_names(self):
        return "data_cogdl.pt"
    
    def _process(self):
        pass
    
    def get_edge_split(self):
        idx = self.dataset.get_edge_split()
        train_edge = torch.from_numpy(idx['train']['edge'].T)
        val_edge = torch.from_numpy(idx['valid']['edge'].T)
        test_edge = torch.from_numpy(idx['test']['edge'].T)
        return train_edge, val_edge, test_edge

class OGBLPpaDataset(OGBLDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbl-ppa"
        super(OGBLPpaDataset, self).__init__(data_path, dataset)
        
        
class OGBLCollabDataset(OGBLDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbl-collab"
        super(OGBLCollabDataset, self).__init__(data_path, dataset)
        

class OGBLDdiDataset(OGBLDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbl-ddi"
        super(OGBLDdiDataset, self).__init__(data_path, dataset)

        
class OGBLCitation2Dataset(OGBLDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbl-citation2"
        super(OGBLCitation2Dataset, self).__init__(data_path, dataset)
                           

