import copy
import os
import torch
import numpy as np
from ogb.nodeproppred import NodePropPredDataset
from ogb.nodeproppred import Evaluator as NodeEvaluator
from ogb.graphproppred import GraphPropPredDataset
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.lsc import PCQM4MDataset, PCQM4Mv2Dataset, PCQM4MEvaluator, PCQM4Mv2Evaluator
from ogb.utils import smiles2graph

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


class MultiGraph:
    def __init__(self, x_l, y_l, edge_index_l, edge_attr_l, x_s, y_s, e_s, n_s):
        self._x = torch.cat(x_l, dim=0) if x_l else None
        self._y = torch.cat(y_l, dim=0) if y_l else None
        self._edge_index = torch.cat(edge_index_l, dim=0) if edge_index_l else None
        self._edge_attr = torch.cat(edge_attr_l, dim=0) if edge_attr_l else None
        self.x_s, self.y_s, self.e_s, self.n_s = x_s, y_s, e_s, n_s
        self._indices = None

    def __getitem__(self, n):
        n = self.indices()[n]
        x = self._x[self.x_s[n]:self.x_s[n+1]] if self._x is not None else None
        y = self._y[self.y_s[n]:self.y_s[n+1]] if self._y is not None else None
        edge_index = self._edge_index[self.e_s[n]:self.e_s[n+1]] if self._edge_index is not None else None
        edge_attr = self._edge_attr[self.e_s[n]:self.e_s[n+1]] if self._edge_attr is not None else None

        data = Graph(
            x=x,
            edge_index = edge_index.t(),
            edge_attr=edge_attr,
            y=y,
        )
        data.num_nodes = self.n_s[n]
        return data
    
    def __len__(self):
        return len(self._indices) if self._indices is not None else len(self.n_s)
    
    @property
    def edge_index(self):
        row, col = self._edge_index
        return (row, col)

    def indices(self):
        return range(len(self.n_s)) if self._indices is None else self._indices

class MultiGraphCode:
    def __init__(self, x_l, y_l, edge_index_l, edge_attr_l, x_s, y_s, e_s, n_s):
        self._x = torch.cat(x_l, dim=0) if x_l else None
        self._y = torch.cat(y_l, dim=0) if y_l else None
        self._edge_index = torch.cat(edge_index_l, dim=0) if edge_index_l else None
        self._edge_attr = torch.cat(edge_attr_l, dim=0) if edge_attr_l else None
        self.x_s, self.y_s, self.e_s, self.n_s = x_s, y_s, e_s, n_s
        self._indices = None

    def __getitem__(self, n):
        n = self.indices()[n]
        x = self._x[self.x_s[n]:self.x_s[n+1]] if self._x is not None else None
        y = self._y[self.y_s[n]:self.y_s[n+1]] if self._y is not None else None
        edge_index = self._edge_index[self.e_s[n]:self.e_s[n+1]] if self._edge_index is not None else None
        edge_attr = self._edge_attr[self.e_s[n]:self.e_s[n+1]] if self._edge_attr is not None else None

        num_nodes = x.shape[0]
        row, col = edge_index[:, 0], edge_index[:, 1]
        zero, one = torch.zeros_like(row), torch.ones_like(row)
        
        edge_index_ast = torch.stack([row, col], dim=0)
        edge_attr_ast = torch.stack([zero, zero], dim=0).t()
        
        edge_index_ast_inverse = torch.stack([col, row], dim=0)
        edge_attr_ast_inverse = torch.stack([zero, one], dim=0).t()

        node_is_attributed = x[:, 2].clone()
        node_dfs_order = x[:, 3].clone
        attributed_node_idx_in_dfs_order = torch.where(node_is_attributed.view(-1,) == 1)[0]

        edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]], dim = 0)
        edge_attr_nextoken = torch.cat([torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim = 1)

        edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim = 0)
        edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

        edge_index = torch.cat([edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim=1)
        edge_attr = torch.cat([edge_attr_ast, edge_attr_ast_inverse, edge_attr_nextoken,  edge_attr_nextoken_inverse], dim=0).to(torch.float32)

        # edge_index = torch.cat([edge_index_ast, edge_index_ast_inverse], dim=1)
        # edge_attr = torch.cat([edge_attr_ast, edge_attr_ast_inverse], dim=0).to(torch.float32)

        data = Graph(
            x=x,
            edge_index = edge_index,
            edge_attr=edge_attr,
            y=y,
        )
        data.num_nodes = self.n_s[n]

        return data
    
    def __len__(self):
        return len(self._indices) if self._indices is not None else len(self.n_s)
    
    @property
    def edge_index(self):
        row, col = self._edge_index
        return (row, col)
    
    def indices(self):
        return range(len(self.n_s)) if self._indices is None else self._indices

class OGBGDataset(Dataset):
    def __init__(self, root, name):
        name = name.replace("-", "_")
        root = os.path.join(root, name)
        self.root, self.name = root, name
        super(OGBGDataset, self).__init__(root)
        self.data, self.all_nodes, self.all_edges, self.transform, self.split_index = torch.load(self.processed_paths[0])
        self._indices = None
    
    def get_subset(self, subset):
        # datalist = []
        # for idx in subset:
        #     datalist.append(self.data[idx])
        # return datalist
        data = copy.copy(self.data)
        data._indices = subset
        return data

    def get(self, idx):
        return self.data[idx]

    def _download(self):
        pass

    def process(self):
        name = self.name.replace("_", "-")
        dataset = GraphPropPredDataset(name, self.root)

        if name == 'ogbg-molhiv':
            x_dtype, attr_type = torch.long, torch.long
        elif name == 'ogbg-molpcba':
            x_dtype, attr_type = torch.long, torch.long
        elif name == 'ogbg-ppa':
            x_dtype, attr_type = torch.long, torch.float32
        elif name == 'ogbg-code2':
            x_dtype, attr_type = torch.long, torch.float32

        all_nodes, all_edges = 0, 0
        
        x_l, y_l, edge_index_l, edge_attr_l = [], [], [], []
        x_s, y_s, e_s, n_s = [0], [0], [0], []
        
        label_l = []
        
        for i in range(len(dataset.graphs)):
            graph, label = dataset[i]

            if "node_feat" in graph and graph["node_feat"] is not None:
                x = torch.tensor(graph["node_feat"], dtype=x_dtype) 
            else:
                x = torch.zeros(graph["num_nodes"], dtype=x_dtype)
            
            if name == 'ogbg-code2':
                x = torch.cat([x, torch.tensor(graph["node_is_attributed"], dtype=x_dtype)], dim=1)
                x = torch.cat([x, torch.tensor(graph["node_dfs_order"], dtype=x_dtype)], dim=1)
                x = torch.cat([x, torch.tensor(graph["node_depth"], dtype=x_dtype)], dim=1)
                label_l.append(label)
                y = torch.zeros((1, ), dtype=torch.long)
            else:
                y = torch.tensor(label) 

            edge_index = torch.tensor(graph["edge_index"]).t()
            edge_attr = torch.tensor(graph["edge_feat"], dtype=attr_type) if "edge_feat" in graph and graph["edge_feat"] is not None else None

            if x is None:
                x_l = None
            else:
                x_l.append(x)
                x_s.append(x_s[-1] + x.shape[0])
            
            if y is None:
                y_l = None
            else:
                y_l.append(y)
                y_s.append(y_s[-1] + y.shape[0])
            
            if edge_index is None:
                edge_index_l = None
            else:
                edge_index_l.append(edge_index)
                e_s.append(e_s[-1] + edge_index.shape[0])
            
            if edge_attr is None:
                edge_attr_l = None
            else:
                edge_attr_l.append(edge_attr)
            
            n_s.append(graph["num_nodes"])
            all_nodes += graph["num_nodes"]
            all_edges += graph["edge_index"].shape[1]

        transform = None
        if name == 'ogbg-code2':
            data = MultiGraphCode(x_l, y_l, edge_index_l, edge_attr_l, x_s, y_s, e_s, n_s)
        else:
            data = MultiGraph(x_l, y_l, edge_index_l, edge_attr_l, x_s, y_s, e_s, n_s)
        split_index = dataset.get_idx_split()

        if name == 'ogbg-code2':
            data.label = label_l

        torch.save([data, all_nodes, all_edges, transform, split_index], self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def num_classes(self):
        return int(self.data[0].y.shape[-1])


class OGBGLSCDataset(Dataset):
    def __init__(self, root, name):
        name = name.replace("-", "_")
        root = os.path.join(root, name)
        self.root, self.name = root, name
        super(OGBGLSCDataset, self).__init__(root)

        self.data, self.all_nodes, self.all_edges, self.transform, self.split_index = torch.load(self.processed_paths[0])
    
    def get_subset(self, subset):
        # datalist = []
        # for idx in subset:
        #     datalist.append(self.data[idx])
        # return datalist
        data = copy.copy(self.data)
        data._indices = subset
        return data

    def get(self, idx):
        return self.data[idx]

    def _download(self):
        pass

    def process(self):
        name = self.name.replace("_", "-")
        if name == 'ogbg-pcqm4m':
            dataset = PCQM4MDataset(root = self.root, smiles2graph = smiles2graph)
        elif name == 'ogbg-pcqm4mv2':
            dataset = PCQM4Mv2Dataset(root = self.root, smiles2graph = smiles2graph)

        all_nodes,  all_edges = 0, 0
        
        x_l, y_l, edge_index_l, edge_attr_l = [], [], [], []
        x_s, y_s, e_s, n_s = [0], [0], [0], []
        
        for i in range(len(dataset.graphs)):
            graph, label = dataset[i]

            x_dtype, attr_type = torch.long, torch.long

            if "node_feat" in graph and graph["node_feat"] is not None:
                x = torch.tensor(graph["node_feat"], dtype=x_dtype) 
            else:
                x = torch.zeros(graph["num_nodes"], dtype=x_dtype)
            y = torch.tensor([label]) if name != 'ogbg-code2' else None
            edge_index = torch.tensor(graph["edge_index"]).t()
            edge_attr = torch.tensor(graph["edge_feat"], dtype=attr_type) if "edge_feat" in graph and graph["edge_feat"] is not None else None

            if x is None:
                x_l = None
            else:
                x_l.append(x)
                x_s.append(x_s[-1] + x.shape[0])
            
            if y is None:
                y_l = None
            else:
                y_l.append(y)
                y_s.append(y_s[-1] + y.shape[0])
            
            if edge_index is None:
                edge_index_l = None
            else:
                edge_index_l.append(edge_index)
                e_s.append(e_s[-1] + edge_index.shape[0])
            
            if edge_attr is None:
                edge_attr_l = None
            else:
                edge_attr_l.append(edge_attr)
            
            n_s.append(graph["num_nodes"])
            all_nodes += graph["num_nodes"]
            all_edges += graph["edge_index"].shape[1]

        transform = None
        data = MultiGraph(x_l, y_l, edge_index_l, edge_attr_l, x_s, y_s, e_s, n_s)       
        split_index = dataset.get_idx_split()

        torch.save([data, all_nodes, all_edges, transform, split_index], self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def num_classes(self):
        return int(self.data[0].y.shape[-1])

class OGBGEvaluator(object):
    def __init__(self, evaluator=None, metric=None, preprocess=None):
        super(OGBGEvaluator, self).__init__()
        self.evaluator = evaluator
        self.metric = metric
        self.preprocess = preprocess
        self.pred = list()
        self.true = list()

    def __call__(self, y_pred, y_true):
        self.pred.append(y_pred)
        self.true.append(y_true)

        return None

    def evaluate(self):
        if len(self.pred) > 0:
            pred = torch.cat(self.pred, dim=0)
            true = torch.cat(self.true, dim=0)
            self.pred = list()
            self.true = list()
            if self.preprocess is not None:
                pred, true = self.preprocess(pred, true)
            if self.metric != 'F1':
                input_dict = {'y_pred': pred, 'y_true': true}
            else:
                input_dict = {'seq_pred': pred, 'seq_ref': true}
            result_dict = self.evaluator.eval(input_dict)
            return result_dict[self.metric]

        warnings.warn("pre-computing list is empty")
        return 0

    def clear(self):
        self.tp = list()
        self.total = list()


# OGB Graph Property Prediction

class OGBMolhivDataset(OGBGDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-molhiv"
        super(OGBMolhivDataset, self).__init__(data_path, dataset)
    
    def get_metric_name(self):
        return 'rocauc'

    def get_evaluator(self):
        name = self.name.replace("_", "-")
        return OGBGEvaluator(evaluator=GraphEvaluator(name), metric='rocauc', preprocess=None)

    def get_loss_fn(self):
        def loss(input, target):
            input = input.to(torch.float32)
            target = target.to(torch.float32)
            return torch.nn.functional.binary_cross_entropy_with_logits(input, target)
        return loss


class OGBMolpcbaDataset(OGBGDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-molpcba"
        super(OGBMolpcbaDataset, self).__init__(data_path, dataset)
    
    def get_metric_name(self):
        return 'ap'
    
    def get_evaluator(self):
        name = self.name.replace("_", "-")
        return OGBGEvaluator(evaluator=GraphEvaluator(name), metric='ap', preprocess=None)

    def get_loss_fn(self):
        def loss(input, target):
            is_labeled = target == target
            input = input[is_labeled].to(torch.float32)
            target = target[is_labeled].to(torch.float32)
            return torch.nn.functional.binary_cross_entropy_with_logits(input, target)
        return loss

class OGBPpaDataset(OGBGDataset):
    def __init__(self):
        dataset = "ogbg-ppa"
        path = "data"
        super(OGBPpaDataset, self).__init__(path, dataset)
    
    def get_metric_name(self):
        return 'acc'
    
    def get_evaluator(self):
        name = self.name.replace("_", "-")
        def preprocess(input, target):
            return torch.argmax(input, dim=1).view(-1,1), target
        return OGBGEvaluator(evaluator=GraphEvaluator(name), metric='acc', preprocess=preprocess)

    def get_loss_fn(self):
        def loss(input, target):
            input = input.to(torch.float32)
            target = target.view(-1)
            return torch.nn.functional.cross_entropy(input, target)
        return loss
    
    @property
    def num_classes(self):
        return int(self.data._y.max()+1)


class OGBCodeDataset(OGBGDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-code2"
        super(OGBCodeDataset, self).__init__(data_path, dataset)
        
        self.generate_y()
    
    def generate_y(self, num_vocab=5000, max_seq_len=5):
        seq_list = self.data.label
        vocab_cnt = {}
        vocab_list = []
        for seq in seq_list:
            for w in seq:
                if w in vocab_cnt:
                    vocab_cnt[w] += 1
                else:
                    vocab_cnt[w] = 1
                    vocab_list.append(w)

        cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
        topvocab = np.argsort(-cnt_list, kind = 'stable')[:num_vocab]

        print('Coverage of top {} vocabulary:'.format(num_vocab))
        print(float(np.sum(cnt_list[topvocab]))/np.sum(cnt_list))

        vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
        idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

        vocab2idx['__UNK__'] = num_vocab
        idx2vocab.append('__UNK__')

        vocab2idx['__EOS__'] = num_vocab + 1
        idx2vocab.append('__EOS__')

        for idx, vocab in enumerate(idx2vocab):
            assert(idx == vocab2idx[vocab])

        # test that the idx of '__EOS__' is len(idx2vocab) - 1.
        # This fact will be used in decode_arr_to_seq, when finding __EOS__
        assert(vocab2idx['__EOS__'] == len(idx2vocab) - 1)

        self.vocab2idx, self.idx2vocab, self.num_vocab, self.max_seq_len = vocab2idx, idx2vocab, num_vocab, max_seq_len

        ys = []
        for seq in seq_list:
            y = []
            for i in range(max_seq_len):
                w = seq[i] if i < len(seq) else '__EOS__'
                v = vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__']
                y.append(v)
            ys.append(y)

        self.data._y = torch.tensor(ys)
        
        self.train_label = [self.data.label[i] for i in self.split_index["train"]]
        self.valid_label = [self.data.label[i] for i in self.split_index["valid"]]
        self.test_label = [self.data.label[i] for i in self.split_index["test"]]

    def get_metric_name(self):
        return 'F1'
    
    def get_evaluator(self):
        name = self.name.replace("_", "-")
        def preprocess(input, target, idx2vocab, train_label, valid_label, test_label):
            input = torch.argmax(input, dim=1).view(-1, self.max_seq_len).cpu().numpy().tolist()
            preds = []
            for items in input:
                pred = []
                for item in items:
                    if item == self.num_vocab + 1:
                        break
                    else:
                        pred.append(idx2vocab[item])
                preds.append(pred)
                        
            if len(preds) == len(train_label):
                label = train_label
            elif len(preds) == len(valid_label):
                label = valid_label
            elif len(preds) == len(test_label):
                label = test_label
            else:
                label = None

            return preds, label
        return OGBGEvaluator(evaluator=GraphEvaluator(name), metric='F1', preprocess=lambda x,y : preprocess(x, y, self.idx2vocab, self.train_label, self.valid_label, self.test_label))

    def get_loss_fn(self):
        def loss(input, target):
            input = input.to(torch.float32)
            target = target.view(-1)
            return torch.nn.functional.cross_entropy(input, target)
        return loss
    
    @property
    def num_classes(self):
        return len(self.vocab2idx)

# OGB Large-Scale Challenge 

class OGBPCQM4MDataset(OGBGLSCDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-pcqm4m"
        super(OGBPCQM4MDataset, self).__init__(data_path, dataset)
    
    def get_metric_name(self):
        return 'mae'
    
    def get_evaluator(self):
        def preprocess(input, target):
            return input.view(-1), target.view(-1)
        return OGBGEvaluator(evaluator=PCQM4MEvaluator(), metric='mae', preprocess=preprocess)

    def get_loss_fn(self):
        def loss(input, target):
            input = input.to(torch.float32)
            target = target.to(torch.float32)
            return torch.nn.functional.l1_loss(input, target)
        return loss


class OGBPCQM4Mv2Dataset(OGBGLSCDataset):
    def __init__(self, data_path="data"):
        dataset = "ogbg-pcqm4mv2"
        super(OGBPCQM4Mv2Dataset, self).__init__(data_path, dataset)
        self.split_index['test'] = self.split_index['test-dev']
    
    def get_metric_name(self):
        return 'mae'
    
    def get_evaluator(self):
        def preprocess(input, target):
            return input.view(-1), target.view(-1)
        return OGBGEvaluator(evaluator=PCQM4Mv2Evaluator(), metric='mae', preprocess=preprocess)

    def get_loss_fn(self):
        def loss(input, target):
            input = input.to(torch.float32)
            target = target.to(torch.float32)
            return torch.nn.functional.l1_loss(input, target)
        return loss
