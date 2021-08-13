import collections
import os.path as osp
from itertools import repeat

import torch.utils.data

from cogdl.data import Adjacency, Graph
from cogdl.utils import makedirs
from cogdl.utils import accuracy, cross_entropy_loss


def to_list(x):
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x]
    return x


def files_exist(files):
    return all([osp.exists(f) for f in files])


class Dataset(torch.utils.data.Dataset):
    r"""Dataset base class for creating graph datasets.
    See `here <https://rusty1s.github.io/pycogdl/build/html/notes/
    create_dataset.html>`__ for the accompanying tutorial.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`cogdl.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`cogdl.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`cogdl.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    @staticmethod
    def add_args(parser):
        """Add dataset-specific arguments to the parser."""
        pass

    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        raise NotImplementedError

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        raise NotImplementedError

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def __len__(self):
        r"""The number of examples in the dataset."""
        return 1

    def get(self, idx):
        r"""Gets the data object at index :obj:`idx`."""
        raise NotImplementedError

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(Dataset, self).__init__()

        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, "raw")
        self.processed_dir = osp.join(self.root, "processed")
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self._download()
        self._process()

    @property
    def num_features(self):
        r"""Returns the number of features per node in the graph."""
        if hasattr(self, "data") and isinstance(self.data, Graph):
            return self.data.num_features
        else:
            return 0

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def _process(self):
        if files_exist(self.processed_paths):  # pragma: no cover
            return

        print("Processing...")

        makedirs(self.processed_dir)
        self.process()

        print("Done!")

    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss

    def __getitem__(self, idx):  # pragma: no cover
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given)."""
        assert idx == 0
        data = self.data
        data = data if self.transform is None else self.transform(data)
        return data

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        if hasattr(self, "y") and self.y is not None:
            y = self.y
        elif hasattr(self, "data") and hasattr(self.data, "y") and self.data.y is not None:
            y = self.data.y
        else:
            return 0
        return y.max().item() + 1 if y.dim() == 1 else y.size(1)

    @property
    def edge_attr_size(self):
        return None

    def __repr__(self):  # pragma: no cover
        return "{}({})".format(self.__class__.__name__, len(self))


class MultiGraphDataset(Dataset):
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        super(MultiGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = None, None

    @property
    def num_classes(self):
        if hasattr(self, "y"):
            y = self.y
        elif hasattr(self, "data") and hasattr(self.data[0], "y"):
            y = torch.cat([x.y for x in self.data], dim=0)
        else:
            return 0
        return y.max().item() + 1 if y.dim() == 1 else y.size(1)

    @property
    def num_features(self):
        if isinstance(self[0], Graph):
            return self[0].num_features
        else:
            return 0

    def len(self):
        if isinstance(self.data, list):
            return len(self.data)
        else:
            for item in self.slices.values():
                return len(item) - 1
            return 0

    def _get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, "__num_nodes__"):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.__old_keys__():
            item, slices = self.data[key], self.slices[key]
            # start, end = slices[idx].item(), slices[idx + 1].item()
            start, end = int(slices[idx]), int(slices[idx + 1])
            if key == "edge_index":
                data[key] = (item[0][start:end], item[1][start:end])
            else:
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[self.data.__cat_dim__(key, item)] = slice(start, end)
                elif start + 1 == end:
                    s = slices[start]
                else:
                    s = slice(start, end)
                data[key] = item[s]
        return data

    def get(self, idx):
        try:
            idx = int(idx)
        except Exception:
            idx = idx
        if torch.is_tensor(idx):
            idx = idx.numpy().tolist()
        if isinstance(idx, int):
            if self.slices is not None:
                return self._get(idx)
            return self.data[idx]
        if isinstance(idx, slice):
            start = idx.start
            end = idx.stop
            step = idx.step
            idx = list(range(start, end, step))

        if len(idx) > 1:
            # unsupport `slice`
            if self.slices is not None:
                return [self._get(int(i)) for i in idx]
            return [self.data[i] for i in idx]

    def __getitem__(self, item):
        return self.get(item)

    @staticmethod
    def from_data_list(data_list):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = Graph()
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        num_nodes_cum = [0]
        num_nodes = None
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.cat_dim(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                cumsum[key] = cumsum[key] + data.__inc__(key, item)
                batch[key].append(item)

                # if key in follow_batch:
                #     item = torch.full((size,), i, dtype=torch.long)
                #     batch["{}_batch".format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                num_nodes_cum.append(num_nodes + num_nodes_cum[-1])
                item = torch.full((num_nodes,), i, dtype=torch.long)
                batch.batch.append(item)
        if num_nodes is None:
            batch.batch = None
        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key], dim=data_list[0].cat_dim(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            elif isinstance(item, Adjacency):
                target = Adjacency()
                for k in item.keys:
                    if k == "row" or k == "col":
                        _item = torch.cat(
                            [x[k] + num_nodes_cum[i] for i, x in enumerate(batch[key])], dim=item.cat_dim(k, None)
                        )
                    elif k == "row_ptr":
                        _item = torch.cat(
                            [x[k][:-1] + num_nodes_cum[i] for i, x in enumerate(batch[key][:-1])],
                            dim=item.cat_dim(k, None),
                        )
                        _item = torch.cat([_item, batch[key][-1] + num_nodes_cum[-1]], dim=item.cat_dim(k, None))
                    else:
                        _item = torch.cat([x[k] for i, x in enumerate(batch[key])], dim=item.cat_dim(k, None))
                    target[k] = _item
                batch[key] = target.to(item.device)
        return batch.contiguous()

    def __len__(self):
        return len(self.data)
