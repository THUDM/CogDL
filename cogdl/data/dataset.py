import collections
import copy
import os.path as osp
from itertools import repeat, product

import torch.utils.data

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
        raise NotImplementedError

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
        return self[0].num_features

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
        data = self.get(idx)
        data = data if self.transform is None else self.transform(data)
        return data

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        y = self.data.y
        return y.max().item() + 1 if y.dim() == 1 else y.size(1)

    def __repr__(self):  # pragma: no cover
        return "{}({})".format(self.__class__.__name__, len(self))


class MultiGraphDataset(Dataset):
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        super(MultiGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = None, None

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        y = self.data.y
        return y.max().item() + 1 if y.dim() == 1 else y.size(1)

    def len(self):
        for item in self.slices.values():
            return len(item) - 1
        return 0

    def _get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, "__num_nodes__"):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
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
        if isinstance(idx, int) or (len(idx) == 0):
            return self._get(idx)
        elif len(idx) > 1:
            data_list = [self._get(i) for i in idx]
            data, slices = self.from_data_list(data_list)
            dataset = copy.copy(self)
            dataset.data = data
            dataset.slices = slices
            return dataset

    @staticmethod
    def from_data_list(data_list):
        r""" Borrowed from PyG"""

        keys = data_list[0].keys
        data = data_list[0].__class__()

        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            data[key].append(item[key])
            if torch.is_tensor(item[key]):
                s = slices[key][-1] + item[key].size(item.__cat_dim__(key, item[key]))
            else:
                s = slices[key][-1] + 1
            slices[key].append(s)

        if hasattr(data_list[0], "__num_nodes__"):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            item = data_list[0][key]
            if torch.is_tensor(item):
                data[key] = torch.cat(data[key], dim=data.__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                data[key] = torch.tensor(data[key])

            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slices

    def __len__(self):
        for item in self.slices.values():
            return len(item) - 1
        return 0
