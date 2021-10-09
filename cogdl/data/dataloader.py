from abc import ABCMeta
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from cogdl.data import Batch, Graph

try:
    from typing import GenericMeta  # python 3.6
except ImportError:
    # in 3.7, genericmeta doesn't exist but we don't need it
    class GenericMeta(type):
        pass


class RecordParameters(ABCMeta):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.record_parameters([args, kwargs])
        return obj


class GenericRecordParameters(GenericMeta, RecordParameters):
    pass


class DataLoader(torch.utils.data.DataLoader, metaclass=GenericRecordParameters):
    r"""Data loader which merges data objects from a
    :class:`cogdl.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        if "collate_fn" not in kwargs or kwargs["collate_fn"] is None:
            kwargs["collate_fn"] = self.collate_fn

        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            **kwargs,
        )

    @staticmethod
    def collate_fn(batch):
        item = batch[0]
        if isinstance(item, Graph):
            return Batch.from_data_list(batch)
        elif isinstance(item, torch.Tensor):
            return default_collate(batch)
        elif isinstance(item, float):
            return torch.tensor(batch, dtype=torch.float)

        raise TypeError("DataLoader found invalid type: {}".format(type(item)))

    def get_parameters(self):
        return self.default_kwargs

    def record_parameters(self, params):
        self.default_kwargs = params
