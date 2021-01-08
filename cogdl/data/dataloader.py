import torch.utils.data
from torch.utils.data.dataloader import default_collate

from cogdl.data import Batch


class DataLoader(torch.utils.data.DataLoader):
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
        super(DataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=lambda data_list: Batch.from_data_list(data_list), **kwargs
        )
