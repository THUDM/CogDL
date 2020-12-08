import torch

from cogdl.datasets import register_dataset
from cogdl.data import Dataset, Data

@register_dataset("test_small")
class TestSmallDataset(Dataset):
    r"""small dataset for debug"""
    def __init__(self):
        x = torch.FloatTensor([[-2, -1], [-2, 1], [-1, 0], [0, 0], [0, 1], [1, 0], [2, 1], [3, 0], [2, -1]])
        edge_index = torch.LongTensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8],
                                       [1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 3, 6, 7, 8, 5, 7, 5, 6, 8, 5, 7]])
        y = torch.LongTensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        self.data = Data(x, edge_index, None, y, None)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data