from typing import Optional
from torch.utils.data import DataLoader
from cogdl.wrappers.data_wrapper import DataWrapper


class DataController(object):
    def __init__(self):
        self.data_w: Optional[DataWrapper] = None

    def train_loader(self):
        pass

    def val_loader(self):
        pass

    def test_loader(self):
        pass
