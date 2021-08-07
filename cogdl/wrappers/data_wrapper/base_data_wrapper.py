from torch.utils.data import DataLoader
from cogdl.data import Graph


class DataWrapper(object):
    @staticmethod
    def add_args(parser):
        pass

    def __init__(self, dataset=None):
        if dataset is not None:
            if hasattr(dataset, "get_loss_fn"):
                self.__loss_fn__ = dataset.get_loss_fn()
            if hasattr(dataset, "get_evaluator"):
                self.__evaluator__ = dataset.get_evaluator()
        else:
            self.__loss_fn__ = None
            self.__evaluator__ = None
        self.__dataset__ = dataset

    def get_default_loss_fn(self):
        return self.__loss_fn__

    def get_default_evaluator(self):
        return self.__evaluator__

    def training_wrapper(self):
        pass

    def val_wrapper(self):
        pass

    def test_wrapper(self):
        pass

    def transform(self, batch):
        return batch

    def pre_transform(self):
        pass

    @staticmethod
    def __batch_wrapper__(loader):
        if loader is None:
            return None
        elif isinstance(loader, DataLoader):
            return loader
        else:
            return [loader]

    def on_training_wrapper(self):
        train_loader = self.training_wrapper()
        return self.__batch_wrapper__(train_loader)

    def on_val_wrapper(self):
        val_loader = self.val_wrapper()
        return self.__batch_wrapper__(val_loader)

    def on_test_wrapper(self):
        val_loader = self.test_wrapper()
        return self.__batch_wrapper__(val_loader)

    def on_transform(self, batch):
        return self.transform(batch)

    def train(self):
        if self.__dataset__ is not None and isinstance(self.__dataset__.data, Graph):
            self.__dataset__.data.train()

    def eval(self):
        if self.__dataset__ is not None and isinstance(self.__dataset__.data, Graph):
            self.__dataset__.data.eval()
