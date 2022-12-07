import numpy as np
import torch
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
        self.__training_data, self.__val_data, self.__test_data = None, None, None
        self.__num_training_data, self.__num_val_data, self.__num_test_data = 0, 0, 0
        self.__prepare_dataloader_per_epoch__ = False
        self.__back_to_cpu__ = False

    @property
    def data_back_to_cpu(self):
        return (
            self.__val_data is not None
            and self.__test_data is not None
            and (isinstance(self.__val_data.raw_data, Graph) or isinstance(self.__test_data.raw_data, Graph))
            and not isinstance(self.__training_data.raw_data, Graph)
        )

    def get_train_dataset(self):
        """
        Return the `wrapped` dataset for specific usage.
        For example, return `ClusteredDataset` in cluster_dw for DDP training.
        """
        raise NotImplementedError

    def get_val_dataset(self):
        """
        Similar to `self.get_train_dataset` but for validation.
        """
        raise NotImplementedError

    def get_test_dataset(self):
        """
        Similar to `self.get_train_dataset` but for test.
        """
        raise NotImplementedError

    def train_wrapper(self):
        """
        Return:
            1. DataLoader
            2. cogdl.Graph
            3. list of DataLoader or Graph
        Any other data formats other than DataLoader will not be traversed
        """
        pass

    def val_wrapper(self):
        pass

    def test_wrapper(self):
        pass

    def evaluation_wrapper(self):
        if self.__dataset__ is None:
            self.__dataset__ = getattr(self, "dataset", None)
        if self.__dataset__ is not None:
            return self.__dataset__

    def train_transform(self, batch):
        return batch

    def val_transform(self, batch):
        return batch

    def test_transform(self, batch):
        return batch

    def pre_transform(self):
        """Data Preprocessing before all runs"""
        pass

    def pre_stage(self, stage, model_w_out):
        """Processing before each run"""
        pass

    def post_stage(self, stage, model_w_out):
        """Processing after each run"""
        pass

    def refresh_per_epoch(self, name="train"):
        self.__prepare_dataloader_per_epoch__ = True

    def __refresh_per_epoch__(self):
        return self.__prepare_dataloader_per_epoch__

    def get_default_loss_fn(self):
        return self.__loss_fn__

    def get_default_evaluator(self):
        return self.__evaluator__

    def get_dataset(self):
        if self.__dataset__ is None:
            self.__dataset__ = getattr(self, "dataset", None)
        return self.__dataset__

    def prepare_training_data(self):
        train_data = self.train_wrapper()
        if train_data is not None:
            self.__training_data = OnLoadingWrapper(train_data, self.train_transform)

    def prepare_val_data(self):
        val_data = self.val_wrapper()
        if val_data is not None:
            self.__val_data = OnLoadingWrapper(val_data, self.val_transform)

    def prepare_test_data(self):
        test_data = self.test_wrapper()
        if test_data is not None:
            self.__test_data = OnLoadingWrapper(test_data, self.test_transform)

    def set_train_data(self, x):
        self.__training_data = x

    def set_val_data(self, x):
        self.__val_data = x

    def set_test_data(self, x):
        self.__test_data = x

    def on_train_wrapper(self):
        if self.__training_data is None:
            return None

        if self.__prepare_dataloader_per_epoch__:
            # TODO: reserve parameters for `prepare training data`
            self.prepare_training_data()
        return self.__training_data

    def on_val_wrapper(self):
        return self.__val_data

    def on_test_wrapper(self):
        return self.__test_data

    def train(self):
        if self.__dataset__ is None:
            self.__dataset__ = getattr(self, "dataset", None)
        if self.__dataset__ is not None and \
                (isinstance(self.__dataset__.data, Graph) 
                    or hasattr(self.__dataset__.data, "graphs")):
            self.__dataset__.data.train()

    def eval(self):
        if self.__dataset__ is None:
            self.__dataset__ = getattr(self, "dataset", None)
        if self.__dataset__ is not None and \
                (isinstance(self.__dataset__.data, Graph) 
                    or hasattr(self.__dataset__.data, "graphs")):
            self.__dataset__.data.eval()


class OnLoadingWrapper(object):
    def __init__(self, data, transform):
        """
        Args:
            data: `data` or `dataset`, that it, `cogdl.Graph` or `DataLoader`
        """
        self.raw_data = data
        self.data = self.__process_iterative_data__(data)
        self.__num_training_data = self.__get_min_len__(self.data)
        self.wrapped_data = self.__wrap_iteration__(self.data)
        self.ptr = 0
        self.transform = transform

    def __next__(self):
        if self.ptr < self.__num_training_data:
            self.ptr += 1
            batch = self.__next_batch__(self.wrapped_data)
            return self.transform(batch)
        else:
            self.ptr = 0
            # re-wrap the dataset per epoch
            self.wrapped_data = self.__wrap_iteration__(self.data)
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.__num_training_data

    def get_dataset_from_loader(self):
        return self.raw_data

    def __wrap_iteration__(self, inputs):
        # if isinstance(inputs, tuple):
        #     inputs = list(inputs)
        def iter_func(in_x):
            if isinstance(in_x, list) or isinstance(in_x, DataLoader):
                for item in in_x:
                    yield item
            else:
                yield in_x

        if isinstance(inputs, list):
            outputs = [None] * len(inputs)
            for i, item in enumerate(inputs):
                outputs[i] = self.__wrap_iteration__(item)
        elif isinstance(inputs, dict):
            outputs = {key: None for key in inputs.keys()}
            for key, val in inputs.items():
                outputs[key] = self.__wrap_iteration__(val)
        else:
            # return LoaderWrapper(inputs)
            return iter_func(inputs)
        return outputs

    def __process_iterative_data__(self, inputs):
        if inputs is None:
            return None
        # if isinstance(inputs, tuple):
        #     inputs = list(inputs)

        if isinstance(inputs, list):
            for i, item in enumerate(inputs):
                inputs[i] = self.__process_iterative_data__(item)
        elif isinstance(inputs, dict):
            for key, val in inputs.items():
                inputs[key] = self.__process_iterative_data__(val)
        else:
            # return self.__batch_wrapper__(inputs)
            return inputs
        return inputs

    def __next_batch__(self, inputs):
        # if isinstance(inputs, tuple):
        #     inputs = list(inputs)

        if isinstance(inputs, list):
            outputs = [None] * len(inputs)
            for i, item in enumerate(inputs):
                outputs[i] = self.__next_batch__(item)
        elif isinstance(inputs, dict):
            outputs = {key: None for key in inputs.keys()}
            for key, val in inputs.items():
                outputs[key] = self.__next_batch__(val)
        else:
            return next(inputs)
        return outputs

    def __get_min_len__(self, inputs):
        if inputs is None:
            return None

        # if isinstance(inputs, tuple):
        #     inputs = list(inputs)
        if isinstance(inputs, list):
            outputs = [0] * len(inputs)
            for i, item in enumerate(inputs):
                outputs[i] = self.__get_min_len__(item)
            return np.min(outputs)
        # elif isinstance(inputs, dict):
        #     outputs = {key: 0 for key in inputs.keys()}
        #     for i, val in enumerate(inputs.values()):
        #         outputs[i] = self.__get_min_len__(val)
        #     return np.min(list(outputs.values()))
        else:
            if isinstance(inputs, DataLoader):
                return len(inputs)
            else:
                return 1
