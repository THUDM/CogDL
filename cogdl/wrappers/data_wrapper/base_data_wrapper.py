import numpy as np
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

    def refresh_per_epoch(self, name="train"):
        self.__prepare_dataloader_per_epoch__ = True

    def get_default_loss_fn(self):
        return self.__loss_fn__

    def get_default_evaluator(self):
        return self.__evaluator__

    def get_dataset(self):
        return self.__dataset__

    def training_wrapper(self):
        """
        Return: DataLoader or cogdl.Graph
        """
        pass

    def val_wrapper(self):
        pass

    def test_wrapper(self):
        pass

    def evaluation_wrapper(self):
        if self.__dataset__ is not None:
            return self.__dataset__

    def train_transform(self, batch):
        return batch

    def val_transform(self, batch):
        return batch

    def test_transform(self, batch):
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

    def __wrap_iteration__(self, inputs):
        def iterative_func(in_x):
            if isinstance(in_x, list) or isinstance(in_x, DataLoader):  # or isinstance(in_x, tuple)
                for x in in_x:
                    yield x
            else:
                yield in_x

        # if isinstance(inputs, tuple):
        #     inputs = list(inputs)

        if isinstance(inputs, list):
            outputs = [None] * len(inputs)
            for i, item in enumerate(inputs):
                outputs[i] = self.__wrap_iteration__(item)
        elif isinstance(inputs, dict):
            outputs = {key: None for key in inputs.keys()}
            for key, val in inputs.items():
                outputs[key] = self.__wrap_iteration__(val)
        else:
            return iterative_func(inputs)
        return outputs

    def __get_min_len__(self, inputs):
        if inputs is None:
            return None

        if isinstance(inputs, tuple):
            inputs = list(inputs)
        if isinstance(inputs, list):
            outputs = [0] * len(inputs)
            for i, item in enumerate(inputs):
                inputs[i] = self.__get_min_len__(item)
            return np.min(outputs)
        elif isinstance(inputs, dict):
            outputs = {key: 0 for key in inputs.keys()}
            for i, val in enumerate(inputs.values()):
                outputs[i] = self.__get_min_len__(val)
            return np.min(outputs)
        else:
            if isinstance(inputs, DataLoader):
                return len(inputs)
            else:
                return 1

    def prepare_dataloader(self, name, distributed=False, rank=0, world_size=0):
        pass

    def prepare_training_data(self, distributed=False, rank=0, world_size=0):
        self.__training_data = self.__process_iterative_data__(self.training_wrapper())
        self.__num_training_data = self.__get_min_len__(self.__training_data)

    def prepare_val_data(self, distributed=False, rank=0, world_size=0):
        self.__val_data = self.__process_iterative_data__(self.val_wrapper())
        self.__num_val_data = self.__get_min_len__(self.__val_data)

    def prepare_test_data(self, distributed=False, rank=0, world_size=0):
        self.__test_data = self.__process_iterative_data__(self.test_wrapper())
        self.__num_test_data = self.__get_min_len__(self.__test_data)

    def on_training_wrapper(self):
        if self.__training_data is None:
            return None

        if self.__prepare_dataloader_per_epoch__:
            # TODO: reserve parameters for `prepare training data`
            self.prepare_training_data()

        train_loader = self.__wrap_iteration__(self.__training_data)
        for _ in range(self.__num_training_data):
            batch = self.__next_batch__(train_loader)
            yield self.train_transform(batch)

    def on_val_wrapper(self):
        if self.__val_data is None:
            return None

        val_loader = self.__wrap_iteration__(self.__val_data)
        for _ in range(self.__num_val_data):
            batch = self.__next_batch__(val_loader)
            yield self.val_transform(batch)

    def on_test_wrapper(self):
        if self.__test_data is None:
            return None

        test_loader = self.__wrap_iteration__(self.__test_data)
        for _ in range(self.__num_test_data):
            batch = self.__next_batch__(test_loader)
            yield self.test_transform(batch)

    def pre_stage(self, stage, model_w_out):
        pass

    def post_stage(self, stage, model_w_out):
        pass

    def train(self):
        if self.__dataset__ is not None and isinstance(self.__dataset__.data, Graph):
            self.__dataset__.data.train()

    def eval(self):
        if self.__dataset__ is not None and isinstance(self.__dataset__.data, Graph):
            self.__dataset__.data.eval()
