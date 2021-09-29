import torch
from cogdl.data import DataLoader

from cogdl.wrappers.data_wrapper.base_data_wrapper import OnLoadingWrapper


class DataController(object):
    def __init__(self, world_size: int = 1, distributed: bool = False):
        self.world_size = world_size
        self.distributed = distributed

    def distributed_dataloader(self, dataloader: DataLoader, dataset, rank):
        # TODO: just a toy implementation
        assert isinstance(dataloader, DataLoader)

        args, kwargs = dataloader.get_parameters()
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
        kwargs["sampler"] = sampler
        dataloader = dataloader.__class__(*args, **kwargs)
        return dataloader

    def prepare_data_wrapper(self, dataset_w, rank=0):
        if self.distributed:
            dataset_w.pre_transform()
            train_loader = dataset_w.train_wrapper()
            assert isinstance(train_loader, DataLoader)
            train_loader = self.distributed_dataloader(train_loader, dataset=dataset_w.get_train_dataset(), rank=rank)
            train_wrapper = OnLoadingWrapper(train_loader, dataset_w.train_transform)
            dataset_w.prepare_val_data()
            dataset_w.prepare_test_data()
            dataset_w.set_train_data(train_wrapper)
            return dataset_w
        else:
            dataset_w.pre_transform()
            dataset_w.prepare_training_data()
            dataset_w.prepare_val_data()
            dataset_w.prepare_test_data()
            return dataset_w

    def training_proc_per_stage(self, dataset_w, rank=0):
        if dataset_w.__refresh_per_epoch__():
            if self.distributed:
                train_loader = dataset_w.train_wrapper()
                assert isinstance(train_loader, DataLoader)
                train_loader = self.distributed_dataloader(train_loader, dataset=dataset_w.get_dataset(), rank=rank)
                train_wrapper = OnLoadingWrapper(train_loader, dataset_w.train_transform)
                dataset_w.__train_data = train_wrapper
            else:
                dataset_w.prepare_training_data()
        return dataset_w
